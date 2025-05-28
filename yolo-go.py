import os
import sys
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("Warning: RPi.GPIO not available. Running in simulation mode.")
    GPIO_AVAILABLE = False

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.4)
parser.add_argument('--resolution', default="320x240")
parser.add_argument('--record', action='store_true')
parser.add_argument('--robot', action='store_true')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record
robot_mode = args.robot

TARGET_OBJECTS = ['bottle', 'paper', 'plant']
PROXIMITY_THRESHOLD = 0.35
CENTER_TOLERANCE = 60
TURN_SPEED = 0.8
FORWARD_SPEED = 0.6
MIN_DETECTION_SIZE = 0.08
INFERENCE_SIZE = 416

class OptimizedRaspberryPiRobotController:
    def __init__(self):
        self.LEFT_MOTOR_PIN1 = 18
        self.LEFT_MOTOR_PIN2 = 19
        self.LEFT_MOTOR_PWM = 12
        self.RIGHT_MOTOR_PIN1 = 20
        self.RIGHT_MOTOR_PIN2 = 21
        self.RIGHT_MOTOR_PWM = 13
        self.is_moving = False
        self.current_target = None
        self.target_reached = False
        if GPIO_AVAILABLE:
            self.setup_gpio()

    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        pins = [self.LEFT_MOTOR_PIN1, self.LEFT_MOTOR_PIN2, 
                self.RIGHT_MOTOR_PIN1, self.RIGHT_MOTOR_PIN2]
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        GPIO.setup(self.LEFT_MOTOR_PWM, GPIO.OUT)
        GPIO.setup(self.RIGHT_MOTOR_PWM, GPIO.OUT)
        self.left_pwm = GPIO.PWM(self.LEFT_MOTOR_PWM, 1000)
        self.right_pwm = GPIO.PWM(self.RIGHT_MOTOR_PWM, 1000)
        self.left_pwm.start(0)
        self.right_pwm.start(0)

    def _execute_movement(self, command, speed):
        if not GPIO_AVAILABLE:
            return
        if command == 'stop':
            GPIO.output(self.LEFT_MOTOR_PIN1, GPIO.LOW)
            GPIO.output(self.LEFT_MOTOR_PIN2, GPIO.LOW)
            GPIO.output(self.RIGHT_MOTOR_PIN1, GPIO.LOW)
            GPIO.output(self.RIGHT_MOTOR_PIN2, GPIO.LOW)
            self.left_pwm.ChangeDutyCycle(0)
            self.right_pwm.ChangeDutyCycle(0)
        elif command == 'forward':
            GPIO.output(self.LEFT_MOTOR_PIN1, GPIO.HIGH)
            GPIO.output(self.LEFT_MOTOR_PIN2, GPIO.LOW)
            GPIO.output(self.RIGHT_MOTOR_PIN1, GPIO.HIGH)
            GPIO.output(self.RIGHT_MOTOR_PIN2, GPIO.LOW)
            duty_cycle = int(speed * 100)
            self.left_pwm.ChangeDutyCycle(duty_cycle)
            self.right_pwm.ChangeDutyCycle(duty_cycle)
        elif command == 'left':
            GPIO.output(self.LEFT_MOTOR_PIN1, GPIO.LOW)
            GPIO.output(self.LEFT_MOTOR_PIN2, GPIO.HIGH)
            GPIO.output(self.RIGHT_MOTOR_PIN1, GPIO.HIGH)
            GPIO.output(self.RIGHT_MOTOR_PIN2, GPIO.LOW)
            duty_cycle = int(speed * 100)
            self.left_pwm.ChangeDutyCycle(duty_cycle)
            self.right_pwm.ChangeDutyCycle(duty_cycle)
        elif command == 'right':
            GPIO.output(self.LEFT_MOTOR_PIN1, GPIO.HIGH)
            GPIO.output(self.LEFT_MOTOR_PIN2, GPIO.LOW)
            GPIO.output(self.RIGHT_MOTOR_PIN1, GPIO.LOW)
            GPIO.output(self.RIGHT_MOTOR_PIN2, GPIO.HIGH)
            duty_cycle = int(speed * 100)
            self.left_pwm.ChangeDutyCycle(duty_cycle)
            self.right_pwm.ChangeDutyCycle(duty_cycle)

    def stop(self):
        self._execute_movement('stop', 0)
        self.is_moving = False

    def move_forward(self, speed=FORWARD_SPEED):
        self._execute_movement('forward', speed)
        self.is_moving = True

    def turn_left(self, speed=TURN_SPEED):
        self._execute_movement('left', speed)
        self.is_moving = True

    def turn_right(self, speed=TURN_SPEED):
        self._execute_movement('right', speed)
        self.is_moving = True

    def navigate_to_object(self, bbox, frame_width, class_name):
        xmin, _, xmax, _ = bbox
        obj_center_x = (xmin + xmax) >> 1
        obj_width = xmax - xmin
        frame_center_x = frame_width >> 1
        obj_size_ratio = obj_width / frame_width

        if obj_size_ratio > PROXIMITY_THRESHOLD:
            self.stop()
            self.target_reached = True
            print(f"Target {class_name} reached!")
            return True
        if obj_size_ratio < MIN_DETECTION_SIZE:
            return False

        horizontal_offset = obj_center_x - frame_center_x
        if abs(horizontal_offset) > CENTER_TOLERANCE:
            if horizontal_offset > 0:
                self.turn_right(TURN_SPEED)
            else:
                self.turn_left(TURN_SPEED)
        else:
            self.move_forward(FORWARD_SPEED)
        return False

    def cleanup(self):
        if GPIO_AVAILABLE:
            self.stop()
            self.left_pwm.stop()
            self.right_pwm.stop()
            GPIO.cleanup()

if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

print("Loading YOLO model...")
model = YOLO(model_path, task='detect')
model.model.eval()
if hasattr(model.model, 'float'):
    model.model.half()
    print("Using half precision for faster inference")
labels = model.names
print(f"Model loaded. Detected classes: {len(labels)}")

robot = None
if robot_mode:
    robot = OptimizedRaspberryPiRobotController()
    print("Robot navigation mode enabled")

width, height = map(int, user_res.split('x'))
cap = cv2.VideoCapture(0 if img_source.isdigit() else img_source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

out = None
if record:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

print("Starting detection loop...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received from camera.")
            break

        results = model(frame, imgsz=INFERENCE_SIZE, conf=min_thresh)[0]
        annotated_frame = frame.copy()

        target_found = False
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = labels[cls_id]
            if class_name in TARGET_OBJECTS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {box.conf[0]:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if robot_mode and robot and not robot.target_reached:
                    robot.navigate_to_object((x1, y1, x2, y2), frame.shape[1], class_name)
                    target_found = True
                    break  # prioritize first target

        if not target_found and robot_mode and robot:
            robot.stop()

        cv2.imshow("Detection", annotated_frame)
        if record and out:
            out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    if robot:
        robot.cleanup()
