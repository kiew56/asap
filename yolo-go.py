import os
import sys
import argparse
import cv2
import numpy as np
import time
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
            print(f"Simulating: {command} at speed {speed}")
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

    def stop(self):
        self._execute_movement('stop', 0)
        self.is_moving = False

    def move_forward(self, speed=FORWARD_SPEED):
        self._execute_movement('forward', speed)
        self.is_moving = True

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
    print("Robot basic movement mode enabled")
    print("Moving forward for 3 seconds...")
    robot.move_forward()
    time.sleep(3)
    robot.stop()
    print("Stopped moving")
    robot.cleanup()
