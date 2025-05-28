import os
import sys
import argparse
import glob
import time
import math
import threading
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# Raspberry Pi GPIO imports
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("Warning: RPi.GPIO not available. Running in simulation mode.")
    GPIO_AVAILABLE = False

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.4)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "320x240"), \
                    otherwise, match source resolution. Lower resolution = faster processing',
                    default="320x240")
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--robot', help='Enable robot navigation mode',
                    action='store_true')
parser.add_argument('--skip-frames', help='Process every Nth frame for detection (1=every frame, 2=every other frame, etc.)',
                    type=int, default=2)

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record
robot_mode = args.robot
skip_frames = args.skip_frames

# Optimized navigation parameters
TARGET_OBJECTS = ['bottle', 'paper', 'plant']  # Objects to navigate toward
PROXIMITY_THRESHOLD = 0.35  # Stop when object occupies 35% of frame width
CENTER_TOLERANCE = 60  # Reduced pixel tolerance for faster response
TURN_SPEED = 0.8  # Increased turning speed
FORWARD_SPEED = 0.6  # Forward movement speed
MIN_DETECTION_SIZE = 0.08  # Minimum object size to consider
INFERENCE_SIZE = 416  # Smaller inference size for speed (was 640 default)

class OptimizedRaspberryPiRobotController:
    """
    Optimized Raspberry Pi Robot Controller with threaded movement
    """
    def __init__(self):
        # Motor GPIO pins - adjust these based on your wiring
        self.LEFT_MOTOR_PIN1 = 18   
        self.LEFT_MOTOR_PIN2 = 19   
        self.LEFT_MOTOR_PWM = 12    
        
        self.RIGHT_MOTOR_PIN1 = 20  
        self.RIGHT_MOTOR_PIN2 = 21  
        self.RIGHT_MOTOR_PWM = 13   
        
        self.is_moving = False
        self.current_target = None
        self.target_reached = False
        self.movement_thread = None
        self.stop_movement = False
        
        # Movement command queue for smoother control
        self.movement_queue = deque(maxlen=3)
        self.last_command = None
        self.command_repeat_count = 0
        
        if GPIO_AVAILABLE:
            self.setup_gpio()
        
    def setup_gpio(self):
        """Initialize GPIO pins for motor control"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Set up motor pins
        pins = [self.LEFT_MOTOR_PIN1, self.LEFT_MOTOR_PIN2, 
                self.RIGHT_MOTOR_PIN1, self.RIGHT_MOTOR_PIN2]
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
            
        # Set up PWM pins
        GPIO.setup(self.LEFT_MOTOR_PWM, GPIO.OUT)
        GPIO.setup(self.RIGHT_MOTOR_PWM, GPIO.OUT)
        
        self.left_pwm = GPIO.PWM(self.LEFT_MOTOR_PWM, 1000)  # 1kHz frequency
        self.right_pwm = GPIO.PWM(self.RIGHT_MOTOR_PWM, 1000)
        
        self.left_pwm.start(0)
        self.right_pwm.start(0)
        
        print("GPIO initialized for robot control")
    
    def _execute_movement(self, command, speed, duration=0.1):
        """Execute movement command in separate thread"""
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
        """Stop all motors immediately"""
        self._execute_movement('stop', 0)
        self.is_moving = False
        self.last_command = 'stop'
    
    def move_forward(self, speed=FORWARD_SPEED):
        """Move robot forward"""
        self._execute_movement('forward', speed)
        self.is_moving = True
        if self.last_command != 'forward':
            print(f"Moving forward")
        self.last_command = 'forward'
    
    def turn_left(self, speed=TURN_SPEED):
        """Turn robot left"""
        self._execute_movement('left', speed)
        self.is_moving = True
        if self.last_command != 'left':
            print(f"Turning left")
        self.last_command = 'left'
    
    def turn_right(self, speed=TURN_SPEED):
        """Turn robot right"""
        self._execute_movement('right', speed)
        self.is_moving = True
        if self.last_command != 'right':
            print(f"Turning right")
        self.last_command = 'right'
    
    def navigate_to_object(self, bbox, frame_width, frame_height, class_name):
        """
        Optimized navigation with reduced calculations
        """
        xmin, ymin, xmax, ymax = bbox
        
        # Quick calculations
        obj_center_x = (xmin + xmax) >> 1  # Bit shift for faster division
        obj_width = xmax - xmin
        frame_center_x = frame_width >> 1
        
        # Calculate object size ratio
        obj_size_ratio = obj_width / frame_width
        
        # Check if target reached
        if obj_size_ratio > PROXIMITY_THRESHOLD:
            self.stop()
            self.target_reached = True
            print(f"Target {class_name} reached!")
            return True
        
        # Skip if object too small
        if obj_size_ratio < MIN_DETECTION_SIZE:
            return False
        
        # Navigation logic - simplified
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
        """Clean up GPIO resources"""
        if GPIO_AVAILABLE:
            self.stop()
            self.left_pwm.stop()
            self.right_pwm.stop()
            GPIO.cleanup()
            print("GPIO cleaned up")

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Load the model with optimizations for Raspberry Pi
print("Loading YOLO model...")
model = YOLO(model_path, task='detect')
model.model.eval()  # Set to evaluation mode

# Optimize model for inference
if hasattr(model.model, 'float'):
    model.model.half()  # Use half precision if supported
    print("Using half precision for faster inference")

labels = model.names
print(f"Model loaded. Detected classes: {len(labels)}")

# Initialize robot controller
robot = None
if robot_mode:
    robot = OptimizedRaspberryPiRobotController()
    print("Robot navigation mode enabled")

# Parse input source
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
else:
    print(f'Input {img_source} is invalid.')
    sys.exit(0)

# Parse resolution - default to 320x240 for speed
resize = True
if user_res:
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
else:
    resW, resH = 320, 240  # Default low resolution for speed

print(f"Processing resolution: {resW}x{resH}")

# Setup recording if needed
if record:
    if source_type not in ['video','usb','picamera']:
        print('Recording only works for video and camera sources.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 15  # Reduced FPS for recording
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Initialize camera/video source with optimizations
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
            
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video': 
        cap_arg = img_source
        print(f"Opening video: {img_source}")
    elif source_type == 'usb': 
        cap_arg = usb_idx
        print(f"Opening USB camera: {usb_idx}")
    
    cap = cv2.VideoCapture(cap_arg)
    
    # Optimize camera settings for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
    
    if source_type == 'usb':
        # Additional optimizations for USB camera
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Reduce auto-exposure for speed

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    config = cap.create_video_configuration(
        main={"format": 'XRGB8888', "size": (resW, resH)},
        controls={"FrameRate": 30}
    )
    cap.configure(config)
    cap.start()
    print(f"PiCamera initialized at {resW}x{resH}")

# Reduced color palette for speed
bbox_colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255)]

# Initialize optimized variables
frame_count = 0
detection_count = 0
avg_frame_rate = 0
frame_times = deque(maxlen=30)  # Smaller buffer for faster calculation
last_detection_time = 0
detection_interval = 0.1  # Minimum time between detections (100ms)

print("Starting optimized YOLO detection...")
if robot_mode:
    print(f"Target objects: {', '.join(TARGET_OBJECTS)}")
print(f"Processing every {skip_frames} frame(s)")

try:
    while True:
        t_start = time.perf_counter()
        frame_count += 1

        # Load frame from source
        if source_type == 'image' or source_type == 'folder':
            if detection_count >= len(imgs_list):
                print('All images processed.')
                break
            frame = cv2.imread(imgs_list[detection_count])
            detection_count += 1
        
        elif source_type in ['video', 'usb']:
            ret, frame = cap.read()
            if not ret:
                print('End of video/camera stream.')
                break
        
        elif source_type == 'picamera':
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            if frame is None:
                print('PiCamera error.')
                break

        # Resize frame immediately for faster processing
        if frame.shape[:2] != (resH, resW):
            frame = cv2.resize(frame, (resW, resH))

        # Skip frames for detection (but still display all frames)
        run_detection = (frame_count % skip_frames == 0)
        
        target_found = False
        best_target = None
        object_count = 0

        if run_detection:
            current_time = time.perf_counter()
            if current_time - last_detection_time >= detection_interval:
                # Run inference with smaller input size for speed
                results = model(frame, imgsz=INFERENCE_SIZE, verbose=False, conf=min_thresh)
                detections = results[0].boxes
                last_detection_time = current_time

                if detections is not None and len(detections) > 0:
                    best_target_size = 0
                    
                    for i in range(len(detections)):
                        # Faster coordinate extraction
                        xyxy = detections[i].xyxy.cpu().numpy().flatten().astype(int)
                        xmin, ymin, xmax, ymax = xyxy
                        
                        classidx = int(detections[i].cls.item())
                        classname = labels[classidx]
                        conf = detections[i].conf.item()

                        if conf > min_thresh:
                            # Quick color selection
                            color = (0, 255, 0) if classname.lower() in [obj.lower() for obj in TARGET_OBJECTS] else bbox_colors[classidx % len(bbox_colors)]
                            
                            # Draw simplified bounding box
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 1)
                            
                            # Simplified label
                            label = f'{classname[:8]}: {int(conf*100)}%'
                            cv2.putText(frame, label, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            
                            object_count += 1

                            # Target detection for robot
                            if robot_mode and classname.lower() in [obj.lower() for obj in TARGET_OBJECTS]:
                                target_found = True
                                obj_width = xmax - xmin
                                obj_size_ratio = obj_width / resW
                                
                                if obj_size_ratio > best_target_size:
                                    best_target = {
                                        'bbox': (xmin, ymin, xmax, ymax),
                                        'class': classname,
                                        'size': obj_size_ratio
                                    }
                                    best_target_size = obj_size_ratio

        # Robot navigation (runs every frame for smooth control)
        if robot_mode and robot is not None:
            if target_found and best_target is not None:
                target_reached = robot.navigate_to_object(
                    best_target['bbox'], resW, resH, best_target['class']
                )
                
                if target_reached:
                    cv2.putText(frame, f"TARGET REACHED!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif robot.is_moving:
                # Stop if no target and currently moving
                robot.stop()

        # Calculate FPS efficiently
        t_stop = time.perf_counter()
        frame_time = t_stop - t_start
        frame_times.append(1.0 / frame_time)
        avg_frame_rate = sum(frame_times) / len(frame_times)

        # Simplified display
        cv2.putText(frame, f'FPS: {avg_frame_rate:.1f}', (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(frame, f'Objects: {object_count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        
        if robot_mode:
            status = "MOVING" if (robot and robot.is_moving) else "STOPPED"
            cv2.putText(frame, f'Robot: {status}', (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0) if robot.is_moving else (0,0,255), 1)

        cv2.imshow('YOLO Detection', frame)
        if record: recorder.write(frame)

        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if robot_mode and robot: robot.stop()
            cv2.waitKey()
        elif key == ord('r'):
            if robot_mode and robot:
                robot.stop()
                robot.target_reached = False

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    print(f'Average FPS: {avg_frame_rate:.2f}')
    
    if robot_mode and robot:
        robot.cleanup()
    
    if source_type in ['video', 'usb']:
        cap.release()
    elif source_type == 'picamera':
        cap.stop()
    
    if record: recorder.release()
    cv2.destroyAllWindows()
    print("Cleanup completed")