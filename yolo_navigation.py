import os
import sys
import argparse
import glob
import time
import math

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
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--robot', help='Enable robot navigation mode',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record
robot_mode = args.robot

# Navigation parameters
TARGET_OBJECTS = ['bottle', 'paper', 'plant']  # Objects to navigate toward
PROXIMITY_THRESHOLD = 0.4  # Stop when object occupies 40% of frame width
CENTER_TOLERANCE = 80  # Pixel tolerance for centering
TURN_SPEED = 0.6  # Turning speed
FORWARD_SPEED = 0.7  # Forward movement speed
MIN_DETECTION_SIZE = 0.05  # Minimum object size to consider (5% of frame)

class RaspberryPiRobotController:
    """
    Raspberry Pi Robot Controller using GPIO for motor control
    Assumes standard 2-motor setup with L298N or similar motor driver
    """
    def __init__(self):
        # Motor GPIO pins - adjust these based on your wiring
        self.LEFT_MOTOR_PIN1 = 18   # Left motor direction pin 1
        self.LEFT_MOTOR_PIN2 = 19   # Left motor direction pin 2
        self.LEFT_MOTOR_PWM = 12    # Left motor PWM (speed control)
        
        self.RIGHT_MOTOR_PIN1 = 20  # Right motor direction pin 1
        self.RIGHT_MOTOR_PIN2 = 21  # Right motor direction pin 2
        self.RIGHT_MOTOR_PWM = 13   # Right motor PWM (speed control)
        
        self.is_moving = False
        self.current_target = None
        self.target_reached = False
        
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
    
    def stop(self):
        """Stop all motors"""
        if GPIO_AVAILABLE:
            GPIO.output(self.LEFT_MOTOR_PIN1, GPIO.LOW)
            GPIO.output(self.LEFT_MOTOR_PIN2, GPIO.LOW)
            GPIO.output(self.RIGHT_MOTOR_PIN1, GPIO.LOW)
            GPIO.output(self.RIGHT_MOTOR_PIN2, GPIO.LOW)
            self.left_pwm.ChangeDutyCycle(0)
            self.right_pwm.ChangeDutyCycle(0)
        
        self.is_moving = False
        print("Robot stopped")
    
    def move_forward(self, speed=FORWARD_SPEED):
        """Move robot forward"""
        if GPIO_AVAILABLE:
            # Set direction for both motors (forward)
            GPIO.output(self.LEFT_MOTOR_PIN1, GPIO.HIGH)
            GPIO.output(self.LEFT_MOTOR_PIN2, GPIO.LOW)
            GPIO.output(self.RIGHT_MOTOR_PIN1, GPIO.HIGH)
            GPIO.output(self.RIGHT_MOTOR_PIN2, GPIO.LOW)
            
            # Set speed
            duty_cycle = int(speed * 100)
            self.left_pwm.ChangeDutyCycle(duty_cycle)
            self.right_pwm.ChangeDutyCycle(duty_cycle)
        
        self.is_moving = True
        print(f"Moving forward at {speed*100:.0f}% speed")
    
    def turn_left(self, speed=TURN_SPEED):
        """Turn robot left (right wheel forward, left wheel backward)"""
        if GPIO_AVAILABLE:
            # Left motor backward
            GPIO.output(self.LEFT_MOTOR_PIN1, GPIO.LOW)
            GPIO.output(self.LEFT_MOTOR_PIN2, GPIO.HIGH)
            # Right motor forward
            GPIO.output(self.RIGHT_MOTOR_PIN1, GPIO.HIGH)
            GPIO.output(self.RIGHT_MOTOR_PIN2, GPIO.LOW)
            
            duty_cycle = int(speed * 100)
            self.left_pwm.ChangeDutyCycle(duty_cycle)
            self.right_pwm.ChangeDutyCycle(duty_cycle)
        
        self.is_moving = True
        print(f"Turning left at {speed*100:.0f}% speed")
    
    def turn_right(self, speed=TURN_SPEED):
        """Turn robot right (left wheel forward, right wheel backward)"""
        if GPIO_AVAILABLE:
            # Left motor forward
            GPIO.output(self.LEFT_MOTOR_PIN1, GPIO.HIGH)
            GPIO.output(self.LEFT_MOTOR_PIN2, GPIO.LOW)
            # Right motor backward
            GPIO.output(self.RIGHT_MOTOR_PIN1, GPIO.LOW)
            GPIO.output(self.RIGHT_MOTOR_PIN2, GPIO.HIGH)
            
            duty_cycle = int(speed * 100)
            self.left_pwm.ChangeDutyCycle(duty_cycle)
            self.right_pwm.ChangeDutyCycle(duty_cycle)
        
        self.is_moving = True
        print(f"Turning right at {speed*100:.0f}% speed")
    
    def navigate_to_object(self, bbox, frame_width, frame_height, class_name):
        """
        Navigate toward detected object based on its bounding box
        Returns True if target is reached, False otherwise
        """
        xmin, ymin, xmax, ymax = bbox
        
        # Calculate object center and size
        obj_center_x = (xmin + xmax) / 2
        obj_center_y = (ymin + ymax) / 2
        obj_width = xmax - xmin
        obj_height = ymax - ymin
        
        # Calculate frame center
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        # Calculate object size relative to frame
        obj_size_ratio = obj_width / frame_width
        
        # Check if object is large enough (close enough)
        if obj_size_ratio > PROXIMITY_THRESHOLD:
            self.stop()
            self.target_reached = True
            print(f"Target {class_name} reached! Object size: {obj_size_ratio:.2f}")
            return True
        
        # Check if object is too small to reliably track
        if obj_size_ratio < MIN_DETECTION_SIZE:
            print(f"Object {class_name} too small/far to track reliably")
            return False
        
        # Calculate horizontal offset from center
        horizontal_offset = obj_center_x - frame_center_x
        
        # Navigation logic
        if abs(horizontal_offset) > CENTER_TOLERANCE:
            # Object not centered, turn toward it
            if horizontal_offset > 0:
                self.turn_right(TURN_SPEED)
                print(f"Turning right toward {class_name} (offset: {horizontal_offset:.0f})")
            else:
                self.turn_left(TURN_SPEED)
                print(f"Turning left toward {class_name} (offset: {horizontal_offset:.0f})")
        else:
            # Object is centered, move forward
            self.move_forward(FORWARD_SPEED)
            print(f"Moving toward {class_name} (size: {obj_size_ratio:.2f})")
        
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
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get label map
model = YOLO(model_path, task='detect')
labels = model.names

# Initialize robot controller if in robot mode
robot = None
if robot_mode:
    robot = RaspberryPiRobotController()
    print("Robot navigation mode enabled")

# Parse input to determine if image source is a file, folder, video, or USB camera
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
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
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
    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    if user_res:
        cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    else:
        cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888'}))
    cap.start()

# Set bounding box colors (using the Tableau 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
last_movement_time = 0
movement_timeout = 2.0  # Stop movement after 2 seconds of no target

print("Starting YOLO detection with navigation...")
if robot_mode:
    print(f"Looking for target objects: {', '.join(TARGET_OBJECTS)}")

try:
    # Begin inference loop
    while True:
        t_start = time.perf_counter()

        # Load frame from image source
        if source_type == 'image' or source_type == 'folder':
            if img_count >= len(imgs_list):
                print('All images have been processed. Exiting program.')
                break
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count = img_count + 1
        
        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret:
                print('Reached end of the video file. Exiting program.')
                break
        
        elif source_type == 'usb':
            ret, frame = cap.read()
            if (frame is None) or (not ret):
                print('Unable to read frames from the camera. Exiting program.')
                break

        elif source_type == 'picamera':
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
            if (frame is None):
                print('Unable to read frames from the Picamera. Exiting program.')
                break

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Resize frame to desired display resolution
        if resize == True:
            frame = cv2.resize(frame,(resW,resH))
            frame_height, frame_width = resH, resW

        # Run inference on frame
        results = model(frame, verbose=False)
        detections = results[0].boxes

        # Initialize variables for navigation and counting
        object_count = 0
        target_found = False
        best_target = None
        best_target_size = 0

        # Go through each detection and get bbox coords, confidence, and class
        if detections is not None:
            for i in range(len(detections)):
                # Get bounding box coordinates
                xyxy_tensor = detections[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze()
                xmin, ymin, xmax, ymax = xyxy.astype(int)

                # Get bounding box class ID and name
                classidx = int(detections[i].cls.item())
                classname = labels[classidx]

                # Get bounding box confidence
                conf = detections[i].conf.item()

                # Draw box if confidence threshold is high enough
                if conf > min_thresh:
                    # Choose color based on whether it's a target object
                    if classname.lower() in [obj.lower() for obj in TARGET_OBJECTS]:
                        color = (0, 255, 0)  # Green for target objects
                    else:
                        color = bbox_colors[classidx % 10]
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

                    label = f'{classname}: {int(conf*100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    object_count = object_count + 1

                    # Check if this is a target object for navigation
                    if robot_mode and classname.lower() in [obj.lower() for obj in TARGET_OBJECTS]:
                        target_found = True
                        obj_width = xmax - xmin
                        obj_size_ratio = obj_width / frame_width
                        
                        # Select the largest (closest) target object
                        if obj_size_ratio > best_target_size:
                            best_target = {
                                'bbox': (xmin, ymin, xmax, ymax),
                                'class': classname,
                                'conf': conf,
                                'size': obj_size_ratio
                            }
                            best_target_size = obj_size_ratio

        # Robot navigation logic
        if robot_mode and robot is not None:
            if target_found and best_target is not None:
                # Navigate toward the best (largest/closest) target
                target_reached = robot.navigate_to_object(
                    best_target['bbox'], 
                    frame_width, 
                    frame_height, 
                    best_target['class']
                )
                
                if target_reached:
                    # Add visual indication that target is reached
                    cv2.putText(frame, f"TARGET {best_target['class'].upper()} REACHED!", 
                              (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                last_movement_time = time.perf_counter()
            else:
                # No target found, stop if we were moving
                current_time = time.perf_counter()
                if robot.is_moving and (current_time - last_movement_time) > movement_timeout:
                    robot.stop()
                    print("No target objects detected, stopping robot")

        # Calculate and draw framerate (if using video, USB, or Picamera source)
        if source_type in ['video', 'usb', 'picamera']:
            cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        
        # Display detection results
        cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        
        # Display robot status
        if robot_mode:
            status_text = "MOVING" if (robot and robot.is_moving) else "STOPPED"
            status_color = (0, 255, 255) if (robot and robot.is_moving) else (0, 0, 255)
            cv2.putText(frame, f'Robot: {status_text}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, .7, status_color, 2)
            
            if target_found and best_target:
                cv2.putText(frame, f'Target: {best_target["class"]} ({best_target["size"]:.2f})', 
                          (10,100), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)

        cv2.imshow('YOLO Detection with Navigation', frame)
        if record: recorder.write(frame)

        # Handle keyboard input
        if source_type in ['image', 'folder']:
            key = cv2.waitKey()
        else:
            key = cv2.waitKey(5)
        
        if key == ord('q') or key == ord('Q'):  # Press 'q' to quit
            break
        elif key == ord('s') or key == ord('S'):  # Press 's' to pause/stop
            if robot_mode and robot:
                robot.stop()
            cv2.waitKey()
        elif key == ord('p') or key == ord('P'):  # Press 'p' to save picture
            cv2.imwrite('capture.png', frame)
            print("Frame saved as capture.png")
        elif key == ord('r') or key == ord('R'):  # Press 'r' to reset robot
            if robot_mode and robot:
                robot.stop()
                robot.target_reached = False
                print("Robot reset")
        
        # Calculate FPS for this frame
        t_stop = time.perf_counter()
        frame_rate_calc = float(1/(t_stop - t_start))

        # Append FPS result to frame_rate_buffer
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
        else:
            frame_rate_buffer.append(frame_rate_calc)

        # Calculate average FPS
        avg_frame_rate = np.mean(frame_rate_buffer)

except KeyboardInterrupt:
    print("\nProgram interrupted by user")

finally:
    # Clean up
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    
    if robot_mode and robot:
        robot.cleanup()
    
    if source_type in ['video', 'usb']:
        cap.release()
    elif source_type == 'picamera':
        cap.stop()
    
    if record: 
        recorder.release()
    
    cv2.destroyAllWindows()
    print("Program terminated cleanly")