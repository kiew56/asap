import os
import sys
import argparse
import glob
import time
import threading

import cv2
import numpy as np
from ultralytics import YOLO
import RPi.GPIO as GPIO

# GPIO setup for motor control
GPIO.setwarnings(False)
# Right Motor
in1 = 17
in2 = 27
en_a = 4
# Left Motor
in3 = 5
in4 = 6
en_b = 13

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(en_a, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(en_b, GPIO.OUT)

# PWM setup
q = GPIO.PWM(en_a, 100)
p = GPIO.PWM(en_b, 100)
p.start(75)
q.start(75)

# Initialize motors to stop
GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
GPIO.output(in4, GPIO.LOW)
GPIO.output(in3, GPIO.LOW)

# Motor control functions
def move_forward():
    """Move robot forward"""
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)
    GPIO.output(in3, GPIO.LOW)
    print("Moving Forward")

def stop_motors():
    """Stop all motors"""
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    print("Motors Stopped")

def move_forward_timed():
    """Move forward for 4 seconds then stop"""
    move_forward()
    time.sleep(4)
    stop_motors()

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    GPIO.cleanup()
    sys.exit(0)

# Load the model into memory and get labelmap
try:
    model = YOLO(model_path, task='detect')
    labels = model.names
    print(f"Model loaded successfully. Available classes: {labels}")
except Exception as e:
    print(f"Error loading model: {e}")
    GPIO.cleanup()
    sys.exit(0)

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

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
        GPIO.cleanup()
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    GPIO.cleanup()
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only works for video and camera sources. Please try again.')
        GPIO.cleanup()
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        GPIO.cleanup()
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

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
    if source_type == 'video': 
        cap_arg = img_source
    elif source_type == 'usb': 
        cap_arg = usb_idx
    
    cap = cv2.VideoCapture(cap_arg)
    if not cap.isOpened():
        print(f"Error: Could not open camera/video source {cap_arg}")
        GPIO.cleanup()
        sys.exit(0)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    try:
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
        cap.start()
    except ImportError:
        print("Picamera2 library not found. Please install it or use a different source.")
        GPIO.cleanup()
        sys.exit(0)

# Set bounding box colors (using the Tableau 10 color scheme)
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106), 
              (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
motor_moving = False
motor_thread = None

# Function to check if both potted plant and bottle are detected
def check_target_objects(detections, labels, confidence_threshold):
    """
    Check if both potted plant and bottle are detected with sufficient confidence
    Returns: (has_plant, has_bottle, both_detected)
    """
    detected_classes = []
    
    for i in range(len(detections)):
        classidx = int(detections[i].cls.item())
        classname = labels[classidx].lower()
        conf = detections[i].conf.item()
        
        if conf > confidence_threshold:
            detected_classes.append(classname)
    
    has_plant = any('potted plant' in cls or 'plant' in cls for cls in detected_classes)
    has_bottle = any('bottle' in cls for cls in detected_classes)
    both_detected = has_plant and has_bottle
    
    return has_plant, has_bottle, both_detected

print("Starting YOLO detection with motor control...")
print("Looking for 'potted plant' and 'bottle' objects...")
print("Press 'q' to quit, 's' to pause, 'p' to save screenshot")

# Wrap main content in a try block for cleanup
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
            if frame is None:
                print(f"Could not load image: {img_filename}")
                img_count += 1
                continue
            img_count += 1
        
        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret:
                print('Reached end of the video file. Exiting program.')
                break
        
        elif source_type == 'usb':
            ret, frame = cap.read()
            if (frame is None) or (not ret):
                print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
                break

        elif source_type == 'picamera':
            try:
                frame_bgra = cap.capture_array()
                frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
                if frame is None:
                    print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
                    break
            except Exception as e:
                print(f"Error capturing from Picamera: {e}")
                break

        # Resize frame to desired display resolution
        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # Run inference on frame
        try:
            results = model(frame, verbose=False)
            detections = results[0].boxes
        except Exception as e:
            print(f"Error during inference: {e}")
            continue

        # Initialize variables for object counting and detection
        object_count = 0
        detected_objects = []

        # Check for target objects (potted plant and bottle)
        if detections is not None and len(detections) > 0:
            has_plant, has_bottle, both_detected = check_target_objects(detections, labels, min_thresh)
            
            # If both objects are detected and motors aren't already moving, start movement
            if both_detected and not motor_moving:
                print("BOTH POTTED PLANT AND BOTTLE DETECTED! Moving forward for 4 seconds...")
                motor_moving = True
                motor_thread = threading.Thread(target=move_forward_timed)
                motor_thread.start()
                
                # Set a timer to reset the motor_moving flag
                def reset_motor_flag():
                    time.sleep(4.5)  # Wait a bit longer than the movement time
                    global motor_moving
                    motor_moving = False
                
                threading.Thread(target=reset_motor_flag).start()

            # Process each detection for display
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
                    color = bbox_colors[classidx % 10]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                    label = f'{classname}: {int(conf*100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    object_count += 1
                    detected_objects.append(classname)

        # Calculate and draw framerate (if using video, USB, or Picamera source)
        if source_type in ['video', 'usb', 'picamera']:
            cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display detection results and motor status
        cv2.putText(frame, f'Objects detected: {object_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show target detection status
        status_color = (0, 255, 0) if motor_moving else (0, 255, 255)
        status_text = "MOVING!" if motor_moving else "Searching for plant & bottle..."
        cv2.putText(frame, status_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow('YOLO Detection with Motor Control', frame)
        if record: 
            recorder.write(frame)

        # Handle key presses
        if source_type in ['image', 'folder']:
            key = cv2.waitKey()
        else:
            key = cv2.waitKey(5)
        
        if key == ord('q') or key == ord('Q'):  # Press 'q' to quit
            break
        elif key == ord('s') or key == ord('S'):  # Press 's' to pause inference
            cv2.waitKey()
        elif key == ord('p') or key == ord('P'):  # Press 'p' to save a picture
            cv2.imwrite('capture.png', frame)
            print("Screenshot saved as capture.png")
        
        # Calculate FPS for this frame
        t_stop = time.perf_counter()
        frame_rate_calc = float(1/(t_stop - t_start))

        # Append FPS result to frame_rate_buffer
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
        else:
            frame_rate_buffer.append(frame_rate_calc)

        # Calculate average FPS for past frames
        if frame_rate_buffer:
            avg_frame_rate = np.mean(frame_rate_buffer)

except KeyboardInterrupt:
    print("\nProgram interrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Clean up
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    
    # Stop motors
    stop_motors()
    
    # Wait for motor thread to finish
    if motor_thread and motor_thread.is_alive():
        motor_thread.join(timeout=5)
    
    # Clean up camera resources
    if source_type in ['video', 'usb'] and 'cap' in locals():
        cap.release()
    elif source_type == 'picamera' and 'cap' in locals():
        cap.stop()
    
    # Clean up recording
    if record and 'recorder' in locals(): 
        recorder.release()
    
    # Clean up OpenCV and GPIO
    cv2.destroyAllWindows()
    p.stop()
    q.stop()
    GPIO.cleanup()
    print("GPIO and resources cleaned up")
