import RPi.GPIO as GPIO
import cv2
import mediapipe as mp
import numpy as np
import time
import datetime
import Adafruit_DHT  # For temperature sensor
import json
import os
from clothing_overlay import ClothingOverlay
from user_interface import UserInterface
from sensor_manager import SensorManager
from pose_estimation import PoseEstimator
from measurements import MeasurementCalculator

# Create directory for storing data if not exists
os.makedirs('data', exist_ok=True)

# === Hardware Configuration ===
GPIO.setwarnings(False)

# Initialize components
sensor_mgr = SensorManager()
ui = UserInterface()
clothing = ClothingOverlay()
pose_estimator = PoseEstimator()
measurements = MeasurementCalculator()

# === OpenCV Camera Feed ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
prev_gray = None
motion_threshold = 5000

# === Application State ===
current_garment = "tshirt"  # Default garment
user_gender = "male"        # Default gender
show_measurements = True
show_debug = False
last_temp_check = 0
current_temp = 0
current_humidity = 0

print("Smart Mirror Fashion Technology System Running... Press 'q' to exit")

try:
    while True:
        # === Read Camera Frame ===
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        frame = cv2.flip(frame, 1)  # Mirror image for better user experience
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # === Update Environmental Data ===
        current_time = time.time()
        if current_time - last_temp_check > 10:  # Update every 10 seconds
            temp, humidity = sensor_mgr.get_temperature()
            if temp is not None:
                current_temp = temp
                current_humidity = humidity
            last_temp_check = current_time
        
        # === Check Sensors ===
        sensor_data = sensor_mgr.read_all_sensors()
        if sensor_data['pir_detected']:
            print("[PIR] Motion Detected!")
        if sensor_data['ir_detected']:
            print("[IR] Gesture Detected!")
            # Example of using IR for garment switching
            if time.time() - sensor_mgr.last_gesture_time > 2:  # Debounce
                current_garment = "shirt" if current_garment == "tshirt" else "tshirt"
                sensor_mgr.last_gesture_time = time.time()
                
        # Distance affects overlay sizing
        distance = sensor_data['distance']
        if distance < 40:
            print(f"[Ultrasonic] User close: {distance} cm")
            
        # === Body and Pose Detection ===
        pose_results = pose_estimator.process_frame(rgb)
        
        # If pose detected, get measurements
        body_keypoints = pose_estimator.get_keypoints(pose_results, frame.shape[1], frame.shape[0])
        
        if body_keypoints:
            user_measurements = measurements.calculate_from_keypoints(body_keypoints, distance)
            
            # Apply clothing overlay if body detected
            frame = clothing.overlay_garment(
                frame, 
                body_keypoints, 
                current_garment,
                user_gender,
                distance
            )
            
        # === Motion Detection ===
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            motion_level = cv2.countNonZero(thresh)
            motion_detected = motion_level > motion_threshold
        else:
            motion_detected = False
        prev_gray = gray
        
        # === Draw User Interface ===
        ui_frame = ui.draw_interface(
            frame,
            datetime.datetime.now(),
            current_temp,
            current_humidity,
            current_garment,
            user_gender,
            user_measurements if body_keypoints else None,
            motion_detected,
            distance,
            sensor_data,
            show_measurements,
            show_debug
        )
        
        # === Display Output ===
        cv2.imshow("Smart Mirror - Fashion Try-On", ui_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # === Handle Key Commands ===
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_measurements = not show_measurements
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('g'):
            user_gender = "female" if user_gender == "male" else "male"
        elif key == ord('t'):
            current_garment = "tshirt"
        elif key == ord('s'):
            current_garment = "shirt"
        elif key == ord('j'):
            current_garment = "jacket"
        elif key == ord('p'):
            current_garment = "pants"
        elif key == ord('c'):
            # Save current measurements to file
            if body_keypoints:
                with open('data/last_measurements.json', 'w') as f:
                    json.dump(user_measurements, f)
                print("Measurements saved!")
            
except KeyboardInterrupt:
    print("\n[INFO] Stopped by user.")
finally:
    # === Cleanup ===
    sensor_mgr.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("System shutdown complete.")
