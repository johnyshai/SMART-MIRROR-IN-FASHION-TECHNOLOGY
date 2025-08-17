import numpy as np
import math

class MeasurementCalculator:
    def __init__(self):
        # Define keypoint indices from MediaPipe Pose
        # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        self.keypoints = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'neck': 0,  # Approximation
            'mid_hip': None  # Will calculate midpoint
        }
        
        # Calibration factor (to convert pixel measurements to cm)
        # This could be calibrated based on known distance
        self.pixel_to_cm_ratio = 0.2  # Initial estimate
        
    def distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)
    
    def calculate_from_keypoints(self, keypoints, actual_distance):
        """Calculate body measurements from keypoints"""
        if not keypoints:
            return None
            
        # Dynamically adjust pixel to cm ratio based on distance sensor
        if 30 < actual_distance < 300:
            # This is a rough calibration - would need testing for accurate conversion
            self.pixel_to_cm_ratio = 0.15 + (actual_distance / 1000)
            
        # Calculate midpoints
        left_hip = keypoints.get(self.keypoints['left_hip'])
        right_hip = keypoints.get(self.keypoints['right_hip'])
        
        if left_hip and right_hip:
            mid_hip = {
                'x': (left_hip['x'] + right_hip['x']) // 2,
                'y': (left_hip['y'] + right_hip['y']) // 2,
                'z': (left_hip['z'] + right_hip['z']) / 2,
                'v': (left_hip['v'] + right_hip['v']) / 2
            }
        else:
            mid_hip = None
            
        # Calculate measurements
        measurements = {}
        
        # Shoulder width
        left_shoulder = keypoints.get(self.keypoints['left_shoulder'])
        right_shoulder = keypoints.get(self.keypoints['right_shoulder'])
        if left_shoulder and right_shoulder:
            shoulder_width_px = self.distance(left_shoulder, right_shoulder)
            measurements['shoulder_width'] = round(shoulder_width_px * self.pixel_to_cm_ratio, 1)
        
        # Arm length (average of both arms)
        arm_lengths = []
        
        # Left arm
        left_shoulder = keypoints.get(self.keypoints['left_shoulder'])
        left_elbow = keypoints.get(self.keypoints['left_elbow'])
        left_wrist = keypoints.get(self.keypoints['left_wrist'])
        
        if left_shoulder and left_elbow and left_wrist:
            upper_arm_px = self.distance(left_shoulder, left_elbow)
            forearm_px = self.distance(left_elbow, left_wrist)
            arm_lengths.append((upper_arm_px + forearm_px) * self.pixel_to_cm_ratio)
        
        # Right arm
        right_shoulder = keypoints.get(self.keypoints['right_shoulder'])
        right_elbow = keypoints.get(self.keypoints['right_elbow'])
        right_wrist = keypoints.get(self.keypoints['right_wrist'])
        
        if right_shoulder and right_elbow and right_wrist:
            upper_arm_px = self.distance(right_shoulder, right_elbow)
            forearm_px = self.distance(right_elbow, right_wrist)
            arm_lengths.append((upper_arm_px + forearm_px) * self.pixel_to_cm_ratio)
        
        if arm_lengths:
            measurements['arm_length'] = round(sum(arm_lengths) / len(arm_lengths), 1)
        
        # Torso length
        neck = keypoints.get(self.keypoints['neck'])
        if neck and mid_hip:
            torso_px = self.distance(neck, mid_hip)
            measurements['torso_length'] = round(torso_px * self.pixel_to_cm_ratio, 1)
        
        # Leg length (average of both legs)
        leg_lengths = []
        
        # Left leg
        left_hip = keypoints.get(self.keypoints['left_hip'])
        left_knee = keypoints.get(self.keypoints['left_knee'])
        left_ankle = keypoints.get(self.keypoints['left_ankle'])
        
        if left_hip and left_knee and left_ankle:
            thigh_px = self.distance(left_hip, left_knee)
            calf_px = self.distance(left_knee, left_ankle)
            leg_lengths.append((thigh_px + calf_px) * self.pixel_to_cm_ratio)
        
        # Right leg
        right_hip = keypoints.get(self.keypoints['right_hip'])
        right_knee = keypoints.get(self.keypoints['right_knee'])
        right_ankle = keypoints.get(self.keypoints['right_ankle'])
        
        if right_hip and right_knee and right_ankle:
            thigh_px = self.distance(right_hip, right_knee)
            calf_px = self.distance(right_knee, right_ankle)
            leg_lengths.append((thigh_px + calf_px) * self.pixel_to_cm_ratio)
        
        if leg_lengths:
            measurements['leg_length'] = round(sum(leg_lengths) / len(leg_lengths), 1)
        
        # Estimate height
        if 'torso_length' in measurements and 'leg_length' in measurements:
            # Rough estimation of height (torso + legs + estimated head)
            estimated_head = measurements['torso_length'] * 0.2  # Rough proportion
            measurements['estimated_height'] = round(
                measurements['torso_length'] + measurements['leg_length'] + estimated_head, 1
            )
        
        # Estimate chest, waist and hip circumference
        # These are very rough approximations and would need calibration for accuracy
        if left_shoulder and right_shoulder:
            chest_width_px = self.distance(left_shoulder, right_shoulder) * 1.4
            measurements['chest'] = round(chest_width_px * self.pixel_to_cm_ratio * 3.14159 / 2, 1)
        
        if left_hip and right_hip:
            hip_width_px = self.distance(left_hip, right_hip) * 1.2
            measurements['hips'] = round(hip_width_px * self.pixel_to_cm_ratio * 3.14159 / 2, 1)
            
            # Estimate waist (usually narrower than hips)
            waist_width_px = hip_width_px * 0.9
            measurements['waist'] = round(waist_width_px * self.pixel_to_cm_ratio * 3.14159 / 2, 1)
        
        # Determine clothing size based on measurements
        measurements['size'] = self.determine_size(measurements)
        
        return measurements
    
    def determine_size(self, measurements):
        """Determine clothing size based on measurements"""
        # Very simplified size estimation - would need proper sizing charts
        
        # Default to Medium if we don't have enough data
        if not measurements or 'chest' not in measurements:
            return "M"
            
        # Simple chest-based sizing for demonstration
        chest = measurements.get('chest', 0)
        
        if chest < 85:
            return "XS"
        elif chest < 95:
            return "S"
        elif chest < 105:
            return "M"
        elif chest < 115:
            return "L"
        elif chest < 125:
            return "XL"
        else:
            return "XXL"
