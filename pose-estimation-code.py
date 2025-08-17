import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self):
        # Initialize mediapipe pose and hands solutions
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.mp_hands = mp.solutions.hands
        
        # Initialize with high accuracy settings
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1  # 0=Lite, 1=Full, 2=Heavy
        )
        
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_hands=2
        )
    
    def process_frame(self, rgb_image):
        """Process a frame through MediaPipe Pose"""
        return self.pose.process(rgb_image)
    
    def process_hands(self, rgb_image):
        """Process a frame through MediaPipe Hands"""
        return self.hands.process(rgb_image)
    
    def draw_pose_landmarks(self, image, results):
        """Draw pose landmarks on the image"""
        if not results.pose_landmarks:
            return image
            
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return image
    
    def draw_hand_landmarks(self, image, hand_results):
        """Draw hand landmarks on the image"""
        if not hand_results.multi_hand_landmarks:
            return image
            
        for hand_landmarks in hand_results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        return image
    
    def get_keypoints(self, results, width, height):
        """Extract keypoints from pose results and normalize to image dimensions"""
        if not results.pose_landmarks:
            return None
            
        keypoints = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            # Convert normalized coordinates to pixel values
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z  # Depth (relative)
            v = landmark.visibility  # Visibility confidence [0-1]
            
            # Store keypoint
            keypoints[idx] = {
                'x': x,
                'y': y,
                'z': z,
                'v': v
            }
        
        return keypoints
