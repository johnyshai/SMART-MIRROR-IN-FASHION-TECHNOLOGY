import cv2
import numpy as np
import os

class ClothingOverlay:
    def __init__(self):
        # Create directory for clothing items if not exists
        self.clothing_dir = 'clothing_images'
        os.makedirs(self.clothing_dir, exist_ok=True)
        
        # Load default clothing items
        self.garments = {
            'tshirt': {
                'male': self._load_image('tshirt_m.png'),
                'female': self._load_image('tshirt_f.png')
            },
            'shirt': {
                'male': self._load_image('shirt_m.png'),
                'female': self._load_image('shirt_f.png')
            },
            'jacket': {
                'male': self._load_image('jacket_m.png'),
                'female': self._load_image('jacket_f.png')
            },
            'pants': {
                'male': self._load_image('pants_m.png'),
                'female': self._load_image('pants_f.png')
            }
        }
        
        # If images don't exist, create placeholder colored rectangles
        for garment, genders in self.garments.items():
            for gender, img in genders.items():
                if img is None:
                    # Create a placeholder with transparency
                    if garment in ['tshirt', 'shirt', 'jacket']:
                        # Upper body garments
                        img = self._create_placeholder(200, 300, garment)
                    else:
                        # Lower body garments
                        img = self._create_placeholder(200, 400, garment)
                    
                    # Save the placeholder
                    filename = f"{garment}_{gender[0]}.png"
                    cv2.imwrite(os.path.join(self.clothing_dir, filename), img)
                    self.garments[garment][gender] = img
    
    def _load_image(self, filename):
        """Load an image with alpha channel"""
        path = os.path.join(self.clothing_dir, filename)
        if os.path.exists(path):
            return cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return None
    
    def _create_placeholder(self, width, height, garment_type):
        """Create a placeholder garment image with transparency"""
        # Create a transparent image
        img = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Choose color based on garment type
        if garment_type == 'tshirt':
            color = (0, 120, 255, 180)  # Orange with transparency
        elif garment_type == 'shirt':
            color = (0, 0, 255, 180)    # Red with transparency
        elif garment_type == 'jacket':
            color = (255, 0, 0, 180)    # Blue with transparency
        elif garment_type == 'pants':
            color = (139, 69, 19, 180)  # Brown with transparency
        else:
            color = (150, 150, 150, 180)  # Gray with transparency
        
        # Draw garment shape
        if garment_type in ['tshirt', 'shirt', 'jacket']:
            # Upper body - rough T shape
            cv2.rectangle(img, (width//4, 0), (width*3//4, height//4), color, -1)  # Shoulders
            cv2.rectangle(img, (width*3//8, height//4), (width*5//8, height), color, -1)  # Body
            
            # Sleeves
            if garment_type != 'tshirt':
                sleeve_length = width//2 if garment_type == 'jacket' else width//3
                cv2.rectangle(img, (width//4, height//10), (width//4-sleeve_length//2, height//3), color, -1)  # Left sleeve
                cv2.rectangle(img, (width*3//4, height//10), (width*3//4+sleeve_length//2, height//3), color, -1)  # Right sleeve
        else:
            # Pants - simple leg shape
            cv2.rectangle(img, (width//4, 0), (width*3//4, height//5), color, -1)  # Waist
            cv2.rectangle(img, (width//4, height//5), (width*3//8, height), color, -1)  # Left leg
            cv2.rectangle(img, (width*5//8, height//5), (width*3//4, height), color, -1)  # Right leg
        
        return img
    
    def overlay_garment(self, frame, keypoints, garment_type, gender, distance):
        """Overlay a garment on the frame based on pose keypoints"""
        if not keypoints or garment_type not in self.garments:
            return frame
            
        # Select the appropriate garment image
        garment_img = self.garments.get(garment_type, {}).get(gender)
        if garment_img is None:
            return frame
            
        # Get relevant keypoints for positioning
        left_shoulder = keypoints.get(11)
        right_shoulder = keypoints.get(12)
        left_hip = keypoints.get(23)
        right_hip = keypoints.get(24)
        left_knee = keypoints.get(25)
        right_knee = keypoints.get(26)
        left_ankle = keypoints.get(27)
        right_ankle = keypoints.get(28)
        
        # Need at least shoulders for upper body and hips for lower body
        if garment_type in ['tshirt', 'shirt', 'jacket'] and (not left_shoulder or not right_shoulder):
            return frame
        if garment_type == 'pants' and (not left_hip or not right_hip):
            return frame
        
        # Copy the frame to avoid modifying the original
        result = frame.copy()
        h, w, _ = result.shape
        
        # Calculate positioning based on garment type
        if garment_type in ['tshirt', 'shirt', 'jacket']:
            # Upper body garment
            
            # Calculate width based on shoulder width
            shoulder_width = abs(right_shoulder['x'] - left_shoulder['x'])
            garment_width = int(shoulder_width * 1.5)  # A bit wider than shoulders
            
            # Calculate height - either to hip if visible or estimated
            if left_hip and right_hip:
                # Average of left and right side lengths
                left_length = abs(left_hip['y'] - left_shoulder['y'])
                right_length = abs(right_hip['y'] - right_shoulder['y'])
                garment_height = int((left_length + right_length) / 2 * 1.1)  # Slightly longer
            else:
                # Estimate based on shoulder width
                garment_height = int(garment_width * 1.5)
            
            # Scale garment image
            garment_resized = cv2.resize(
                garment_img, 
                (garment_width, garment_height),
                interpolation=cv2.INTER_AREA
            )
            
            # Calculate position (center between shoulders, at top)
            pos_x = min(left_shoulder['x'], right_shoulder['x'])
            pos_y = min(left_shoulder['y'], right_shoulder['y']) - int(garment_height * 0.1)  # Slightly above shoulders
            
        else:  # pants
            # Lower body garment
            
            # Calculate width based on hip width
            hip_width = abs(right_hip['x'] - left_hip['x'])
            garment_width = int(hip_width * 1.2)  # A bit wider than hips
            
            # Calculate height - either to ankles if visible or estimated
            if left_ankle and right_ankle:
                # Average of left and right leg lengths
                left_length = abs(left_ankle['y'] - left_hip['y'])
                right_length = abs(right_ankle['y'] - right_hip['y'])
                garment_height = int((left_length + right_length) / 2 * 1.0)  # Full length
            elif left_knee and right_knee:
                # Use knees and estimate full length
                left_length = abs(left_knee['y'] - left_hip['y']) * 2.1
                right_length = abs(right_knee['y'] - right_hip['y']) * 2.1
                garment_height = int((left_length + right_length) / 2)
            else:
                # Estimate based on hip width
                garment_height = int(garment_width * 2)
            
            # Scale garment image
            garment_resized = cv2.resize(
                garment_img, 
                (garment_width, garment_height),
                interpolation=cv2.INTER_AREA
            )
            
            # Calculate position (center between hips, at waist)
            pos_x = min(left_hip['x'], right_hip['x'])
            pos_y = min(left_hip['y'], right_hip['y'])
        
        # Ensure coordinates are within frame
        pos_x = max(0, min(pos_x, w - garment_resized.shape[1]))
        pos_y = max(0, min(pos_y, h - garment_resized.shape[0]))
        
        # Calculate ROI for the garment
        roi_width = min(garment_resized.shape[1], w - pos_x)
        roi_height = min(garment_resized.shape[0], h - pos_y)
        
        if roi_width <= 0 or roi_height <= 0:
            return result
            
        # Extract ROI from the frame
        roi = result[pos_y:pos_y + roi_height, pos_x:pos_x + roi_width]
        
        # Get the portion of the garment that fits in the ROI
        garment_roi = garment_resized[0:roi_height, 0:roi_width]
        
        # Check shapes match (safety check)
        if roi.shape[0:2] != garment_roi.shape[0:2]:
            return result
            
        # Apply