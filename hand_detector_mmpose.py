import cv2
import numpy as np
import math
from typing import Tuple, List, Dict, Optional
import warnings

try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import merge_data_samples
    import mmengine
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    warnings.warn("MMPose not installed. Please install: pip install mmpose")


class HandDetectorMMPose:
    """
    Hand Detection class using MMPose's neural network models
    Replaces MediaPipe for better Python 3.13 compatibility
    """
    
    def __init__(self, detection_confidence=0.7, max_hands=2, device='cpu'):
        """
        Initialize MMPose Hand solution
        
        Args:
            detection_confidence: Minimum confidence for hand detection
            max_hands: Maximum number of hands to detect
            device: Device to run inference on ('cpu' or 'cuda')
        """
        if not MMPOSE_AVAILABLE:
            raise ImportError("MMPose is required but not installed. Please install with: pip install mmpose")
        
        self.detection_confidence = detection_confidence
        self.max_hands = max_hands
        self.device = device
        
        # Initialize MMPose hand model
        # Using a lightweight hand model for real-time performance
        try:
            # You can download this config and checkpoint
            config_file = 'https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_coco_wholebody_hand_256x256-1c028db7_20210908.py'
            checkpoint_file = 'https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_coco_wholebody_hand_256x256-1c028db7_20210908.pth'
            
            # Initialize the model
            self.model = init_model(config_file, checkpoint_file, device=device)
            print("✅ MMPose hand model initialized successfully!")
            
        except Exception as e:
            # Fallback to a simpler approach if model download fails
            print(f"⚠️ Failed to load remote model: {e}")
            print("📝 Please manually download MMPose hand detection model")
            self.model = None
        
        # Hand landmark IDs (21 points for hand - compatible with MediaPipe)
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        
        # Store last detected landmarks for consistency
        self.lm_list = []
        self.results = None
        
    def find_hands(self, img, draw=True):
        """
        Detect hands in the image using MMPose
        
        Args:
            img: Input image
            draw: Whether to draw hand landmarks
            
        Returns:
            img: Image with hand landmarks drawn (if draw=True)
        """
        if self.model is None:
            # Fallback to simple hand detection simulation
            return self._fallback_detection(img, draw)
        
        try:
            # Run inference with MMPose
            # Note: MMPose expects person bounding boxes for top-down approach
            # For simplicity, we'll use the whole image as a bounding box
            h, w = img.shape[:2]
            bbox = [0, 0, w, h, 1.0]  # x1, y1, x2, y2, confidence
            
            # Run inference
            pose_results = inference_topdown(self.model, img, [bbox])
            self.results = pose_results
            
            # Draw landmarks if requested
            if draw and len(pose_results) > 0:
                img = self._draw_landmarks(img, pose_results[0])
                
        except Exception as e:
            print(f"❌ MMPose inference failed: {e}")
            # Fallback to simple detection
            return self._fallback_detection(img, draw)
        
        return img
    
    def _draw_landmarks(self, img, pose_result):
        """Draw hand landmarks on image"""
        if hasattr(pose_result, 'pred_instances') and len(pose_result.pred_instances) > 0:
            keypoints = pose_result.pred_instances.keypoints[0]  # First hand
            keypoint_scores = pose_result.pred_instances.keypoint_scores[0]
            
            # Draw keypoints
            for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
                if score > self.detection_confidence:
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
                    
            # Draw connections (simplified hand skeleton)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            ]
            
            for start_idx, end_idx in connections:
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                    keypoint_scores[start_idx] > self.detection_confidence and
                    keypoint_scores[end_idx] > self.detection_confidence):
                    
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(img, start_point, end_point, (255, 0, 0), 2)
        
        return img
    
    def _fallback_detection(self, img, draw=True):
        """Fallback hand detection when MMPose is not available"""
        # Simple color-based hand detection (very basic fallback)
        # This is just to keep the app running if MMPose fails
        h, w = img.shape[:2]
        
        # Create dummy hand landmarks in center of image for demo
        center_x, center_y = w // 2, h // 2
        self.lm_list = []
        
        # Generate 21 dummy landmarks in a hand-like pattern
        for i in range(21):
            angle = (i / 21) * 2 * math.pi
            radius = 50 + (i % 5) * 10
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            self.lm_list.append([i, x, y])
            
            if draw:
                cv2.circle(img, (x, y), 3, (0, 255, 255), -1)  # Yellow circles
        
        if draw:
            cv2.putText(img, "FALLBACK MODE - MMPose not available", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img
    
    def find_position(self, img, hand_no=0, draw=True):
        """
        Find positions of all hand landmarks
        
        Args:
            img: Input image
            hand_no: Hand number (0 for first hand, 1 for second)
            draw: Whether to draw landmark points
            
        Returns:
            lm_list: List of landmark positions [id, x, y]
            bbox: Bounding box coordinates
        """
        self.lm_list = []
        bbox = []
        
        if self.model is None:
            # Use fallback landmarks
            if len(self.lm_list) == 0:
                self._fallback_detection(img, False)
            return self.lm_list, bbox
        
        if self.results and len(self.results) > hand_no:
            pose_result = self.results[hand_no]
            
            if hasattr(pose_result, 'pred_instances') and len(pose_result.pred_instances) > 0:
                keypoints = pose_result.pred_instances.keypoints[0]
                keypoint_scores = pose_result.pred_instances.keypoint_scores[0]
                
                x_list, y_list = [], []
                
                for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
                    if score > self.detection_confidence:
                        x, y = int(kpt[0]), int(kpt[1])
                        self.lm_list.append([i, x, y])
                        x_list.append(x)
                        y_list.append(y)
                        
                        if draw:
                            cv2.circle(img, (x, y), 5, (255, 0, 255), cv2.FILLED)
                
                # Calculate bounding box
                if x_list and y_list:
                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)
                    bbox = [x_min, y_min, x_max, y_max]
                    
                    if draw:
                        cv2.rectangle(img, (x_min - 20, y_min - 20),
                                    (x_max + 20, y_max + 20), (0, 255, 0), 2)
        
        return self.lm_list, bbox
    
    def fingers_up(self):
        """
        Identify which fingers are up
        
        Returns:
            fingers: List of 1s and 0s indicating which fingers are up
        """
        fingers = []
        
        if len(self.lm_list) >= 21:  # Need all 21 landmarks
            # Thumb (different logic because thumb moves horizontally)
            if len(self.lm_list) > 4:
                if self.lm_list[4][1] > self.lm_list[3][1]:  # Tip vs previous joint
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            # Four fingers (Index, Middle, Ring, Pinky)
            tip_indices = [8, 12, 16, 20]
            pip_indices = [6, 10, 14, 18]  # PIP joints (2 joints before tip)
            
            for tip_idx, pip_idx in zip(tip_indices, pip_indices):
                if len(self.lm_list) > tip_idx and len(self.lm_list) > pip_idx:
                    if self.lm_list[tip_idx][2] < self.lm_list[pip_idx][2]:  # Y coordinate (up is less)
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    fingers.append(0)
        else:
            # Not enough landmarks detected
            fingers = [0, 0, 0, 0, 0]
        
        return fingers
    
    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        """
        Calculate distance between two landmark points
        
        Args:
            p1: First point ID
            p2: Second point ID
            img: Input image
            draw: Whether to draw the line and points
            r: Circle radius
            t: Line thickness
            
        Returns:
            length: Distance between the two points
            img: Image with line drawn
            line_info: [x1, y1, x2, y2, cx, cy] line coordinates and center
        """
        if len(self.lm_list) > max(p1, p2):
            x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
            x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
                cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            
            # Calculate Euclidean distance
            length = math.hypot(x2 - x1, y2 - y1)
            
            return length, img, [x1, y1, x2, y2, cx, cy]
        
        return 0, img, []
    
    def identify_fingers(self, img):
        """
        Identify and label all fingers
        
        Args:
            img: Input image
            
        Returns:
            finger_info: Dictionary with finger positions and status
        """
        finger_info = {}
        fingers = self.fingers_up()
        
        if len(self.lm_list) >= 21:
            for i, finger_name in enumerate(self.finger_names):
                if i < len(fingers) and len(self.lm_list) > self.tip_ids[i]:
                    tip_pos = self.lm_list[self.tip_ids[i]]
                    finger_info[finger_name] = {
                        'position': (tip_pos[1], tip_pos[2]),
                        'is_up': fingers[i],
                        'landmark_id': self.tip_ids[i]
                    }
                    
                    # Draw finger labels
                    cv2.putText(img, f"{finger_name}: {'UP' if fingers[i] else 'DOWN'}", 
                              (tip_pos[1] - 40, tip_pos[2] - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return finger_info 