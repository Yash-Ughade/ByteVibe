import cv2
import mediapipe as mp
import numpy as np
import math


class HandDetector:
    """
    Hand Detection class using MediaPipe's neural network models
    Detects hands and identifies all finger landmarks
    """
    
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize MediaPipe Hand solution
        
        Args:
            mode: Static image mode or video mode
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum confidence for hand detection
            tracking_confidence: Minimum confidence for hand tracking
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe hands solution (uses neural networks)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        
        # Drawing utilities
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        # Finger tip landmark IDs (MediaPipe hand model)
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        # Finger names for identification
        self.finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        
    def find_hands(self, img, draw=True):
        """
        Detect hands in the image using MediaPipe's neural network
        
        Args:
            img: Input image
            draw: Whether to draw hand landmarks
            
        Returns:
            img: Image with hand landmarks drawn (if draw=True)
        """
        # Convert BGR to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image through the neural network
        self.results = self.hands.process(img_rgb)
        
        # Draw hand landmarks if hands are detected
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks and connections
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        self.mp_draw_styles.get_default_hand_connections_style()
                    )
        
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
        """
        x_list = []
        y_list = []
        bbox = []
        self.lm_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_list.append(cx)
                    y_list.append(cy)
                    self.lm_list.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
                # Calculate bounding box
                if x_list and y_list:
                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)
                    bbox = x_min, y_min, x_max, y_max
                    
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
        
        if len(self.lm_list) != 0:
            # Thumb (different logic because thumb moves horizontally)
            if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # Four fingers (Index, Middle, Ring, Pinky)
            for id in range(1, 5):
                if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
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
        if len(self.lm_list) != 0:
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
        
        if len(self.lm_list) != 0:
            for i, finger_name in enumerate(self.finger_names):
                if i < len(fingers):
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