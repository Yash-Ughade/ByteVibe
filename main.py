import cv2
import numpy as np
import time
from collections import deque
try:
    from hand_detector_mmpose import HandDetectorMMPose as HandDetector
    print("✅ Using MMPose hand detector (Python 3.13 compatible)")
except ImportError:
    try:
        from hand_detector import HandDetector
        print("⚠️ Falling back to MediaPipe hand detector")
    except ImportError:
        print("❌ No hand detector available. Please install MMPose: pip install mmpose")
        raise
from volume_controller import VolumeController


class FingerVolumeController:
    """
    Main application class for finger-based volume control
    Combines ML hand detection with real-time volume control
    """
    
    def __init__(self):
        """Initialize the application components"""
        print("🎵 Initializing Finger Volume Controller...")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)  # Width
        self.cap.set(4, 720)   # Height
        
        # Initialize hand detector with MediaPipe neural networks
        self.detector = HandDetector(detection_confidence=0.7, tracking_confidence=0.5)
        
        # Initialize volume controller
        self.volume_controller = VolumeController(min_distance=30, max_distance=300)
        
        # Distance smoothing for stable volume control
        self.distance_buffer = deque(maxlen=5)  # Keep last 5 distance measurements
        
        # UI and display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Application state
        self.is_running = True
        self.show_finger_labels = True
        self.show_volume_bar = True
        
        print("✅ Initialization complete!")
        print("📋 Controls:")
        print("   - 'q' or 'ESC': Quit")
        print("   - 'l': Toggle finger labels")
        print("   - 'v': Toggle volume bar")
        print("   - 'm': Toggle mute")
        print("   - Use thumb and index finger distance to control volume")
    
    def draw_volume_bar(self, img, volume_percent):
        """
        Draw volume bar on the image
        
        Args:
            img: Input image
            volume_percent: Current volume percentage (0-100)
        """
        h, w, _ = img.shape
        coords = self.volume_controller.get_volume_bar_coordinates(h, w, volume_percent)
        
        # Draw outer rectangle (volume bar background)
        cv2.rectangle(img, 
                     (coords['bar_x'], coords['bar_y']), 
                     (coords['bar_x'] + coords['bar_width'], coords['bar_y'] + coords['bar_height']), 
                     (200, 200, 200), 2)
        
        # Draw filled rectangle (current volume level)
        if coords['filled_height'] > 0:
            # Color gradient based on volume level
            if volume_percent < 30:
                color = (0, 255, 0)    # Green for low volume
            elif volume_percent < 70:
                color = (0, 255, 255)  # Yellow for medium volume
            else:
                color = (0, 0, 255)    # Red for high volume
            
            cv2.rectangle(img, 
                         (coords['bar_x'] + 2, coords['filled_y']), 
                         (coords['bar_x'] + coords['bar_width'] - 2, coords['bar_y'] + coords['bar_height'] - 2), 
                         color, cv2.FILLED)
        
        # Draw volume percentage text
        cv2.putText(img, f"{volume_percent}%", 
                   (coords['bar_x'] - 20, coords['bar_y'] + coords['bar_height'] + 30), 
                   self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        # Draw volume label
        cv2.putText(img, "VOLUME", 
                   (coords['bar_x'] - 30, coords['bar_y'] - 10), 
                   self.font, 0.5, (255, 255, 255), 1)
    
    def draw_distance_info(self, img, distance, volume_percent):
        """
        Draw distance and volume information
        
        Args:
            img: Input image
            distance: Current distance between thumb and index finger
            volume_percent: Current volume percentage
        """
        # Distance information
        cv2.putText(img, f"Distance: {int(distance)}px", 
                   (10, 30), self.font, self.font_scale, (0, 255, 0), self.font_thickness)
        
        # Volume information
        cv2.putText(img, f"Volume: {volume_percent}%", 
                   (10, 70), self.font, self.font_scale, (255, 0, 255), self.font_thickness)
        
        # Mute status
        if self.volume_controller.is_muted():
            cv2.putText(img, "MUTED", 
                       (10, 110), self.font, self.font_scale, (0, 0, 255), self.font_thickness)
    
    def draw_fps(self, img):
        """Draw FPS counter"""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.start_time >= 1.0:  # Update every second
            fps = self.frame_count / (current_time - self.start_time)
            self.fps = fps
            self.frame_count = 0
            self.start_time = current_time
        
        if hasattr(self, 'fps'):
            cv2.putText(img, f"FPS: {self.fps:.1f}", 
                       (img.shape[1] - 150, 30), self.font, 0.6, (255, 255, 0), 1)
    
    def draw_instructions(self, img):
        """Draw control instructions on the image"""
        instructions = [
            "Controls:",
            "Q/ESC: Quit",
            "L: Toggle labels",
            "V: Toggle volume bar",
            "M: Toggle mute"
        ]
        
        y_start = img.shape[0] - len(instructions) * 25 - 10
        for i, instruction in enumerate(instructions):
            cv2.putText(img, instruction, 
                       (10, y_start + i * 25), 
                       self.font, 0.5, (200, 200, 200), 1)
    
    def process_hand_gestures(self, img):
        """
        Process hand gestures and calculate volume
        
        Args:
            img: Input image
            
        Returns:
            distance: Distance between thumb and index finger
            volume_percent: Calculated volume percentage
        """
        # Find hands using MediaPipe neural networks
        img = self.detector.find_hands(img)
        lm_list, bbox = self.detector.find_position(img, draw=False)
        
        distance = 0
        volume_percent = self.volume_controller.get_current_volume()
        
        if len(lm_list) != 0:
            # Identify all fingers
            if self.show_finger_labels:
                finger_info = self.detector.identify_fingers(img)
            
            # Calculate distance between thumb (4) and index finger (8)
            length, img, line_info = self.detector.find_distance(4, 8, img)
            distance = length
            
            # Add distance to buffer for smoothing
            self.distance_buffer.append(distance)
            
            # Apply smoothing
            smoothed_distance = self.volume_controller.calculate_volume_smoothing(
                list(self.distance_buffer)
            )
            
            # Set volume based on smoothed distance
            success, volume_percent = self.volume_controller.set_volume(smoothed_distance)
            
            # Visual feedback based on distance
            if distance < 50:
                # Very close - highlight in red
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 0, 255), cv2.FILLED)
            elif distance > 250:
                # Very far - highlight in green
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
        
        return distance, volume_percent
    
    def handle_keyboard_input(self):
        """Handle keyboard input for controls"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            self.is_running = False
        elif key == ord('l'):  # Toggle finger labels
            self.show_finger_labels = not self.show_finger_labels
            print(f"Finger labels: {'ON' if self.show_finger_labels else 'OFF'}")
        elif key == ord('v'):  # Toggle volume bar
            self.show_volume_bar = not self.show_volume_bar
            print(f"Volume bar: {'ON' if self.show_volume_bar else 'OFF'}")
        elif key == ord('m'):  # Toggle mute
            if self.volume_controller.is_muted():
                self.volume_controller.unmute_volume()
                print("Volume unmuted")
            else:
                self.volume_controller.mute_volume()
                print("Volume muted")
    
    def run(self):
        """Main application loop"""
        print("\n🚀 Starting Finger Volume Controller...")
        print("Show your hand to the camera and use thumb-index distance to control volume!")
        
        while self.is_running:
            # Read frame from camera
            success, img = self.cap.read()
            if not success:
                print("❌ Failed to read from camera")
                break
            
            # Flip image horizontally for mirror effect
            img = cv2.flip(img, 1)
            
            # Process hand gestures and calculate volume
            distance, volume_percent = self.process_hand_gestures(img)
            
            # Draw UI elements
            self.draw_distance_info(img, distance, volume_percent)
            
            if self.show_volume_bar:
                self.draw_volume_bar(img, volume_percent)
            
            self.draw_fps(img)
            self.draw_instructions(img)
            
            # Display the image
            cv2.imshow("Finger Volume Controller", img)
            
            # Handle keyboard input
            self.handle_keyboard_input()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Thanks for using Finger Volume Controller!")


def main():
    """Main entry point"""
    try:
        app = FingerVolumeController()
        app.run()
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your camera is connected and dependencies are installed")


if __name__ == "__main__":
    main() 