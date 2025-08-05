import cv2
import numpy as np
import math
from volume_controller import VolumeController
import time


class SimpleVolumeController:
    """
    Simple volume controller using OpenCV face detection and mouse tracking
    as a workaround for MediaPipe compatibility issues
    """
    
    def __init__(self):
        print("🎵 Simple Volume Controller (OpenCV-only)")
        print("=" * 50)
        print("⚠️  MediaPipe compatibility workaround")
        print("📖 Instructions:")
        print("   1. Move your mouse left-right to control volume")
        print("   2. Mouse on LEFT side = Low volume")
        print("   3. Mouse on RIGHT side = High volume")
        print("   4. Press 'q' to quit")
        print("=" * 50)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)  # Width
        self.cap.set(4, 720)   # Height
        
        # Initialize volume controller
        self.volume_controller = VolumeController(min_distance=50, max_distance=600)
        
        # Face detection (as visual feedback)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Application state
        self.is_running = True
        
        # Get screen dimensions for mouse tracking
        import tkinter as tk
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        
        print(f"✅ Screen resolution: {self.screen_width}x{self.screen_height}")
    
    def get_mouse_position(self):
        """Get current mouse position"""
        try:
            import win32gui
            return win32gui.GetCursorPos()
        except ImportError:
            # Fallback to center if win32gui not available
            return (self.screen_width // 2, self.screen_height // 2)
    
    def draw_volume_info(self, img, mouse_x, volume_percent):
        """Draw volume information on the image"""
        # Draw instructions
        cv2.putText(img, "Move mouse LEFT-RIGHT to control volume", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw mouse position
        cv2.putText(img, f"Mouse X: {mouse_x}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw volume
        cv2.putText(img, f"Volume: {volume_percent}%", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Draw volume bar
        bar_x, bar_y = 50, 150
        bar_width, bar_height = 400, 30
        
        # Background bar
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 2)
        
        # Filled bar
        filled_width = int((volume_percent / 100) * bar_width)
        if filled_width > 0:
            color = (0, 255, 0) if volume_percent < 50 else (0, 255, 255) if volume_percent < 80 else (0, 0, 255)
            cv2.rectangle(img, (bar_x + 2, bar_y + 2), 
                         (bar_x + filled_width - 2, bar_y + bar_height - 2), color, -1)
        
        # Draw progress indicator
        indicator_x = bar_x + int((mouse_x / self.screen_width) * bar_width)
        cv2.circle(img, (indicator_x, bar_y + bar_height // 2), 8, (255, 255, 255), -1)
        cv2.circle(img, (indicator_x, bar_y + bar_height // 2), 8, (0, 0, 0), 2)
    
    def detect_faces(self, img):
        """Detect faces for visual feedback"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Face Detected!", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return len(faces) > 0
    
    def run(self):
        """Main application loop"""
        print("🚀 Starting Simple Volume Controller...")
        
        while self.is_running:
            # Read frame from camera
            success, img = self.cap.read()
            if not success:
                print("❌ Failed to read from camera")
                break
            
            # Flip image horizontally for mirror effect
            img = cv2.flip(img, 1)
            
            # Get mouse position
            mouse_x, mouse_y = self.get_mouse_position()
            
            # Map mouse X position to volume (0-100%)
            volume_percent = int((mouse_x / self.screen_width) * 100)
            volume_percent = max(0, min(100, volume_percent))
            
            # Calculate equivalent "distance" for volume controller
            # Map volume percent back to distance range
            equivalent_distance = np.interp(volume_percent, [0, 100], 
                                          [self.volume_controller.min_distance, 
                                           self.volume_controller.max_distance])
            
            # Set volume
            success, actual_volume = self.volume_controller.set_volume(equivalent_distance)
            
            # Detect faces for visual feedback
            face_detected = self.detect_faces(img)
            
            # Draw UI
            self.draw_volume_info(img, mouse_x, actual_volume)
            
            # Show face detection status
            status_color = (0, 255, 0) if face_detected else (0, 0, 255)
            status_text = "Face: YES" if face_detected else "Face: NO"
            cv2.putText(img, status_text, (10, img.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Instructions
            cv2.putText(img, "Press 'q' to quit", (img.shape[1] - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the image
            cv2.imshow("Simple Volume Controller", img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                self.is_running = False
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Simple Volume Controller stopped!")


def main():
    """Main entry point"""
    try:
        print("🎵 Simple Volume Controller (OpenCV Workaround)")
        print("This version works without MediaPipe!")
        print()
        
        app = SimpleVolumeController()
        app.run()
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your camera is connected")


if __name__ == "__main__":
    main() 