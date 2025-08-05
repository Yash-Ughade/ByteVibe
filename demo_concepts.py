#!/usr/bin/env python3
"""
Demo script showcasing ML and Neural Network concepts
in the Finger Volume Controller project
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time


def demonstrate_mediapipe_model():
    """
    Demonstrate MediaPipe's neural network architecture and hand detection
    """
    print("🧠 MediaPipe Neural Network Demo")
    print("=" * 40)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    print("📊 Model Information:")
    print(f"  - Input: 256x256 RGB image")
    print(f"  - Output: 21 3D hand landmarks")
    print(f"  - Architecture: Lightweight CNN")
    print(f"  - Detection Confidence: 70%")
    print(f"  - Tracking Confidence: 50%")
    
    # Explain the landmark structure
    print("\n🖐️ Hand Landmark Structure (21 points):")
    landmark_groups = {
        "Wrist": [0],
        "Thumb": [1, 2, 3, 4],
        "Index": [5, 6, 7, 8],
        "Middle": [9, 10, 11, 12],
        "Ring": [13, 14, 15, 16],
        "Pinky": [17, 18, 19, 20]
    }
    
    for finger, landmarks in landmark_groups.items():
        print(f"  - {finger}: landmarks {landmarks}")
    
    return hands


def demonstrate_distance_calculation():
    """
    Demonstrate the mathematical concepts used for distance calculation
    """
    print("\n📏 Distance Calculation Demo")
    print("=" * 40)
    
    # Sample coordinates (thumb tip and index tip)
    thumb_pos = (100, 150)
    index_pos = (200, 180)
    
    print(f"📍 Sample Coordinates:")
    print(f"  - Thumb tip: {thumb_pos}")
    print(f"  - Index tip: {index_pos}")
    
    # Calculate Euclidean distance
    x1, y1 = thumb_pos
    x2, y2 = index_pos
    distance = math.hypot(x2 - x1, y2 - y1)
    
    print(f"\n🧮 Euclidean Distance Formula:")
    print(f"  distance = √((x2-x1)² + (y2-y1)²)")
    print(f"  distance = √(({x2}-{x1})² + ({y2}-{y1})²)")
    print(f"  distance = √({x2-x1}² + {y2-y1}²)")
    print(f"  distance = √({(x2-x1)**2} + {(y2-y1)**2})")
    print(f"  distance = √{(x2-x1)**2 + (y2-y1)**2}")
    print(f"  distance = {distance:.2f} pixels")
    
    return distance


def demonstrate_volume_mapping():
    """
    Demonstrate the linear interpolation used for volume mapping
    """
    print("\n🔊 Volume Mapping Demo")
    print("=" * 40)
    
    # Distance range and volume range
    min_distance = 30
    max_distance = 300
    min_volume = 0
    max_volume = 100
    
    print(f"📊 Mapping Parameters:")
    print(f"  - Distance range: {min_distance} - {max_distance} pixels")
    print(f"  - Volume range: {min_volume} - {max_volume}%")
    
    # Test different distances
    test_distances = [30, 50, 100, 150, 200, 250, 300]
    
    print(f"\n🔄 Linear Interpolation Examples:")
    print(f"{'Distance':<10} {'Volume':<8} {'Formula'}")
    print("-" * 50)
    
    for distance in test_distances:
        # Clamp distance to valid range
        clamped_distance = max(min_distance, min(distance, max_distance))
        
        # Linear interpolation formula
        volume = np.interp(clamped_distance, [min_distance, max_distance], [min_volume, max_volume])
        
        print(f"{distance:<10} {volume:<8.1f}% np.interp({clamped_distance}, [{min_distance}, {max_distance}], [{min_volume}, {max_volume}])")


def demonstrate_smoothing_algorithm():
    """
    Demonstrate the moving average smoothing algorithm
    """
    print("\n📈 Smoothing Algorithm Demo")
    print("=" * 40)
    
    # Simulated noisy distance measurements
    raw_distances = [120, 135, 110, 140, 125, 145, 115, 130, 120, 135]
    window_size = 5
    
    print(f"📊 Raw Distance Measurements (noisy):")
    print(f"  {raw_distances}")
    
    print(f"\n🔄 Moving Average Smoothing (window size = {window_size}):")
    print(f"{'Index':<6} {'Raw':<6} {'Window':<20} {'Smoothed':<10}")
    print("-" * 50)
    
    smoothed_distances = []
    for i in range(len(raw_distances)):
        # Get window of recent measurements
        start_idx = max(0, i - window_size + 1)
        window = raw_distances[start_idx:i+1]
        
        # Calculate moving average
        smoothed = np.mean(window)
        smoothed_distances.append(smoothed)
        
        window_str = str(window) if len(window) <= 5 else f"[...{len(window)} values]"
        print(f"{i:<6} {raw_distances[i]:<6} {window_str:<20} {smoothed:<10.1f}")
    
    print(f"\n✨ Noise Reduction:")
    print(f"  - Raw data standard deviation: {np.std(raw_distances):.2f}")
    print(f"  - Smoothed data standard deviation: {np.std(smoothed_distances):.2f}")
    print(f"  - Noise reduction: {(1 - np.std(smoothed_distances)/np.std(raw_distances))*100:.1f}%")


def demonstrate_neural_network_pipeline():
    """
    Demonstrate the complete neural network processing pipeline
    """
    print("\n🔬 Neural Network Processing Pipeline")
    print("=" * 40)
    
    pipeline_steps = [
        "1. 📷 Camera Input",
        "   - Capture RGB frame from camera",
        "   - Resolution: 1280x720 pixels",
        "",
        "2. 🔄 Preprocessing",
        "   - Convert BGR to RGB color space",
        "   - Normalize pixel values to [0, 1]",
        "   - Resize to 256x256 for model input",
        "",
        "3. 🧠 Neural Network Inference",
        "   - Feed image through MediaPipe model",
        "   - CNN processes spatial features",
        "   - Output 21 hand landmarks (x, y, z)",
        "",
        "4. 📊 Post-processing",
        "   - Convert normalized coordinates to pixels",
        "   - Apply confidence thresholding",
        "   - Track landmarks across frames",
        "",
        "5. 🎯 Gesture Recognition",
        "   - Identify finger positions",
        "   - Calculate thumb-index distance",
        "   - Apply smoothing algorithm",
        "",
        "6. 🔊 Volume Control",
        "   - Map distance to volume percentage",
        "   - Update system volume via API",
        "   - Provide visual feedback"
    ]
    
    for step in pipeline_steps:
        print(step)


def run_interactive_demo():
    """
    Run an interactive demo with live camera feed
    """
    print("\n🎥 Interactive Neural Network Demo")
    print("=" * 40)
    print("Press 'q' to exit, 's' to save frame")
    
    # Initialize camera and MediaPipe
    cap = cv2.VideoCapture(0)
    hands = demonstrate_mediapipe_model()
    mp_draw = mp.solutions.drawing_utils
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process through neural network
        results = hands.process(img_rgb)
        
        # Draw information
        cv2.putText(img, "Neural Network Demo", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        if current_time - start_time >= 1.0:
            fps = frame_count / (current_time - start_time)
            frame_count = 0
            start_time = current_time
        
        if 'fps' in locals():
            cv2.putText(img, f"FPS: {fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Show landmark count
                cv2.putText(img, f"Landmarks: 21", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show confidence
                cv2.putText(img, "Model: MediaPipe CNN", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        cv2.imshow("Neural Network Hand Detection Demo", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"demo_frame_{int(time.time())}.jpg", img)
            print("Frame saved!")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Main demo function
    """
    print("🎵 Finger Volume Controller - ML Concepts Demo")
    print("=" * 60)
    print("This demo showcases the machine learning and neural network")
    print("concepts used in the Finger Volume Controller project.")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_mediapipe_model()
    demonstrate_distance_calculation()
    demonstrate_volume_mapping()
    demonstrate_smoothing_algorithm()
    demonstrate_neural_network_pipeline()
    
    print("\n" + "=" * 60)
    print("🎓 Key ML/Neural Network Concepts Covered:")
    print("  ✅ Convolutional Neural Networks (CNN)")
    print("  ✅ Real-time inference and processing")
    print("  ✅ Computer vision preprocessing")
    print("  ✅ Landmark detection and tracking")
    print("  ✅ Signal processing and smoothing")
    print("  ✅ Mathematical transformations")
    print("=" * 60)
    
    # Ask if user wants to see interactive demo
    response = input("\nWould you like to see the interactive camera demo? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        run_interactive_demo()
    
    print("\n👨‍🏫 Demo completed! You now understand the ML concepts behind the project.")


if __name__ == "__main__":
    main() 