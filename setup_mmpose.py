#!/usr/bin/env python3
"""
Setup script for Finger Volume Controller with MMPose
Python 3.13 compatible version
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"   Current version: {platform.python_version()}")
        return False
    
    if sys.version_info >= (3, 13):
        print(f"✅ Python {platform.python_version()} detected - using MMPose for compatibility")
    else:
        print(f"✅ Python {platform.python_version()} is compatible")
    return True

def install_mmpose_dependencies():
    """Install MMPose dependencies"""
    print("\n📦 Installing MMPose dependencies...")
    
    try:
        # Install MMPose and dependencies
        print("Installing MMPose...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_mmpose.txt"
        ])
        print("✅ MMPose dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install MMPose dependencies: {e}")
        print("Trying alternative installation...")
        
        # Try installing MMPose directly
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "mmpose", "mmcv", "mmengine"
            ])
            print("✅ MMPose installed successfully!")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ Alternative installation also failed: {e2}")
            print("Please try manual installation:")
            print("1. pip install torch torchvision")
            print("2. pip install mmpose")
            return False

def test_mmpose_imports():
    """Test if MMPose can be imported"""
    print("\n🧪 Testing MMPose imports...")
    
    try:
        import mmpose
        print(f"✅ MMPose {mmpose.__version__} imported successfully")
        
        # Test core imports
        from mmpose.apis import inference_topdown, init_model
        print("✅ MMPose APIs imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import MMPose: {e}")
        return False

def test_basic_imports():
    """Test basic dependencies"""
    print("\n🔍 Testing basic imports...")
    
    modules = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pycaw.pycaw", "pycaw"),
        ("comtypes", "comtypes")
    ]
    
    failed_imports = []
    
    for module, name in modules:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    return True

def create_demo_script():
    """Create a simple demo script to test the setup"""
    demo_content = '''#!/usr/bin/env python3
"""
Simple demo to test MMPose hand detection setup
"""

try:
    from hand_detector_mmpose import HandDetectorMMPose
    import cv2
    import numpy as np
    
    print("🎯 Testing MMPose Hand Detector...")
    
    # Create a test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Initialize detector
    detector = HandDetectorMMPose(detection_confidence=0.7, device='cpu')
    
    print("✅ MMPose Hand Detector initialized successfully!")
    print("🎵 Your Finger Volume Controller is ready to use!")
    print("📝 Run 'python main.py' to start the application")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install MMPose: pip install mmpose")
except Exception as e:
    print(f"⚠️ Setup test completed with warnings: {e}")
    print("The application should still work in fallback mode")
'''
    
    with open("test_mmpose_setup.py", "w") as f:
        f.write(demo_content)
    
    print("📝 Created test script: test_mmpose_setup.py")

def main():
    """Main setup function"""
    print("=" * 60)
    print("🎵 Finger Volume Controller - MMPose Setup")
    print("🐍 Python 3.13 Compatible Version")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check operating system
    print(f"\n💻 Operating System: {platform.system()} {platform.release()}")
    if platform.system() != "Windows":
        print("⚠️ Volume control is optimized for Windows")
        print("   Some features may not work on other operating systems")
    
    # Install MMPose dependencies
    if not install_mmpose_dependencies():
        print("\n❌ Failed to install MMPose. Trying basic dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "opencv-python", "numpy", "pycaw", "comtypes"
            ])
            print("✅ Basic dependencies installed. Application will run in fallback mode.")
        except subprocess.CalledProcessError:
            print("❌ Failed to install basic dependencies")
            return
    
    # Test imports
    mmpose_available = test_mmpose_imports()
    basic_available = test_basic_imports()
    
    if not basic_available:
        print("\n❌ Basic dependencies not available. Setup failed.")
        return
    
    # Create demo script
    create_demo_script()
    
    # Show setup complete message
    print("\n" + "=" * 60)
    if mmpose_available:
        print("✅ MMPose setup completed successfully!")
        print("🚀 You're ready to use the advanced hand detection!")
    else:
        print("⚠️ Setup completed with basic dependencies")
        print("🔄 Application will run in fallback mode")
    print("=" * 60)
    
    print("\n📋 Usage Instructions:")
    print("1. Position your hand in front of the camera")
    print("2. Use thumb and index finger distance to control volume")
    print("3. Keyboard shortcuts:")
    print("   - Q/ESC: Quit")
    print("   - L: Toggle finger labels")
    print("   - V: Toggle volume bar")
    print("   - M: Toggle mute")
    
    print("\n🎯 Ready to run:")
    print("   python main.py")
    
    # Ask user if they want to test the setup
    print("\n" + "=" * 60)
    response = input("Would you like to test the setup now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        try:
            subprocess.run([sys.executable, "test_mmpose_setup.py"])
        except FileNotFoundError:
            print("❌ Test script not found")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print("\n👍 Setup complete! Run 'python main.py' when ready")

if __name__ == "__main__":
    main() 