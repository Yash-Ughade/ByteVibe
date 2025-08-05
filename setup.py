#!/usr/bin/env python3
"""
Setup script for Finger Volume Controller
Helps users install dependencies and run the application
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
    
    print(f"✅ Python {platform.python_version()} is compatible")
    return True


def check_camera():
    """Check if camera is available"""
    print("\n📷 Checking camera availability...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is available")
            cap.release()
            return True
        else:
            print("⚠️  Camera not detected - make sure it's connected")
            return False
    except ImportError:
        print("⚠️  OpenCV not installed yet - will install with dependencies")
        return True


def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Try running: pip install -r requirements.txt manually")
        return False


def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing imports...")
    
    modules = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
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
        print("Please check your installation and try again")
        return False
    
    print("✅ All imports successful!")
    return True


def run_application():
    """Run the main application"""
    print("\n🚀 Starting Finger Volume Controller...")
    print("Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except FileNotFoundError:
        print("❌ main.py not found in current directory")
        return False
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return False
    
    return True


def main():
    """Main setup function"""
    print("=" * 50)
    print("🎵 Finger Volume Controller Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check operating system
    print(f"\n💻 Operating System: {platform.system()} {platform.release()}")
    if platform.system() != "Windows":
        print("⚠️  Volume control is optimized for Windows")
        print("   Some features may not work on other operating systems")
    
    # Check camera
    camera_available = check_camera()
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Test imports
    if not test_imports():
        return
    
    # Show setup complete message
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("=" * 50)
    
    print("\n📋 Usage Instructions:")
    print("1. Position your hand in front of the camera")
    print("2. Use thumb and index finger distance to control volume")
    print("3. Use keyboard shortcuts:")
    print("   - Q/ESC: Quit")
    print("   - L: Toggle finger labels")
    print("   - V: Toggle volume bar")
    print("   - M: Toggle mute")
    
    if not camera_available:
        print("\n⚠️  Camera not detected - please connect camera before running")
    
    # Ask user if they want to run the application
    print("\n" + "=" * 50)
    response = input("Would you like to run the application now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        run_application()
    else:
        print("\n👍 Setup complete! Run 'python main.py' when ready")


if __name__ == "__main__":
    main() 