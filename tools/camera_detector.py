#!/usr/bin/env python3
"""
Camera Detection Tool for macOS
Helps identify available cameras and their properties
"""

import cv2
import time

def detect_cameras():
    """Detect all available cameras and their properties"""
    print("🔍 Camera Detection Tool")
    print("=" * 50)
    
    available_cameras = []
    
    # Test first 10 camera indices
    for i in range(10):
        try:
            print(f"\n📷 Testing camera index {i}...")
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Try to read a frame
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    camera_info = {
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'working': True
                    }
                    available_cameras.append(camera_info)
                    
                    print(f"  ✅ Camera {i}: {width}x{height} @ {fps:.1f}fps - WORKING")
                    
                    # Check if this looks like a built-in MacBook camera
                    if width in [1280, 1920, 640] and height in [720, 1080, 480]:
                        print(f"  🎯 This looks like a built-in MacBook camera!")
                    elif width > 1920 or height > 1080:
                        print(f"  📱 This might be an iPhone/external camera (high resolution)")
                    
                else:
                    print(f"  ❌ Camera {i}: Available but can't read frames")
                
                cap.release()
            else:
                print(f"  ❌ Camera {i}: Not available")
                
        except Exception as e:
            print(f"  ❌ Camera {i}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    if available_cameras:
        print(f"Found {len(available_cameras)} working camera(s):")
        
        for cam in available_cameras:
            resolution_type = ""
            if cam['width'] in [1280, 1920, 640] and cam['height'] in [720, 1080, 480]:
                resolution_type = " (Built-in MacBook camera)"
            elif cam['width'] > 1920 or cam['height'] > 1080:
                resolution_type = " (iPhone/External camera)"
            
            print(f"  📷 Camera {cam['index']}: {cam['width']}x{cam['height']}{resolution_type}")
        
        # Recommend the best camera
        builtin_cameras = [cam for cam in available_cameras 
                          if cam['width'] in [1280, 1920, 640] and cam['height'] in [720, 1080, 480]]
        
        if builtin_cameras:
            recommended = builtin_cameras[0]
            print(f"\n🎯 RECOMMENDED: Use camera {recommended['index']} (Built-in MacBook camera)")
        else:
            print(f"\n🎯 RECOMMENDED: Use camera {available_cameras[0]['index']} (First available)")
            
    else:
        print("❌ No working cameras found!")
    
    return available_cameras

def test_recommended_camera():
    """Test the recommended camera with a brief capture"""
    cameras = detect_cameras()
    
    if not cameras:
        return
    
    # Find recommended camera
    builtin_cameras = [cam for cam in cameras 
                      if cam['width'] in [1280, 1920, 640] and cam['height'] in [720, 1080, 480]]
    
    recommended_index = builtin_cameras[0]['index'] if builtin_cameras else cameras[0]['index']
    
    print(f"\n🧪 Testing recommended camera {recommended_index} for 3 seconds...")
    
    cap = cv2.VideoCapture(recommended_index)
    if cap.isOpened():
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 3.0:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                # You could add face detection here if needed
                
        cap.release()
        fps_actual = frame_count / 3.0
        print(f"✅ Camera {recommended_index} test successful!")
        print(f"   Captured {frame_count} frames in 3 seconds ({fps_actual:.1f} fps)")
    else:
        print(f"❌ Failed to open camera {recommended_index}")

if __name__ == "__main__":
    print("🚀 Starting camera detection...")
    print("This will help identify which camera to use instead of your iPhone")
    print()
    
    test_recommended_camera()
    
    print("\n💡 To force the app to use a specific camera:")
    print("   Edit main.py and set: CAMERA_OVERRIDE = X")
    print("   (where X is your preferred camera index)")
