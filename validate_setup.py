#!/usr/bin/env python
"""
YOLOPv2 GUI Validation Script
Checks all dependencies and configurations before running GUI
"""

import sys
import platform

def check_python():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or version.minor < 7:
        print("  ⚠ Warning: Python 3.7+ recommended")
        return False
    return True

def check_module(name, import_name=None):
    """Check if module can be imported"""
    if import_name is None:
        import_name = name
    try:
        __import__(import_name)
        print(f"✓ {name}")
        return True
    except ImportError:
        print(f"✗ {name} - Install with: pip install {name}")
        return False

def check_files():
    """Check required files exist"""
    import os
    files = [
        'data/weights/yolopv2.pt',
        'utils/utils.py',
        'yolopv2_combined_gui.py',
        'requirements.txt',
    ]
    
    ok = True
    for f in files:
        if os.path.exists(f):
            print(f"✓ {f}")
        else:
            print(f"✗ {f} - Not found")
            ok = False
    return ok

def check_torch_device():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ GPU Not Available - will use CPU (very slow)")
            return False
    except:
        return False

def main():
    print("=" * 60)
    print("YOLOPv2 + YOLOv8 GUI - Dependency Checker")
    print("=" * 60)
    print()
    
    print("System Information:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print()
    
    checks = []
    
    print("Python & Core Libraries:")
    checks.append(check_python())
    checks.append(check_module('NumPy', 'numpy'))
    checks.append(check_module('OpenCV', 'cv2'))
    checks.append(check_module('Pillow', 'PIL'))
    checks.append(check_module('Tkinter', 'tkinter'))
    print()
    
    print("PyTorch & Deep Learning:")
    checks.append(check_module('PyTorch', 'torch'))
    checks.append(check_module('Torchvision', 'torchvision'))
    print()
    
    print("YOLOv8 & Detection:")
    checks.append(check_module('Ultralytics', 'ultralytics'))
    print()
    
    print("GPU Support:")
    check_torch_device()
    print()
    
    print("Required Files:")
    checks.append(check_files())
    print()
    
    if all(checks):
        print("=" * 60)
        print("✓ ALL CHECKS PASSED - Ready to run!")
        print("=" * 60)
        print()
        print("Start GUI with:")
        print("  python yolopv2_combined_gui.py")
        print()
        return 0
    else:
        print("=" * 60)
        print("✗ SOME CHECKS FAILED - See above for details")
        print("=" * 60)
        print()
        print("Fix issues and run again:")
        print("  python validate_setup.py")
        print()
        return 1

if __name__ == '__main__':
    sys.exit(main())
