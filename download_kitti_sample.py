#!/usr/bin/env python3
"""
Simple KITTI sample downloader
Downloads a small KITTI sequence for testing YOLOPv2
"""

import os
import urllib.request
import zipfile
import shutil

def download_kitti_sample():
    """Download a sample KITTI sequence"""
    
    kitti_dir = "data/kitti"
    os.makedirs(kitti_dir, exist_ok=True)
    
    # Try to download from GitHub mirror (more reliable)
    print("Attempting to download KITTI sample...")
    
    try:
        # Use a lightweight KITTI sample from a reliable source
        urls = [
            # Option 1: Small KITTI sample from alternative host
            ("https://github.com/yzhao062/anomaly-detection-resources/raw/master/kitti_sample_data.zip", 
             "kitti_sample.zip"),
        ]
        
        for url, filename in urls:
            zip_path = os.path.join(kitti_dir, filename)
            
            if os.path.exists(zip_path):
                print(f"File already exists: {zip_path}")
                extract_to = os.path.join(kitti_dir, "sample")
                if not os.path.exists(extract_to):
                    print(f"Extracting to {extract_to}...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                return True
            
            print(f"Downloading from: {url}")
            try:
                urllib.request.urlretrieve(url, zip_path)
                print(f"✓ Downloaded to {zip_path}")
                
                # Extract
                extract_to = os.path.join(kitti_dir, "sample")
                print(f"Extracting to {extract_to}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                
                print("✓ KITTI sample ready!")
                return True
            except Exception as e:
                print(f"✗ Failed: {e}")
                continue
        
        # If all automated downloads fail, provide instructions
        print("\n" + "="*60)
        print("AUTOMATED DOWNLOAD FAILED - MANUAL SETUP REQUIRED")
        print("="*60)
        print("\nTo test with KITTI data, you have two options:\n")
        
        print("OPTION 1: Use test images folder")
        print("  $ mkdir -p images")
        print("  $ cp your_images/*.jpg images/")
        print("  Then click PHOTO GALLERY in the GUI\n")
        
        print("OPTION 2: Download KITTI manually")
        print("  1. Visit: http://www.cvlibs-dataset.net/download.php")
        print("  2. Download 'Raw Data' for any date (2011_09_26 recommended)")
        print("  3. Create folder: mkdir -p data/kitti/2011_09_26")
        print("  4. Extract contents to that folder")
        print("  5. The structure should be:")
        print("     data/kitti/2011_09_26/")
        print("     └── 2011_09_26_drive_0001_sync/")
        print("         └── image_00/")
        print("             └── data/")
        print("                 └── 0000000000.png")
        print("                 └── 0000000001.png")
        print("                 └── ...\n")
        
        print("Then click KITTI DATASET button in the GUI to load the sequence.\n")
        
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    success = download_kitti_sample()
    exit(0 if success else 1)
