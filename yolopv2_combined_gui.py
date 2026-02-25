"""
YOLOPv2 + YOLOv8 Combined Perception System
Features:
- Road segmentation (green)
- Lane lines (red/orange)
- Vehicle instance segmentation (blue) with counter
- People segmentation (orange) with counter
- Bicycle segmentation (purple) with counter
- Traffic light detection (yellow)
- Live camera feed (laptop/phone)
- Photo gallery processing
"""

import tkinter as tk
from tkinter import messagebox
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
import glob
import queue
import urllib.request
import zipfile

# YOLOPv2 imports
from utils.utils import (
    select_device,
    driving_area_mask,
    lane_line_mask,
)

# YOLOv8 for comprehensive detection
try:
    from ultralytics import YOLO
    YOLOV8_AVAILABLE = True
except ImportError:
    YOLOV8_AVAILABLE = False
    print("Note: ultralytics not available - using YOLOPv2 segmentation only")


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, stride=32):
    """Resize and pad image"""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


class CombinedPerceptionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOPv2 + YOLOv8 Perception System")
        self.root.geometry("1600x1200")
        self.root.configure(bg='#1a1a2e')
        
        # CONFIGURATION
        self.DROIDCAM_IP = "10.110.191.74"  # Updated IP
        self.DROIDCAM_PORT = "4747"
        self.YOLOPV2_WEIGHTS = "data/weights/yolopv2.pt"
        self.IMAGES_FOLDER = "images"
        self.VIDEOS_FOLDER = "videos"
        self.KITTI_FOLDER = "data/kitti"
        self.IMG_SIZE = 640
        self.CONF_THRESHOLD = 0.25
        self.IOU_THRESHOLD = 0.45
        
        # Colors (BGR format for OpenCV)
        self.COLOR_ROAD = (0, 200, 0)        # Green for drivable area
        self.COLOR_LANE = (0, 0, 255)        # Red for lane lines (BGR: Blue channel is 0, Red is 255)
        self.COLOR_VEHICLE = (255, 150, 0)   # Cyan/Blue for vehicles
        self.COLOR_PERSON = (0, 165, 255)    # Orange for people
        self.COLOR_BICYCLE = (255, 0, 255)   # Magenta for bicycles
        self.COLOR_TRAFFIC_LIGHT = (0, 255, 255)  # Yellow for traffic lights
        self.COLOR_ANOMALY = (255, 255, 0)   # Cyan for potholes/anomalies
        
        # State variables
        self.current_mode = tk.StringVar(value="none")
        self.is_running = False
        self.cap = None
        self.yolopv2_model = None
        self.yolov8_detection_model = None
        self.yolov8_segmentation_model = None
        self.device = None
        self.half = False
        
        # Detection tracking
        self.detected_classes = {}  # Track which classes are detected
        self.vehicle_count = 0
        self.person_count = 0
        self.bicycle_count = 0
        self.traffic_light_count = 0
        self.anomaly_count = 0
        
        # Canny edge detection variables
        self.canny_enabled = tk.BooleanVar(value=False)
        self.canny_threshold1 = 50    # Lower threshold
        self.canny_threshold2 = 150   # Upper threshold
        self.canny_blur_ksize = 5     # Gaussian blur kernel size
        self.show_canny_only = tk.BooleanVar(value=False)  # Show edges-only view
        self.canny_on_lanes = tk.BooleanVar(value=False)   # Apply to lanes instead of road
        self.last_road_mask = None    # Store for visualization
        self.last_canny_edges = None  # Store canny result
        
        # Photo gallery variables
        self.image_list = []
        self.current_image_index = 0
        
        # Video variables
        self.video_frames = []
        self.current_frame_index = 0
        self.video_thread = None
        
        # Performance tracking
        self.fps = 0
        self.inference_time = 0
        self.last_processed_frame = None
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Create images folder if not exists
        if not os.path.exists(self.IMAGES_FOLDER):
            os.makedirs(self.IMAGES_FOLDER)
        
        # Create videos folder if not exists
        if not os.path.exists(self.VIDEOS_FOLDER):
            os.makedirs(self.VIDEOS_FOLDER)
        
        # Create KITTI folder if not exists
        if not os.path.exists(self.KITTI_FOLDER):
            os.makedirs(self.KITTI_FOLDER)
        
        # Initialize UI
        self.setup_ui()
        
        # Load models in background
        self.models_loaded = False
        self.loading_thread = threading.Thread(target=self.load_models, daemon=True)
        self.loading_thread.start()
        
        # Bind keyboard events
        self.root.bind('<Left>', self.previous_image)
        self.root.bind('<Right>', self.next_image)
        self.root.bind('<Escape>', self.on_closing)
        self.root.bind('q', self.on_closing)
        self.root.bind('s', self.save_screenshot)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container with two columns
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side (video + controls)
        left_frame = tk.Frame(main_frame, bg='#1a1a2e')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right side (stats panel)
        right_frame = tk.Frame(main_frame, bg='#16213e', width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(left_frame, text="PERCEPTION SYSTEM",
                               font=('Helvetica', 24, 'bold'), fg='#00ff88', bg='#1a1a2e')
        title_label.pack(pady=(0, 10))
        
        # Status bar
        self.status_frame = tk.Frame(left_frame, bg='#16213e')
        self.status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(self.status_frame, text="Loading models...",
                                     font=('Helvetica', 12), fg='#ffd700', bg='#16213e', padx=10, pady=5)
        self.status_label.pack(side=tk.LEFT)
        
        self.fps_label = tk.Label(self.status_frame, text="FPS: --",
                                  font=('Helvetica', 12, 'bold'), fg='#00ff88', bg='#16213e', padx=10, pady=5)
        self.fps_label.pack(side=tk.RIGHT)
        
        self.inference_label = tk.Label(self.status_frame, text="Inference: -- ms",
                                        font=('Helvetica', 12), fg='#00d4ff', bg='#16213e', padx=10, pady=5)
        self.inference_label.pack(side=tk.RIGHT)
        
        # Control panel
        control_frame = tk.Frame(left_frame, bg='#16213e', pady=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_style = {'font': ('Helvetica', 11, 'bold'), 'width': 14, 'height': 2}
        
        self.btn_laptop = tk.Button(control_frame, text="LAPTOP CAMERA", command=self.start_laptop_camera,
                                    bg='#4a69bd', fg='white', **btn_style)
        self.btn_laptop.pack(side=tk.LEFT, padx=4, pady=5)
        
        self.btn_phone = tk.Button(control_frame, text="PHONE CAMERA", command=self.start_phone_camera,
                                   bg='#e55039', fg='white', **btn_style)
        self.btn_phone.pack(side=tk.LEFT, padx=4, pady=5)
        
        self.btn_photos = tk.Button(control_frame, text="PHOTO GALLERY", command=self.start_photo_mode,
                                    bg='#78e08f', fg='black', **btn_style)
        self.btn_photos.pack(side=tk.LEFT, padx=4, pady=5)
        
        self.btn_video = tk.Button(control_frame, text="VIDEO PLAYER", command=self.start_video_mode,
                                   bg='#9b59b6', fg='white', **btn_style)
        self.btn_video.pack(side=tk.LEFT, padx=4, pady=5)
        
        # NEW: Dedicated Canny Edge Detection button
        self.btn_canny_mode = tk.Button(control_frame, text="CANNY EDGES", command=self.start_canny_mode,
                                        bg='#00d4ff', fg='black', **btn_style)
        self.btn_canny_mode.pack(side=tk.LEFT, padx=4, pady=5)
        
        self.btn_stop = tk.Button(control_frame, text="STOP", command=self.stop_all,
                                  bg='#c0392b', fg='white', **btn_style)
        self.btn_stop.pack(side=tk.LEFT, padx=4, pady=5)
        
        # Video display area
        self.display_frame = tk.Frame(left_frame, bg='#0f0f1a', relief=tk.SUNKEN, bd=2)
        self.display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.display_frame, bg='#0f0f1a', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.create_text(500, 300,
            text="Select a mode to begin\n\nLAPTOP CAMERA - Built-in webcam\nPHONE CAMERA - DroidCam\nPHOTO GALLERY - Browse images\n\nArrow keys: Navigate | S: Screenshot | Q: Quit",
            font=('Helvetica', 14), fill='#888888', justify=tk.CENTER, tags='welcome')
        
        # Photo info label
        self.photo_info_label = tk.Label(left_frame, text="", font=('Helvetica', 11),
                                         fg='#00d4ff', bg='#1a1a2e')
        self.photo_info_label.pack(pady=5)
        
        # Confidence slider
        slider_frame = tk.Frame(left_frame, bg='#16213e', pady=5)
        slider_frame.pack(fill=tk.X)
        
        tk.Label(slider_frame, text="Confidence:", font=('Helvetica', 10),
                 fg='white', bg='#16213e').pack(side=tk.LEFT, padx=10)
        
        self.conf_slider = tk.Scale(slider_frame, from_=0.1, to=0.9, resolution=0.05,
                                    orient=tk.HORIZONTAL, length=150, bg='#16213e', fg='white',
                                    highlightthickness=0)
        self.conf_slider.set(0.3)
        self.conf_slider.pack(side=tk.LEFT, padx=5)
        
        tk.Label(slider_frame, text=f"DroidCam: {self.DROIDCAM_IP}:{self.DROIDCAM_PORT}",
                 font=('Helvetica', 10), fg='#888888', bg='#16213e').pack(side=tk.RIGHT, padx=10)
        
        # ========== CANNY EDGE DETECTION CONTROLS ==========
        canny_frame = tk.Frame(left_frame, bg='#16213e', pady=8)
        canny_frame.pack(fill=tk.X)
        
        # Canny toggle button
        self.btn_canny = tk.Checkbutton(canny_frame, text="🔲 CANNY EDGES",
                                         variable=self.canny_enabled,
                                         font=('Helvetica', 11, 'bold'),
                                         fg='#00ffff', bg='#16213e', selectcolor='#2d3436',
                                         activebackground='#16213e', activeforeground='#00ffff')
        self.btn_canny.pack(side=tk.LEFT, padx=10)
        
        # Canny target: Road vs Lanes
        self.btn_canny_lanes = tk.Checkbutton(canny_frame, text="Lanes Only",
                                               variable=self.canny_on_lanes,
                                               font=('Helvetica', 10),
                                               fg='#ff6b6b', bg='#16213e', selectcolor='#2d3436',
                                               activebackground='#16213e', activeforeground='#ff6b6b')
        self.btn_canny_lanes.pack(side=tk.LEFT, padx=5)
        
        # Show edges-only view toggle
        self.btn_edges_only = tk.Checkbutton(canny_frame, text="Edges Only View",
                                              variable=self.show_canny_only,
                                              font=('Helvetica', 10),
                                              fg='#ffd93d', bg='#16213e', selectcolor='#2d3436',
                                              activebackground='#16213e', activeforeground='#ffd93d')
        self.btn_edges_only.pack(side=tk.LEFT, padx=5)
        
        # Canny threshold 1 slider
        tk.Label(canny_frame, text="Low:", font=('Helvetica', 9),
                 fg='white', bg='#16213e').pack(side=tk.LEFT, padx=(15, 2))
        
        self.canny1_slider = tk.Scale(canny_frame, from_=10, to=200, resolution=5,
                                       orient=tk.HORIZONTAL, length=80, bg='#16213e', fg='white',
                                       highlightthickness=0, command=self.update_canny_threshold1)
        self.canny1_slider.set(50)
        self.canny1_slider.pack(side=tk.LEFT, padx=2)
        
        # Canny threshold 2 slider
        tk.Label(canny_frame, text="High:", font=('Helvetica', 9),
                 fg='white', bg='#16213e').pack(side=tk.LEFT, padx=(10, 2))
        
        self.canny2_slider = tk.Scale(canny_frame, from_=50, to=400, resolution=10,
                                       orient=tk.HORIZONTAL, length=80, bg='#16213e', fg='white',
                                       highlightthickness=0, command=self.update_canny_threshold2)
        self.canny2_slider.set(150)
        self.canny2_slider.pack(side=tk.LEFT, padx=2)
        
        # Blur kernel size slider
        tk.Label(canny_frame, text="Blur:", font=('Helvetica', 9),
                 fg='white', bg='#16213e').pack(side=tk.LEFT, padx=(10, 2))
        
        self.blur_slider = tk.Scale(canny_frame, from_=1, to=15, resolution=2,
                                     orient=tk.HORIZONTAL, length=60, bg='#16213e', fg='white',
                                     highlightthickness=0, command=self.update_blur_ksize)
        self.blur_slider.set(5)
        self.blur_slider.pack(side=tk.LEFT, padx=2)
        
        # ========== RIGHT PANEL (Stats) ==========
        stats_title = tk.Label(right_frame, text="DETECTION STATS",
                               font=('Helvetica', 16, 'bold'), fg='#00ff88', bg='#16213e')
        stats_title.pack(pady=(20, 20))
        
        # Vehicle counter
        vehicle_frame = tk.Frame(right_frame, bg='#16213e')
        vehicle_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(vehicle_frame, text="🚗 VEHICLES", font=('Helvetica', 12, 'bold'),
                 fg='#00d4ff', bg='#16213e').pack(anchor='w')
        
        self.vehicle_count_label = tk.Label(vehicle_frame, text="0",
                                            font=('Helvetica', 48, 'bold'), fg='#00d4ff', bg='#16213e')
        self.vehicle_count_label.pack()
        
        # Color indicator for vehicles
        vehicle_color_frame = tk.Frame(vehicle_frame, bg='#FF9600', width=200, height=10)
        vehicle_color_frame.pack(fill=tk.X, pady=5)
        
        # Person counter
        person_frame = tk.Frame(right_frame, bg='#16213e')
        person_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(person_frame, text="🚶 PEOPLE", font=('Helvetica', 12, 'bold'),
                 fg='#ff8c00', bg='#16213e').pack(anchor='w')
        
        self.person_count_label = tk.Label(person_frame, text="0",
                                           font=('Helvetica', 48, 'bold'), fg='#ff8c00', bg='#16213e')
        self.person_count_label.pack()
        
        # Color indicator for people
        person_color_frame = tk.Frame(person_frame, bg='#FF8C00', width=200, height=10)
        person_color_frame.pack(fill=tk.X, pady=5)
        
        # Bicycle counter
        bicycle_frame = tk.Frame(right_frame, bg='#16213e')
        bicycle_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(bicycle_frame, text="🚲 BICYCLES", font=('Helvetica', 12, 'bold'),
                 fg='#ff00ff', bg='#16213e').pack(anchor='w')
        
        self.bicycle_count_label = tk.Label(bicycle_frame, text="0",
                                            font=('Helvetica', 48, 'bold'), fg='#ff00ff', bg='#16213e')
        self.bicycle_count_label.pack()
        
        # Color indicator for bicycles
        bicycle_color_frame = tk.Frame(bicycle_frame, bg='#FF00FF', width=200, height=10)
        bicycle_color_frame.pack(fill=tk.X, pady=5)
        
        # Anomaly counter
        anomaly_frame = tk.Frame(right_frame, bg='#16213e')
        anomaly_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(anomaly_frame, text="⚠️ ANOMALIES", font=('Helvetica', 12, 'bold'),
                 fg='#00ffff', bg='#16213e').pack(anchor='w')
        
        self.anomaly_count_label = tk.Label(anomaly_frame, text="0",
                                            font=('Helvetica', 48, 'bold'), fg='#00ffff', bg='#16213e')
        self.anomaly_count_label.pack()
        
        # Color indicator for anomalies
        anomaly_color_frame = tk.Frame(anomaly_frame, bg='#00FFFF', width=200, height=10)
        anomaly_color_frame.pack(fill=tk.X, pady=5)
        
        # Legend
        legend_frame = tk.Frame(right_frame, bg='#16213e')
        legend_frame.pack(fill=tk.X, padx=10, pady=(30, 10))
        
        tk.Label(legend_frame, text="LEGEND", font=('Helvetica', 12, 'bold'),
                 fg='white', bg='#16213e').pack(anchor='w', pady=(0, 10))
        
        self.legend_labels = {}
        legend_items = {
            'road': ("■ Road Area", (0, 200, 0)),
            'lane': ("■ Lane Lines", (0, 0, 255)),
            'vehicle': ("■ Vehicles", (255, 150, 0)),
            'person': ("■ People", (0, 165, 255)),
            'bicycle': ("■ Bicycles", (255, 0, 255)),
            'traffic': ("■ Traffic Lights", (0, 255, 255)),
            'anomaly': ("■ Potholes/Anomalies", (255, 255, 0)),
            'canny': ("■ Canny Edges", (0, 255, 255)),
        }
        
        for key, (text, color) in legend_items.items():
            label = tk.Label(legend_frame, text=text, font=('Helvetica', 10),
                            fg='#666666', bg='#16213e')
            label.pack(anchor='w', pady=2)
            self.legend_labels[key] = label
            self.detected_classes[key] = False
    
    def load_models(self):
        """Load both YOLOPv2 and YOLOv8 models"""
        try:
            self.update_status("Initializing device...")
            # RTX 5090 requires special PyTorch build for sm_120
            # For now, use CPU. TODO: Build/install PyTorch with sm_120 support
            try:
                if torch.cuda.is_available():
                    # Test if CUDA works with this GPU
                    test_tensor = torch.zeros(1, 3, 32, 32).to('cuda:0')
                    self.device = select_device('0')
                    self.half = True
                    print(f"✓ GPU Mode: {torch.cuda.get_device_name(0)}")
                else:
                    raise RuntimeError("CUDA not available")
            except Exception as e:
                print(f"⚠ GPU init failed ({type(e).__name__}), falling back to CPU")
                self.device = select_device('cpu')
                self.half = False
                self.update_status("GPU incompatible, using CPU")
            
            # Load YOLOPv2 for road/lane segmentation
            self.update_status("Loading YOLOPv2 (roads/lanes)...")
            if self.device.type == 'cpu':
                self.yolopv2_model = torch.jit.load(self.YOLOPV2_WEIGHTS, map_location='cpu')
            else:
                self.yolopv2_model = torch.jit.load(self.YOLOPV2_WEIGHTS)
            
            self.yolopv2_model = self.yolopv2_model.to(self.device)
            if self.half:
                self.yolopv2_model.half()
            self.yolopv2_model.eval()
            
            # Warmup YOLOPv2
            self.update_status("Warming up YOLOPv2...")
            warmup_input = torch.zeros(1, 3, self.IMG_SIZE, self.IMG_SIZE).to(self.device)
            if self.half:
                warmup_input = warmup_input.half()
            with torch.no_grad():
                self.yolopv2_model(warmup_input)
            print("✓ YOLOPv2 loaded and warmed up")
            
            # Load YOLOv8 models if available
            if YOLOV8_AVAILABLE:
                try:
                    self.update_status("Loading YOLOv8 (people/vehicle detection)...")
                    from ultralytics import YOLO
                    # Use yolov8m-seg for GPU, yolov8n-seg for CPU
                    model_name = 'yolov8m-seg.pt' if self.device.type == 'cuda' else 'yolov8n-seg.pt'
                    self.yolov8_segmentation_model = YOLO(model_name)
                    # Explicitly set device
                    self.yolov8_segmentation_model.to(self.device)
                    self.update_status("Models loaded! YOLOPv2 + YOLOv8 ready")
                    print(f"✓ YOLOv8 ({model_name}) loaded on {self.device}")
                except Exception as e:
                    print(f"YOLOv8 load error: {e}")
                    self.yolov8_segmentation_model = None
                    self.update_status("Models loaded! (YOLOPv2 only)")
            else:
                self.update_status("Models loaded! (YOLOPv2 only)")
            
            self.models_loaded = True
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            print(f"Model loading error: {e}")
            import traceback
            traceback.print_exc()

    
    def update_status(self, text):
        try:
            self.root.after(0, lambda: self.status_label.config(text=text))
        except:
            pass
    
    # ========== CANNY EDGE DETECTION METHODS ==========
    def update_canny_threshold1(self, val):
        """Update Canny lower threshold"""
        self.canny_threshold1 = int(float(val))
    
    def update_canny_threshold2(self, val):
        """Update Canny upper threshold"""
        self.canny_threshold2 = int(float(val))
    
    def update_blur_ksize(self, val):
        """Update Gaussian blur kernel size (must be odd)"""
        ksize = int(float(val))
        self.canny_blur_ksize = ksize if ksize % 2 == 1 else ksize + 1
    
    def apply_masked_canny(self, frame, mask):
        """
        Apply Canny edge detection ONLY within the road/lane mask region.
        This prevents detecting edges from buildings, trees, cars, etc.
        
        OpenCV Pipeline:
        1. Convert to grayscale
        2. Apply Gaussian blur (reduces noise for cleaner edges)
        3. Run Canny edge detector
        4. Mask the edges to keep only road/lane region
        5. Apply morphological operations to clean up edges
        
        Args:
            frame: BGR image (1280x720)
            mask: Binary mask (any size, will be resized)
        
        Returns:
            edges_masked: Clean Canny edges within mask (uint8, same size as frame)
            mask_resized: The resized mask used (uint8, 0-255)
        """
        h, w = frame.shape[:2]
        
        # Step 1: Ensure mask matches frame dimensions
        if mask.shape[:2] != (h, w):
            mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask.astype(np.uint8)
        
        # Ensure mask is binary 0-255
        mask_binary = (mask_resized > 0).astype(np.uint8) * 255
        
        # Step 2: Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 3: Apply Gaussian blur to reduce noise
        # This is critical for clean Canny edges
        ksize = self.canny_blur_ksize
        if ksize > 1:
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=0)
        else:
            blurred = gray
        
        # Step 4: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This improves edge detection on roads with varying lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Step 5: Apply Canny edge detection
        edges = cv2.Canny(enhanced, self.canny_threshold1, self.canny_threshold2, apertureSize=3, L2gradient=True)
        
        # Step 6: Apply the mask - ONLY keep edges within road/lane region
        edges_masked = cv2.bitwise_and(edges, edges, mask=mask_binary)
        
        # Step 7: Clean up edges with morphological operations
        # Use a small kernel to connect nearby edge fragments
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_masked = cv2.morphologyEx(edges_masked, cv2.MORPH_CLOSE, kernel_close)
        
        # Remove tiny noise specs
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges_masked = cv2.morphologyEx(edges_masked, cv2.MORPH_OPEN, kernel_open)
        
        return edges_masked, mask_binary
    
    def create_edge_overlay(self, frame, edges, color=(0, 255, 255)):
        """
        Create an overlay of edges on the frame.
        
        Args:
            frame: BGR image
            edges: Binary edge image (single channel)
            color: BGR color for edges (default: cyan)
        
        Returns:
            frame with colored edges overlaid
        """
        result = frame.copy()
        
        # Create colored edge layer
        edge_layer = np.zeros_like(frame)
        edge_layer[edges > 0] = color
        
        # Blend edges onto frame with addWeighted for visibility
        result = cv2.addWeighted(result, 1.0, edge_layer, 1.0, 0)
        
        return result
    
    def create_canny_visualization(self, original, edges_masked, mask, edges_colored):
        """
        Create a 3-panel visualization:
        - Panel A: Original + overlay (mask + edges on top)
        - Panel B: Masked Canny edges (black background, white/cyan edges)
        - Panel C: Mask image (binary green)
        
        Returns a horizontally stacked image.
        """
        h, w = original.shape[:2]
        panel_width = w // 3
        panel_height = h
        
        # Panel A: Original with overlay already applied
        panel_a = original.copy()
        
        # Panel B: Edges on black background (convert to BGR for stacking)
        panel_b = np.zeros((h, w, 3), dtype=np.uint8)
        # Make edges bright cyan for visibility
        panel_b[edges_masked > 0] = (255, 255, 0)  # Cyan in BGR
        
        # Panel C: Binary mask visualization
        panel_c = np.zeros((h, w, 3), dtype=np.uint8)
        # Ensure mask is same size
        if mask.shape[:2] != (h, w):
            mask_display = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_display = mask
        # Color the mask green
        panel_c[mask_display > 0] = (0, 200, 0)
        
        # Add labels to panels with background for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Panel A label
        cv2.rectangle(panel_a, (5, 5), (150, 35), (0, 0, 0), -1)
        cv2.putText(panel_a, "OVERLAY", (10, 28), font, font_scale, (0, 255, 255), thickness)
        
        # Panel B label  
        cv2.rectangle(panel_b, (5, 5), (180, 35), (50, 50, 50), -1)
        cv2.putText(panel_b, "CANNY EDGES", (10, 28), font, font_scale, (255, 255, 255), thickness)
        
        # Panel C label
        cv2.rectangle(panel_c, (5, 5), (160, 35), (0, 0, 0), -1)
        cv2.putText(panel_c, "ROAD MASK", (10, 28), font, font_scale, (0, 255, 0), thickness)
        
        # Resize panels to equal width for clean stacking
        panel_a_resized = cv2.resize(panel_a, (panel_width, panel_height), interpolation=cv2.INTER_LINEAR)
        panel_b_resized = cv2.resize(panel_b, (panel_width, panel_height), interpolation=cv2.INTER_LINEAR)
        panel_c_resized = cv2.resize(panel_c, (panel_width, panel_height), interpolation=cv2.INTER_LINEAR)
        
        # Stack horizontally
        visualization = np.hstack([panel_a_resized, panel_b_resized, panel_c_resized])
        
        return visualization
    
    def update_counters(self):
        """Update the counter displays"""
        try:
            self.root.after(0, lambda: self.vehicle_count_label.config(text=str(self.vehicle_count)))
            self.root.after(0, lambda: self.person_count_label.config(text=str(self.person_count)))
            self.root.after(0, lambda: self.bicycle_count_label.config(text=str(self.bicycle_count)))
            self.root.after(0, lambda: self.anomaly_count_label.config(text=str(self.anomaly_count)))
        except:
            pass
    
    def update_legend(self):
        """Update legend colors based on detected classes"""
        try:
            color_map = {
                'road': (0, 200, 0),
                'lane': (0, 165, 255),
                'vehicle': (255, 150, 0),
                'person': (0, 165, 255),
                'bicycle': (255, 0, 255),
                'traffic': (0, 255, 255),
                'anomaly': (255, 255, 0),
                'canny': (0, 255, 255),
            }
            
            for key, label in self.legend_labels.items():
                if self.detected_classes.get(key, False):
                    color = color_map[key]
                    # Convert BGR to hex for Tkinter
                    hex_color = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
                    self.root.after(0, lambda l=label, c=hex_color: l.config(fg=c))
                else:
                    self.root.after(0, lambda l=label: l.config(fg='#666666'))
        except:
            pass
    
    def process_frame(self, frame):
        """Process a single frame through both models"""
        if not self.models_loaded:
            return frame
        
        t1 = time.time()
        
        # Resize frame to standard size
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        result_frame = frame.copy()
        
        # Reset counters and detected classes
        self.vehicle_count = 0
        self.person_count = 0
        self.bicycle_count = 0
        self.traffic_light_count = 0
        self.anomaly_count = 0
        for key in self.detected_classes:
            self.detected_classes[key] = False
        
        # ========== YOLOPv2: Road and Lane Segmentation ==========
        try:
            img_yolopv2, ratio, pad = letterbox(frame, self.IMG_SIZE)
            img_yolopv2 = img_yolopv2[:, :, ::-1].transpose(2, 0, 1)
            img_yolopv2 = np.ascontiguousarray(img_yolopv2)
            
            img_tensor = torch.from_numpy(img_yolopv2).to(self.device)
            img_tensor = img_tensor.half() if self.half else img_tensor.float()
            img_tensor /= 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            with torch.no_grad():
                [pred, anchor_grid], seg, ll = self.yolopv2_model(img_tensor)
            
            # Get segmentation masks with better post-processing
            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)
            
            # ===== OPTIMIZED ROAD SEGMENTATION FOR HIGH FPS =====
            # Faster processing when GPU available: use lighter morphology
            is_gpu = self.device.type == 'cuda'
            
            # Step 1: Denoise while preserving edges (lighter on GPU)
            da_seg_mask_uint8 = da_seg_mask.astype(np.uint8) * 255
            if is_gpu:
                # Skip bilateral filter on GPU (already fast enough from model)
                da_seg_mask = (da_seg_mask_uint8 > 127).astype(np.uint8)
            else:
                # Keep bilateral on CPU for quality
                da_seg_mask_uint8 = cv2.bilateralFilter(da_seg_mask_uint8, 9, 75, 75)
                da_seg_mask = (da_seg_mask_uint8 > 127).astype(np.uint8)
            
            # Step 2: Morphological operations - optimized for speed
            if is_gpu:
                # GPU speed: minimal morphology (model is already accurate)
                kernel_road = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                da_seg_mask = cv2.morphologyEx(da_seg_mask, cv2.MORPH_CLOSE, kernel_road, iterations=1)
            else:
                # CPU quality: more aggressive morphology
                kernel_road_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                kernel_road_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                da_seg_mask = cv2.morphologyEx(da_seg_mask, cv2.MORPH_CLOSE, kernel_road_close, iterations=3)
                da_seg_mask = cv2.morphologyEx(da_seg_mask, cv2.MORPH_OPEN, kernel_road_open, iterations=1)
            
            # Step 3: Skip contour filling on GPU (too slow)
            if not is_gpu:
                # Contour-based road completion - CPU only
                contours, _ = cv2.findContours(da_seg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                da_seg_mask_filled = da_seg_mask.copy()
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:
                        cv2.drawContours(da_seg_mask_filled, [contour], 0, 1, -1)
                da_seg_mask = da_seg_mask_filled
            
            # Step 4: Spatial constraint - lighter on GPU
            h, w = da_seg_mask.shape
            roi_threshold = int(h * 0.4)
            if not is_gpu:
                # Aggressive spatial filtering on CPU
                da_seg_mask[:roi_threshold] = cv2.morphologyEx(
                    da_seg_mask[:roi_threshold], cv2.MORPH_OPEN, 
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                )
            # GPU: skip spatial filtering - too slow for high FPS
            
            # Lane line processing (minimal, even on GPU)
            ll_seg_mask_uint8 = ll_seg_mask.astype(np.uint8) * 255
            if is_gpu:
                # Skip filtering on GPU - model output is clean enough
                ll_seg_mask = (ll_seg_mask_uint8 > 127).astype(np.uint8)
                kernel_lane = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                ll_seg_mask = cv2.morphologyEx(ll_seg_mask, cv2.MORPH_CLOSE, kernel_lane, iterations=1)
            else:
                # Keep filtering on CPU for quality
                ll_seg_mask_uint8 = cv2.bilateralFilter(ll_seg_mask_uint8, 7, 50, 50)
                ll_seg_mask = (ll_seg_mask_uint8 > 127).astype(np.uint8)
                kernel_lane = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                ll_seg_mask = cv2.morphologyEx(ll_seg_mask, cv2.MORPH_CLOSE, kernel_lane, iterations=2)
            
            # ===== LAYERED COLOR RENDERING (Fast) =====
            # Layer 1: Apply road segmentation (green) first - 50% opacity
            road_mask = (da_seg_mask == 1)
            result_frame[road_mask] = result_frame[road_mask] * 0.5 + np.array(self.COLOR_ROAD, dtype=np.float32) * 0.5
            
            # Layer 2: Apply lane lines (red) on top - 70% opacity, overwrites any previous pixel
            lane_mask = (ll_seg_mask == 1)
            result_frame[lane_mask] = result_frame[lane_mask] * 0.3 + np.array(self.COLOR_LANE, dtype=np.float32) * 0.7
            
            # Clamp values to valid range
            result_frame = np.clip(result_frame, 0, 255).astype(np.uint8)
            
            self.detected_classes['road'] = (da_seg_mask == 1).sum() > 100
            self.detected_classes['lane'] = (ll_seg_mask == 1).sum() > 100
            
            # ===== CANNY EDGE DETECTION ON ROAD/LANE REGION =====
            # This applies Canny edge detection ONLY within the road/lane mask
            # No edges from buildings, trees, cars, etc.
            if self.canny_enabled.get():
                try:
                    # Choose mask based on user preference
                    if self.canny_on_lanes.get():
                        # Apply Canny on lane lines only (thinner, more specific)
                        canny_source_mask = ll_seg_mask.copy()
                    else:
                        # Apply Canny on drivable road area (wider region)
                        canny_source_mask = da_seg_mask.copy()
                    
                    # Apply the masked Canny edge detection
                    edges_masked, mask_used = self.apply_masked_canny(frame, canny_source_mask)
                    
                    # Store for potential later use
                    self.last_road_mask = mask_used
                    self.last_canny_edges = edges_masked
                    
                    # Count edge pixels for stats
                    edge_pixel_count = np.sum(edges_masked > 0)
                    
                    # Check if we should show 3-panel edges-only view
                    if self.show_canny_only.get():
                        # Create 3-panel visualization: Overlay | Edges | Mask
                        result_frame = self.create_canny_visualization(
                            result_frame, edges_masked, mask_used, None
                        )
                    else:
                        # Overlay edges on the result frame in bright cyan
                        result_frame = self.create_edge_overlay(
                            result_frame, edges_masked, color=(0, 255, 255)  # Cyan
                        )
                    
                    # Mark as detected if we have significant edges
                    if edge_pixel_count > 100:
                        self.detected_classes['canny'] = True
                    
                except Exception as canny_err:
                    print(f"Canny edge detection error: {canny_err}")
                    import traceback
                    traceback.print_exc()
            
        except Exception as e:
            print(f"YOLOPv2 error: {e}")
        
        # ========== YOLOv8: Instance Segmentation + Detection ==========
        if YOLOV8_AVAILABLE and self.yolov8_segmentation_model is not None:
            try:
                conf_thres = self.conf_slider.get()
                results = self.yolov8_segmentation_model(frame, conf=conf_thres, verbose=False, imgsz=640)
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                        
                        # Check for segmentation masks
                        has_masks = result.masks is not None
                        if has_masks:
                            masks = result.masks.data.cpu().numpy()
                        
                        for i, (cls, conf) in enumerate(zip(classes, confs)):
                            cls_id = int(cls)
                            
                            # COCO classes mapping:
                            # 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 9=traffic light
                            
                            color = None
                            count_type = None
                            
                            if cls_id == 0:  # Person
                                color = self.COLOR_PERSON
                                count_type = 'person'
                                self.person_count += 1
                                self.detected_classes['person'] = True
                            elif cls_id == 1:  # Bicycle
                                color = self.COLOR_BICYCLE
                                count_type = 'bicycle'
                                self.bicycle_count += 1
                                self.detected_classes['bicycle'] = True
                            elif cls_id in [2, 3, 5, 7]:  # Vehicles (car, motorcycle, bus, truck)
                                color = self.COLOR_VEHICLE
                                count_type = 'vehicle'
                                self.vehicle_count += 1
                                self.detected_classes['vehicle'] = True
                            elif cls_id == 9:  # Traffic light
                                color = self.COLOR_TRAFFIC_LIGHT
                                count_type = 'traffic'
                                self.traffic_light_count += 1
                                self.detected_classes['traffic'] = True
                            else:
                                continue
                            
                            if color is None:
                                continue
                            
                            # Draw mask if available
                            if has_masks and i < len(masks):
                                try:
                                    mask = masks[i]
                                    mask_resized = cv2.resize(mask, (result_frame.shape[1], result_frame.shape[0]))
                                    mask_bool = mask_resized > 0.5
                                    
                                    # Apply colored overlay
                                    overlay = result_frame.copy()
                                    overlay[mask_bool] = color
                                    result_frame = cv2.addWeighted(result_frame, 0.7, overlay, 0.3, 0)
                                    
                                    # Draw outline
                                    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), 
                                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(result_frame, contours, -1, color, 2)
                                except Exception as e:
                                    print(f"Mask drawing error: {e}")
                            else:
                                # Draw bounding box if no mask
                                try:
                                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy[i]
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(result_frame, f"{conf:.2f}", (x1, y1 - 5),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                except Exception as e:
                                    print(f"Box drawing error: {e}")
                
            except Exception as e:
                print(f"YOLOv8 error: {e}")
                import traceback
                traceback.print_exc()
        
        # ========== Anomaly Detection (Potholes) - Only detects REAL road defects ==========
        try:
            # Only detect anomalies within the drivable road area
            if 'da_seg_mask' in locals() and da_seg_mask is not None:
                # Convert to grayscale and apply strict filtering
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Only analyze the road area (exclude everything else)
                road_only = np.zeros_like(gray)
                road_only[da_seg_mask == 1] = gray[da_seg_mask == 1]
                
                # Apply Gaussian blur
                blur = cv2.GaussianBlur(road_only, (5, 5), 0)
                
                # Use VERY strict thresholding - only very dark spots (actual holes)
                _, dark_regions = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY_INV)
                
                # Remove noise with morphology
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_OPEN, kernel)
                dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                anomaly_mask = np.zeros_like(gray, dtype=np.uint8)
                self.anomaly_count = 0
                
                for c in contours:
                    area = cv2.contourArea(c)
                    # Strict area filter: 1000-5000 pixels (actual pothole size)
                    if 1000 <= area <= 5000:
                        x, y, w, h = cv2.boundingRect(c)
                        aspect_ratio = float(w) / h if h > 0 else 0
                        
                        # Must be roughly circular (potholes are round/elliptical)
                        if 0.5 < aspect_ratio < 2.0:
                            # Calculate circularity (4π*area/perimeter²)
                            perimeter = cv2.arcLength(c, True)
                            if perimeter > 0:
                                circularity = 4 * np.pi * area / (perimeter * perimeter)
                                # Only very circular shapes (actual potholes)
                                if circularity > 0.4:
                                    cv2.drawContours(anomaly_mask, [c], -1, 255, -1)
                                    self.anomaly_count += 1
                
                # Apply cyan overlay only if real anomalies detected
                if self.anomaly_count > 0:
                    overlay = np.zeros_like(result_frame, dtype=np.uint8)
                    overlay[anomaly_mask > 0] = self.COLOR_ANOMALY
                    result_frame = cv2.addWeighted(result_frame, 1.0, overlay, 0.5, 0)
                    self.detected_classes['anomaly'] = True
            
        except Exception as e:
            print(f"Anomaly detection error: {e}")
        
        # Update counters
        self.update_counters()
        self.update_legend()
        
        # Calculate FPS and timing
        t3 = time.time()
        self.inference_time = (t3 - t1) * 1000
        self.fps = 1.0 / (t3 - t1) if (t3 - t1) > 0 else 0
        
        # Draw info on frame (top-left corner, semi-transparent background)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        
        info_texts = [
            f'FPS: {self.fps:.1f}',
            f'Inference: {self.inference_time:.1f}ms',
            f'Vehicles: {self.vehicle_count}',
            f'People: {self.person_count}',
            f'Bicycles: {self.bicycle_count}',
            f'Anomalies: {self.anomaly_count}',
        ]
        
        y_offset = 30
        for text in info_texts:
            cv2.putText(result_frame, text, (10, y_offset),
                       font, 0.7, (0, 255, 0), thickness)
            y_offset += 25
        
        self.last_processed_frame = result_frame.copy()
        return result_frame
    
    def save_screenshot(self, event=None):
        if self.last_processed_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, self.last_processed_frame)
            self.update_status(f"Screenshot saved: {filename}")
    
    def start_laptop_camera(self):
        if not self.models_loaded:
            messagebox.showwarning("Wait", "Models still loading...")
            return
        self.stop_all()
        self.current_mode.set("laptop")
        
        try:
            # Try multiple camera indices
            self.cap = None
            for idx in [0, 1, 2]:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Camera {idx} opened successfully")
                        self.cap = cap
                        break
                    cap.release()
            
            if self.cap is None:
                messagebox.showerror("Error", "Could not open any camera")
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_running = True
            self.update_status("Laptop Camera Active")
            self.canvas.delete('welcome')
            self.photo_info_label.config(text="Live: Laptop Camera | Press S to save screenshot")
            
            # Start video feed thread
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
            # Start display thread
            self.display_thread = threading.Thread(target=self.update_display_loop, daemon=True)
            self.display_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()
    
    def start_phone_camera(self):
        if not self.models_loaded:
            messagebox.showwarning("Wait", "Models still loading...")
            return
        self.stop_all()
        self.current_mode.set("phone")
        
        try:
            url = f"http://{self.DROIDCAM_IP}:{self.DROIDCAM_PORT}/video"
            self.update_status(f"Connecting to {self.DROIDCAM_IP}:{self.DROIDCAM_PORT}...")
            
            self.cap = cv2.VideoCapture(url)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not connect to DroidCam at {url}")
                return
            
            # Test read
            ret, frame = self.cap.read()
            if not ret or frame is None:
                messagebox.showerror("Error", "Connected but cannot read frames from DroidCam")
                self.cap.release()
                self.cap = None
                return
            
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_running = True
            self.update_status(f"DroidCam Active: {self.DROIDCAM_IP}:{self.DROIDCAM_PORT}")
            self.canvas.delete('welcome')
            self.photo_info_label.config(text="Live: DroidCam | Press S to save screenshot")
            
            # Start video feed thread
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
            # Start display thread
            self.display_thread = threading.Thread(target=self.update_display_loop, daemon=True)
            self.display_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()
    
    def video_loop(self):
        """Video capture loop running in separate thread - optimized for GPU"""
        frame_count = 0
        fps_start_time = time.time()
        
        # Adaptive FPS targeting
        target_fps = 60
        frame_time = 1.0 / target_fps
        
        while self.is_running and self.cap is not None:
            loop_start = time.time()
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Process frame
                    try:
                        processed = self.process_frame(frame)
                        
                        # Put in queue for display thread
                        try:
                            self.frame_queue.put_nowait(processed)
                        except queue.Full:
                            pass  # Drop frame if display can't keep up
                            
                    except Exception as proc_err:
                        print(f"Processing error: {proc_err}")
                    
                    # Adaptive sleep
                    elapsed = time.time() - loop_start
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed_fps = time.time() - fps_start_time
                        actual_fps = 30 / elapsed_fps
                        fps_start_time = time.time()
                else:
                    time.sleep(0.001)
                    
            except Exception as e:
                print(f"Video loop error: {e}")
                time.sleep(0.05)
    
    def update_display_loop(self):
        """Display frames from queue at steady rate"""
        while self.is_running:
            try:
                # Get latest frame from queue
                frame = None
                try:
                    while True:
                        frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                
                if frame is not None:
                    self.display_frame_on_canvas(frame)
                
                time.sleep(0.01)
                    
            except Exception as e:
                print(f"Display loop error: {e}")
                time.sleep(0.01)
    
    def start_photo_mode(self):
        if not self.models_loaded:
            messagebox.showwarning("Wait", "Models still loading...")
            return
        self.stop_all()
        self.current_mode.set("photos")
        self.load_images()
        
        if not self.image_list:
            messagebox.showinfo("No Images", f"Add images to '{os.path.abspath(self.IMAGES_FOLDER)}'")
            return
        
        self.canvas.delete('welcome')
        self.update_status(f"Photo Gallery - {len(self.image_list)} images")
        self.current_image_index = 0
        self.display_current_image()
    
    def start_video_mode(self):
        if not self.models_loaded:
            messagebox.showwarning("Wait", "Models still loading...")
            return
        self.stop_all()
        self.current_mode.set("video")
        self.load_videos()
        
        if not self.image_list:
            messagebox.showinfo("No Videos", f"Add video files to '{os.path.abspath(self.VIDEOS_FOLDER)}'\n\nSupported: .mp4, .avi, .mov, .mkv, .flv, .wmv")
            return
        
        self.canvas.delete('welcome')
        self.update_status(f"Video Mode - {len(self.image_list)} videos")
        self.current_image_index = 0
        self.play_video_file()
    
    def start_canny_mode(self):
        """
        Dedicated Canny Edge Detection mode.
        Shows ONLY Canny edges on black background - edges detected only within road region.
        Like Photo Gallery but outputs clean edge visualization.
        """
        if not self.models_loaded:
            messagebox.showwarning("Wait", "Models still loading...")
            return
        self.stop_all()
        self.current_mode.set("canny")
        self.load_images()
        
        if not self.image_list:
            messagebox.showinfo("No Images", f"Add images to '{os.path.abspath(self.IMAGES_FOLDER)}'")
            return
        
        self.canvas.delete('welcome')
        self.update_status(f"Canny Edge Mode - {len(self.image_list)} images")
        self.current_image_index = 0
        self.display_canny_image()
    
    def display_canny_image(self):
        """Process and display current image with ONLY Canny edges on black background."""
        if not self.image_list:
            return
        
        image_path = self.image_list[self.current_image_index]
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                self.update_status(f"Could not read: {image_path}")
                return
            
            t1 = time.time()
            
            # Resize frame to standard size
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
            
            # Process through YOLOPv2 to get road mask
            canny_result = self.process_canny_only(frame)
            
            t2 = time.time()
            self.inference_time = (t2 - t1) * 1000
            self.fps = 1.0 / (t2 - t1) if (t2 - t1) > 0 else 0
            
            self.last_processed_frame = canny_result
            self.display_frame_on_canvas(canny_result)
            
            filename = os.path.basename(image_path)
            self.photo_info_label.config(text=f"CANNY: {filename} | {self.current_image_index + 1}/{len(self.image_list)} | Arrow keys to navigate")
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            self.inference_label.config(text=f"Inference: {self.inference_time:.1f} ms")
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def process_canny_only(self, frame):
        """
        Process frame and return filled lane lines + obstacles on top, original image below.
        Uses YOLOPv2 lane mask + YOLOv8 for people/obstacles + color detection for cones.
        
        Returns: BGR image with filled lanes & obstacles (top) stacked over original (bottom)
        """
        h, w = frame.shape[:2]
        
        # Create black output image for lane visualization
        canny_output = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Separate masks for different object types (for coloring)
        lane_mask_final = np.zeros((h, w), dtype=np.uint8)
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)
        cone_mask = np.zeros((h, w), dtype=np.uint8)
        person_mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # ===== PART 1: YOLOPv2 for Road and Lane Detection =====
            img_yolopv2, ratio, pad = letterbox(frame, self.IMG_SIZE)
            img_yolopv2 = img_yolopv2[:, :, ::-1].transpose(2, 0, 1)
            img_yolopv2 = np.ascontiguousarray(img_yolopv2)
            
            img_tensor = torch.from_numpy(img_yolopv2).to(self.device)
            img_tensor = img_tensor.half() if self.half else img_tensor.float()
            img_tensor /= 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            with torch.no_grad():
                [pred, anchor_grid], seg, ll = self.yolopv2_model(img_tensor)
            
            # Get road and lane masks
            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)
            
            # Resize masks to frame size
            if ll_seg_mask.shape[:2] != (h, w):
                ll_resized = cv2.resize(ll_seg_mask.astype(np.uint8), (w, h), 
                                       interpolation=cv2.INTER_NEAREST)
            else:
                ll_resized = ll_seg_mask.astype(np.uint8)
            
            if da_seg_mask.shape[:2] != (h, w):
                da_resized = cv2.resize(da_seg_mask.astype(np.uint8), (w, h), 
                                       interpolation=cv2.INTER_NEAREST)
            else:
                da_resized = da_seg_mask.astype(np.uint8)
            
            # Create filled lane lines
            lane_mask = (ll_resized > 0).astype(np.uint8) * 255
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            lane_thick = cv2.dilate(lane_mask, kernel_dilate, iterations=2)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            lane_filled = cv2.morphologyEx(lane_thick, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            
            # Fill contours for solid lanes
            contours, _ = cv2.findContours(lane_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(lane_mask_final, [contour], -1, 255, -1)
            
            road_mask = (da_resized > 0).astype(np.uint8) * 255
            
            # ===== PART 2: YOLOv8 for People and Obstacles Detection =====
            if YOLOV8_AVAILABLE and self.yolov8_segmentation_model is not None:
                try:
                    results = self.yolov8_segmentation_model(frame, conf=0.3, verbose=False, imgsz=640)
                    
                    for result in results:
                        if result.boxes is not None and result.masks is not None:
                            boxes = result.boxes
                            masks = result.masks.data.cpu().numpy()
                            classes = boxes.cls.cpu().numpy()
                            
                            for i, cls_id in enumerate(classes):
                                cls_id = int(cls_id)
                                
                                if i < len(masks):
                                    mask = masks[i]
                                    mask_resized = cv2.resize(mask, (w, h))
                                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                                    
                                    # Person (class 0)
                                    if cls_id == 0:
                                        person_mask = cv2.bitwise_or(person_mask, mask_binary)
                                    
                                    # Bicycle (class 1), Motorcycle (class 3) - road obstacles
                                    elif cls_id in [1, 3]:
                                        obstacle_mask = cv2.bitwise_or(obstacle_mask, mask_binary)
                                    
                                    # Vehicles on road - car(2), bus(5), truck(7) 
                                    elif cls_id in [2, 5, 7]:
                                        obstacle_mask = cv2.bitwise_or(obstacle_mask, mask_binary)
                                    
                                    # Stop sign (11), parking meter (12), bench (13)
                                    elif cls_id in [11, 12, 13]:
                                        obstacle_mask = cv2.bitwise_or(obstacle_mask, mask_binary)
                                        
                except Exception as yolo_err:
                    print(f"YOLOv8 detection error: {yolo_err}")
            
            # ===== PART 3: Color-based Cone Detection (Orange/Yellow cones) =====
            try:
                # Convert to HSV for color detection
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Orange cone detection (typical traffic cone color)
                # Orange: H=10-25, S=100-255, V=100-255
                lower_orange = np.array([5, 150, 150])
                upper_orange = np.array([25, 255, 255])
                orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
                
                # Also detect bright yellow/lime cones
                lower_yellow = np.array([20, 150, 150])
                upper_yellow = np.array([35, 255, 255])
                yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                
                # Combine orange and yellow
                cone_color_mask = cv2.bitwise_or(orange_mask, yellow_mask)
                
                # Only keep cones that are within or near the road area
                # Dilate road mask to include roadside
                road_dilated = cv2.dilate(road_mask, np.ones((50, 50), np.uint8), iterations=1)
                cone_color_mask = cv2.bitwise_and(cone_color_mask, cone_color_mask, mask=road_dilated)
                
                # Morphological cleanup for cones
                kernel_cone = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                cone_color_mask = cv2.morphologyEx(cone_color_mask, cv2.MORPH_CLOSE, kernel_cone, iterations=2)
                cone_color_mask = cv2.morphologyEx(cone_color_mask, cv2.MORPH_OPEN, kernel_cone, iterations=1)
                
                # Find cone contours and filter by size/shape
                cone_contours, _ = cv2.findContours(cone_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in cone_contours:
                    area = cv2.contourArea(contour)
                    # Filter by reasonable cone size (not too small, not too big)
                    if 200 < area < 15000:
                        x, y, bw, bh = cv2.boundingRect(contour)
                        aspect_ratio = bh / bw if bw > 0 else 0
                        # Cones are typically taller than wide
                        if 0.5 < aspect_ratio < 4.0:
                            cv2.drawContours(cone_mask, [contour], -1, 255, -1)
                            
            except Exception as cone_err:
                print(f"Cone detection error: {cone_err}")
            
            # ===== PART 4: Canny Edges for Road Details =====
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ksize = self.canny_blur_ksize
            if ksize > 1:
                blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
            else:
                blurred = gray
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)
            edges = cv2.Canny(enhanced, self.canny_threshold1, self.canny_threshold2, 
                             apertureSize=3, L2gradient=True)
            edges_road = cv2.bitwise_and(edges, edges, mask=road_mask)
            
            # ===== PART 5: Combine All Obstacles Into One Mask =====
            # Combine all obstacles (people, cones, vehicles) into a single obstacle mask
            all_obstacles = cv2.bitwise_or(cv2.bitwise_or(person_mask, cone_mask), obstacle_mask)
            
            # ===== PART 6: Create Output - Only Two States =====
            # White for clear drivable lanes and road
            canny_output[lane_mask_final > 0] = (255, 255, 255)
            
            # Highlight all obstacles in red
            canny_output[all_obstacles > 0] = (0, 0, 255)  # Red for any obstacle
            
            # Add label
            cv2.putText(canny_output, "DRIVABLE ROAD (White=Clear | Red=Obstacles)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            cv2.putText(canny_output, f"Error: {str(e)}", (50, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Prepare original image with label
        original_labeled = frame.copy()
        cv2.putText(original_labeled, "ORIGINAL", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Stack vertically: Detection output on top, Original below
        stacked = np.vstack([canny_output, original_labeled])
        
        return stacked
    
    def load_videos(self):
        self.image_list = []
        extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
        for ext in extensions:
            self.image_list.extend(glob.glob(os.path.join(self.VIDEOS_FOLDER, ext)))
            self.image_list.extend(glob.glob(os.path.join(self.VIDEOS_FOLDER, ext.upper())))
        self.image_list.sort()
    
    def play_video_file(self):
        if self.current_image_index >= len(self.image_list):
            return
        
        video_path = self.image_list[self.current_image_index]
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video: {video_path}")
            return
        
        self.is_running = True
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.photo_info_label.config(text=f"Playing: {os.path.basename(video_path)} | {total_frames} frames @ {fps:.1f}fps | Arrow keys: Next/Prev video")
        
        # Start video feed thread
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
        # Start display thread
        self.display_thread = threading.Thread(target=self.update_display_loop, daemon=True)
        self.display_thread.start()
    
    def load_images(self):
        self.image_list = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        for ext in extensions:
            self.image_list.extend(glob.glob(os.path.join(self.IMAGES_FOLDER, ext)))
            self.image_list.extend(glob.glob(os.path.join(self.IMAGES_FOLDER, ext.upper())))
        self.image_list.sort()
    
    def display_current_image(self):
        if not self.image_list:
            return
        image_path = self.image_list[self.current_image_index]
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                self.update_status(f"Could not read: {image_path}")
                return
            processed = self.process_frame(frame)
            self.display_frame_on_canvas(processed)
            filename = os.path.basename(image_path)
            self.photo_info_label.config(text=f"{filename} | {self.current_image_index + 1}/{len(self.image_list)} | Arrow keys to navigate")
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            self.inference_label.config(text=f"Inference: {self.inference_time:.1f} ms")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
    
    def previous_image(self, event=None):
        mode = self.current_mode.get()
        if mode not in ("photos", "canny") or not self.image_list:
            return
        self.current_image_index = (self.current_image_index - 1) % len(self.image_list)
        if mode == "canny":
            self.display_canny_image()
        else:
            self.display_current_image()
    
    def next_image(self, event=None):
        mode = self.current_mode.get()
        if mode not in ("photos", "canny") or not self.image_list:
            return
        self.current_image_index = (self.current_image_index + 1) % len(self.image_list)
        if mode == "canny":
            self.display_canny_image()
        else:
            self.display_current_image()
    
    def display_frame_on_canvas(self, frame):
        """Display a frame on the canvas - optimized for 30+ FPS"""
        try:
            if frame is None or frame.size == 0:
                return
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use cached canvas size to avoid expensive update_idletasks()
            if not hasattr(self, '_canvas_size_cache') or self._canvas_size_cache is None:
                self.canvas.update_idletasks()
                self._canvas_size_cache = (self.canvas.winfo_width(), self.canvas.winfo_height())
            
            canvas_width, canvas_height = self._canvas_size_cache
            
            if canvas_width > 1 and canvas_height > 1:
                h, w = frame_rgb.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Only resize if needed (check if scale is close to 1.0)
                if abs(scale - 1.0) > 0.01:
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Use root.after for thread-safe canvas update
            def update_canvas():
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo
            
            self.root.after(0, update_canvas)
            
            # Update FPS labels less frequently (every 5 frames), also thread-safe
            if not hasattr(self, '_display_count'):
                self._display_count = 0
            self._display_count += 1
            
            if self._display_count % 5 == 0:
                def update_labels():
                    self.fps_label.config(text=f"FPS: {self.fps:.1f}")
                    self.inference_label.config(text=f"Inference: {self.inference_time:.1f} ms")
                
                self.root.after(0, update_labels)
                self._display_count = 0
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def stop_all(self):
        self.is_running = False
        time.sleep(0.1)  # Give thread time to stop
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.current_mode.set("none")
        self.update_status("Stopped")
    
    def on_closing(self, event=None):
        self.stop_all()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CombinedPerceptionGUI(root)
    root.mainloop()
