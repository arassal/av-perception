"""
YOLOPv2 + YOLOv8 Segmentation Editor - Professional Edition
Complete segmentation: Roads, Lanes, Vehicles, People, Bicycles, Traffic Lights
Based on working Prototype 1.1 implementation
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from pathlib import Path
import time
import sys

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import utilities
from utils.utils import (
    time_synchronized,
    select_device,
    non_max_suppression,
    split_for_trace_model,
    driving_area_mask,
    lane_line_mask,
    scale_coords,
    letterbox
)

# Check for YOLOv8
try:
    from ultralytics import YOLO
    YOLOV8_AVAILABLE = True
except ImportError:
    YOLOV8_AVAILABLE = False
    print("⚠️ YOLOv8 not available - install with: pip install ultralytics")


class SegmentationEditorPro:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 Professional Segmentation Editor")
        self.root.geometry("1800x950")
        self.root.configure(bg='#0a0a0a')
        
        # Configuration
        self.SCRIPT_DIR = SCRIPT_DIR
        self.YOLOPV2_WEIGHTS = os.path.join(SCRIPT_DIR, "data/weights/yolopv2.pt")
        self.YOLOV8_SEG_WEIGHTS = os.path.join(SCRIPT_DIR, "yolov8m-seg.pt")
        self.IMG_SIZE = 640
        self.OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_segmentation")
        
        # Colors (BGR format for OpenCV)
        self.COLOR_ROAD = (0, 200, 0)         # Green for drivable area
        self.COLOR_LANE = (0, 0, 255)         # Red for lane lines
        self.COLOR_VEHICLE = (255, 150, 0)    # Cyan/Blue for vehicles
        self.COLOR_PERSON = (0, 165, 255)     # Orange for people
        self.COLOR_BICYCLE = (255, 0, 255)    # Magenta for bicycles
        self.COLOR_TRAFFIC_LIGHT = (0, 255, 255)  # Yellow for traffic lights
        
        # Supported formats
        self.IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        self.VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm'}
        
        # State variables
        self.yolopv2_model = None
        self.yolov8_model = None
        self.device = None
        self.half = False
        self.models_loaded = False
        self.current_file = None
        self.current_frame = None
        self.segmented_frame = None
        self.is_processing = False
        self.video_frames = []
        
        # Counters
        self.vehicle_count = 0
        self.person_count = 0
        self.bicycle_count = 0
        self.traffic_light_count = 0
        
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        self.setup_ui()
        
        # Load models in background
        threading.Thread(target=self.load_models, daemon=True).start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Professional UI setup"""
        
        # Main container with gradient effect
        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header with modern styling
        header_frame = tk.Frame(main_frame, bg='#1a1a2e', height=80)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        header_frame.pack_propagate(False)
        
        title = tk.Label(
            header_frame,
            text="🎬 PROFESSIONAL SEGMENTATION EDITOR",
            font=('Helvetica', 28, 'bold'),
            fg='#00ff88',
            bg='#1a1a2e'
        )
        title.pack(pady=20)
        
        # Status bar with loading info
        status_bar = tk.Frame(main_frame, bg='#16213e', height=50)
        status_bar.pack(fill=tk.X, pady=(0, 10))
        status_bar.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_bar,
            text="⏳ Loading models... Please wait...",
            font=('Helvetica', 12),
            fg='#ffd700',
            bg='#16213e',
            anchor='w',
            padx=15
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.stats_label = tk.Label(
            status_bar,
            text="",
            font=('Helvetica', 11, 'bold'),
            fg='#00ff88',
            bg='#16213e',
            padx=15
        )
        self.stats_label.pack(side=tk.RIGHT)
        
        # Control panel
        control_panel = tk.Frame(main_frame, bg='#16213e')
        control_panel.pack(fill=tk.X, pady=(0, 10))
        
        # File selection
        file_frame = tk.Frame(control_panel, bg='#16213e')
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        btn_choose = tk.Button(
            file_frame,
            text="📁 CHOOSE FILE",
            font=('Helvetica', 13, 'bold'),
            bg='#0066ff',
            fg='white',
            padx=25,
            pady=12,
            cursor='hand2',
            relief=tk.FLAT,
            command=self.choose_file
        )
        btn_choose.pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(
            file_frame,
            text="No file selected",
            font=('Helvetica', 12),
            fg='#ffffff',
            bg='#16213e',
            anchor='w',
            padx=15
        )
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Settings panel
        settings_frame = tk.Frame(control_panel, bg='#16213e')
        settings_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Confidence slider
        conf_label = tk.Label(
            settings_frame,
            text="Confidence:",
            font=('Helvetica', 11),
            fg='#ffffff',
            bg='#16213e'
        )
        conf_label.pack(side=tk.LEFT, padx=(10, 5))
        
        self.conf_var = tk.DoubleVar(value=0.25)
        conf_slider = tk.Scale(
            settings_frame,
            from_=0.1,
            to=0.9,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            bg='#0066ff',
            fg='white',
            variable=self.conf_var,
            length=200,
            relief=tk.FLAT
        )
        conf_slider.pack(side=tk.LEFT, padx=5)
        
        # Legend
        legend_label = tk.Label(
            settings_frame,
            text="🟢 Road  🔴 Lanes  🔵 Vehicles  🟠 People  🟣 Bicycles  🟡 Traffic Lights",
            font=('Helvetica', 10, 'bold'),
            fg='#aaaaaa',
            bg='#16213e'
        )
        legend_label.pack(side=tk.RIGHT, padx=20)
        
        # Main content area - side by side previews
        content_frame = tk.Frame(main_frame, bg='#0a0a0a')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original preview
        left_panel = tk.Frame(content_frame, bg='#16213e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        original_header = tk.Frame(left_panel, bg='#16213e', height=40)
        original_header.pack(fill=tk.X)
        original_header.pack_propagate(False)
        
        tk.Label(
            original_header,
            text="📷 ORIGINAL",
            font=('Helvetica', 14, 'bold'),
            fg='#00d4ff',
            bg='#16213e'
        ).pack(pady=10)
        
        self.original_canvas = tk.Canvas(
            left_panel,
            bg='#000000',
            highlightthickness=3,
            highlightbackground='#00d4ff'
        )
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Segmented preview
        right_panel = tk.Frame(content_frame, bg='#16213e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        segmented_header = tk.Frame(right_panel, bg='#16213e', height=40)
        segmented_header.pack(fill=tk.X)
        segmented_header.pack_propagate(False)
        
        tk.Label(
            segmented_header,
            text="✨ SEGMENTED",
            font=('Helvetica', 14, 'bold'),
            fg='#00ff88',
            bg='#16213e'
        ).pack(pady=10)
        
        self.segmented_canvas = tk.Canvas(
            right_panel,
            bg='#000000',
            highlightthickness=3,
            highlightbackground='#00ff88'
        )
        self.segmented_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Action buttons
        action_frame = tk.Frame(main_frame, bg='#0a0a0a')
        action_frame.pack(fill=tk.X, pady=(15, 0))
        
        btn_process = tk.Button(
            action_frame,
            text="⚡ PROCESS",
            font=('Helvetica', 13, 'bold'),
            bg='#ff6600',
            fg='white',
            padx=30,
            pady=15,
            cursor='hand2',
            relief=tk.FLAT,
            command=self.process_file
        )
        btn_process.pack(side=tk.LEFT, padx=5)
        
        btn_download = tk.Button(
            action_frame,
            text="⬇️ DOWNLOAD",
            font=('Helvetica', 13, 'bold'),
            bg='#00aa00',
            fg='white',
            padx=30,
            pady=15,
            cursor='hand2',
            relief=tk.FLAT,
            command=self.download_file,
            state=tk.DISABLED
        )
        btn_download.pack(side=tk.LEFT, padx=5)
        self.btn_download = btn_download
        
        btn_clear = tk.Button(
            action_frame,
            text="🗑️ CLEAR",
            font=('Helvetica', 13, 'bold'),
            bg='#aa0000',
            fg='white',
            padx=30,
            pady=15,
            cursor='hand2',
            relief=tk.FLAT,
            command=self.clear_all
        )
        btn_clear.pack(side=tk.LEFT, padx=5)
        
        self.progress_label = tk.Label(
            action_frame,
            text="",
            font=('Helvetica', 12, 'bold'),
            fg='#00ff88',
            bg='#0a0a0a'
        )
        self.progress_label.pack(side=tk.RIGHT, padx=20)
    
    def load_models(self):
        """Load both YOLOPv2 and YOLOv8 models"""
        try:
            self.status_label.config(text="📦 Loading YOLOPv2 for road/lane detection...")
            self.root.update()
            
            # Check weights exist
            if not os.path.exists(self.YOLOPV2_WEIGHTS):
                raise FileNotFoundError(f"YOLOPv2 weights not found: {self.YOLOPV2_WEIGHTS}")
            
            # Load YOLOPv2
            self.device = select_device('0')
            self.half = self.device.type != 'cpu'
            self.yolopv2_model = torch.jit.load(self.YOLOPV2_WEIGHTS)
            self.yolopv2_model = self.yolopv2_model.to(self.device)
            if self.half:
                self.yolopv2_model.half()
            self.yolopv2_model.eval()
            
            # Warmup
            if self.device.type != 'cpu':
                self.yolopv2_model(torch.zeros(1, 3, self.IMG_SIZE, self.IMG_SIZE).to(self.device).type_as(next(self.yolopv2_model.parameters())))
            
            self.status_label.config(text="📦 Loading YOLOv8 for object segmentation...")
            self.root.update()
            
            # Load YOLOv8 if available
            if YOLOV8_AVAILABLE:
                if os.path.exists(self.YOLOV8_SEG_WEIGHTS):
                    self.yolov8_model = YOLO(self.YOLOV8_SEG_WEIGHTS)
                    self.status_label.config(text="✅ Both models loaded! Ready to segment.")
                else:
                    self.status_label.config(text="✅ YOLOPv2 loaded! (YOLOv8 weights not found)")
            else:
                self.status_label.config(text="✅ YOLOPv2 loaded! (Install ultralytics for full features)")
            
            self.models_loaded = True
            
        except Exception as e:
            self.status_label.config(text=f"❌ Error: {str(e)}")
            messagebox.showerror("Model Error", f"Failed to load models:\n{str(e)}")
    
    def choose_file(self):
        """Choose file to process"""
        if not self.models_loaded:
            messagebox.showwarning("Not Ready", "Models are still loading. Please wait.")
            return
        
        filetypes = [
            ("All Supported", ["*" + fmt for fmt in sorted(self.IMAGE_FORMATS | self.VIDEO_FORMATS)]),
            ("Images", ["*" + fmt for fmt in sorted(self.IMAGE_FORMATS)]),
            ("Videos", ["*" + fmt for fmt in sorted(self.VIDEO_FORMATS)])
        ]
        
        file_path = filedialog.askopenfilename(title="Select Image or Video", filetypes=filetypes)
        
        if file_path:
            self.current_file = file_path
            self.current_frame = None
            self.segmented_frame = None
            self.video_frames = []
            
            file_name = os.path.basename(file_path)
            file_ext = Path(file_path).suffix.lower()
            self.file_label.config(text=f"📄 {file_name}")
            
            # Load preview
            if file_ext in self.IMAGE_FORMATS:
                img = cv2.imread(file_path)
                if img is not None:
                    self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.display_frame(self.current_frame, self.original_canvas)
                    self.status_label.config(text="✅ Image loaded. Click PROCESS to segment.")
            elif file_ext in self.VIDEO_FORMATS:
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.display_frame(self.current_frame, self.original_canvas)
                    self.status_label.config(text="✅ Video loaded. Click PROCESS to segment.")
    
    def display_frame(self, frame_rgb, canvas):
        """Display frame on canvas"""
        try:
            h, w = frame_rgb.shape[:2]
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 600
            
            scale = min(canvas_width / w, canvas_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            img_pil = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk)
            canvas.image = img_tk
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def process_file(self):
        """Process the selected file"""
        if not self.current_file:
            messagebox.showwarning("No File", "Please choose a file first")
            return
        
        if not self.models_loaded:
            messagebox.showwarning("Not Ready", "Models are still loading")
            return
        
        file_ext = Path(self.current_file).suffix.lower()
        
        if file_ext in self.IMAGE_FORMATS:
            self.process_image()
        elif file_ext in self.VIDEO_FORMATS:
            self.process_video()
    
    def process_single_frame(self, frame):
        """Process a single frame with full segmentation"""
        # Reset counters
        self.vehicle_count = 0
        self.person_count = 0
        self.bicycle_count = 0
        self.traffic_light_count = 0
        
        result_frame = frame.copy()
        conf_thres = self.conf_var.get()
        
        # ===== YOLOPV2: Road and Lane Segmentation =====
        try:
            img, ratio, pad = letterbox(frame, self.IMG_SIZE)
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            
            img_tensor = torch.from_numpy(img).to(self.device)
            img_tensor = img_tensor.half() if self.half else img_tensor.float()
            img_tensor /= 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            with torch.no_grad():
                [pred, anchor_grid], seg, ll = self.yolopv2_model(img_tensor)
            
            # Get masks
            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)
            
            # Resize masks to frame size
            h, w = frame.shape[:2]
            da_seg_mask = cv2.resize(da_seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            ll_seg_mask = cv2.resize(ll_seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Apply morphological operations for cleaner masks
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            da_seg_mask = cv2.morphologyEx(da_seg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            ll_seg_mask = cv2.morphologyEx(ll_seg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Apply road segmentation (green) - 50% opacity
            road_mask = (da_seg_mask == 1)
            result_frame[road_mask] = (result_frame[road_mask] * 0.5 + np.array(self.COLOR_ROAD, dtype=np.float32) * 0.5).astype(np.uint8)
            
            # Apply lane lines (red) - 70% opacity
            lane_mask = (ll_seg_mask == 1)
            result_frame[lane_mask] = (result_frame[lane_mask] * 0.3 + np.array(self.COLOR_LANE, dtype=np.float32) * 0.7).astype(np.uint8)
            
        except Exception as e:
            print(f"YOLOPv2 error: {e}")
        
        # ===== YOLOV8: Instance Segmentation =====
        if YOLOV8_AVAILABLE and self.yolov8_model is not None:
            try:
                results = self.yolov8_model(frame, conf=conf_thres, verbose=False, imgsz=640)
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                        
                        has_masks = result.masks is not None
                        if has_masks:
                            masks = result.masks.data.cpu().numpy()
                        
                        for i, (cls, conf) in enumerate(zip(classes, confs)):
                            cls_id = int(cls)
                            
                            # Map COCO classes to colors
                            color = None
                            if cls_id == 0:  # Person
                                color = self.COLOR_PERSON
                                self.person_count += 1
                            elif cls_id == 1:  # Bicycle
                                color = self.COLOR_BICYCLE
                                self.bicycle_count += 1
                            elif cls_id in [2, 3, 5, 7]:  # Vehicles
                                color = self.COLOR_VEHICLE
                                self.vehicle_count += 1
                            elif cls_id == 9:  # Traffic light
                                color = self.COLOR_TRAFFIC_LIGHT
                                self.traffic_light_count += 1
                            
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
                                    print(f"Mask error: {e}")
                            else:
                                # Draw bounding box
                                try:
                                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy[i]
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(result_frame, f"{conf:.2f}", (x1, y1 - 5),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                except Exception as e:
                                    print(f"Box error: {e}")
                
            except Exception as e:
                print(f"YOLOv8 error: {e}")
        
        return result_frame
    
    def process_image(self):
        """Process single image"""
        try:
            self.is_processing = True
            self.status_label.config(text="⏳ Processing image...")
            self.root.update()
            
            img = cv2.imread(self.current_file)
            result_frame = self.process_single_frame(img)
            
            self.segmented_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            self.display_frame(self.segmented_frame, self.segmented_canvas)
            self.btn_download.config(state=tk.NORMAL)
            
            stats = f"🚗 {self.vehicle_count} | 👤 {self.person_count} | 🚲 {self.bicycle_count} | 🚦 {self.traffic_light_count}"
            self.stats_label.config(text=stats)
            self.status_label.config(text="✅ Segmentation complete!")
            self.is_processing = False
            
        except Exception as e:
            self.status_label.config(text=f"❌ Error: {str(e)}")
            self.is_processing = False
    
    def process_video(self):
        """Process video"""
        try:
            self.is_processing = True
            self.status_label.config(text="⏳ Processing video...")
            self.root.update()
            
            cap = cv2.VideoCapture(self.current_file)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.video_frames = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                progress = int((frame_idx / frame_count) * 100)
                self.progress_label.config(text=f"{progress}%")
                self.root.update()
                
                result_frame = self.process_single_frame(frame)
                self.video_frames.append(result_frame)
                frame_idx += 1
            
            cap.release()
            
            if self.video_frames:
                self.segmented_frame = cv2.cvtColor(self.video_frames[0], cv2.COLOR_BGR2RGB)
                self.display_frame(self.segmented_frame, self.segmented_canvas)
                self.btn_download.config(state=tk.NORMAL)
                self.status_label.config(text=f"✅ Video processed! ({len(self.video_frames)} frames)")
                self.progress_label.config(text="100%")
            
            self.is_processing = False
            
        except Exception as e:
            self.status_label.config(text=f"❌ Error: {str(e)}")
            self.is_processing = False
    
    def download_file(self):
        """Save segmented file"""
        if self.segmented_frame is None and not self.video_frames:
            messagebox.showwarning("No Result", "Process a file first")
            return
        
        save_path = filedialog.asksaveasfilename(
            defaultextension="",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("MP4", "*.mp4")]
        )
        
        if not save_path:
            return
        
        try:
            if self.video_frames:
                # Save video
                h, w = self.video_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(save_path, fourcc, 30, (w, h))
                for frame in self.video_frames:
                    out.write(frame)
                out.release()
            else:
                # Save image
                segmented_bgr = cv2.cvtColor(self.segmented_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, segmented_bgr)
            
            self.status_label.config(text=f"✅ Saved: {os.path.basename(save_path)}")
            messagebox.showinfo("Success", f"File saved!\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{str(e)}")
    
    def clear_all(self):
        """Clear everything"""
        self.current_file = None
        self.current_frame = None
        self.segmented_frame = None
        self.video_frames = []
        self.file_label.config(text="No file selected")
        self.status_label.config(text="✅ Ready for new file")
        self.stats_label.config(text="")
        self.progress_label.config(text="")
        self.original_canvas.delete("all")
        self.segmented_canvas.delete("all")
        self.btn_download.config(state=tk.DISABLED)
    
    def on_closing(self):
        """Handle close"""
        if self.is_processing:
            if messagebox.askyesno("Processing", "Still processing. Quit anyway?"):
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = SegmentationEditorPro(root)
    root.mainloop()


if __name__ == "__main__":
    main()
