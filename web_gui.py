"""
Traffic Cone Detection - Fixed stripe handling
Groups orange stripes that belong to the same cone
"""
import os
import cv2
import numpy as np
from flask import Flask, jsonify
import base64

app = Flask(__name__)

class AppState:
    def __init__(self):
        print("Loading images...")
        self.image_list = sorted([
            os.path.join('images', f) for f in os.listdir('images')
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'))
        ])
        self.current_image_index = 0

state = AppState()


def group_vertical_regions(boxes, max_horizontal_gap=50):
    """Group boxes that are vertically aligned (same cone)"""
    if not boxes:
        return []
    
    # Sort by x position
    boxes = sorted(boxes, key=lambda b: b[0])
    
    groups = []
    used = set()
    
    for i, box1 in enumerate(boxes):
        if i in used:
            continue
        
        group = [box1]
        used.add(i)
        x1, y1, w1, h1 = box1
        cx1 = x1 + w1 // 2
        
        for j, box2 in enumerate(boxes):
            if j in used:
                continue
            
            x2, y2, w2, h2 = box2
            cx2 = x2 + w2 // 2
            
            # Check if horizontally aligned (centers within threshold)
            if abs(cx1 - cx2) < max_horizontal_gap:
                group.append(box2)
                used.add(j)
        
        # Merge group into single bounding box
        if group:
            min_x = min(b[0] for b in group)
            min_y = min(b[1] for b in group)
            max_x = max(b[0] + b[2] for b in group)
            max_y = max(b[1] + b[3] for b in group)
            groups.append((min_x, min_y, max_x - min_x, max_y - min_y))
    
    return groups


def detect_cones(frame):
    """Detect traffic cones by finding orange regions and grouping vertical stripes"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect orange color
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([18, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Light cleanup only - don't merge stripes yet
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find all orange regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes for significant orange regions
    orange_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:  # Minimum area for a stripe segment
            x, y, bw, bh = cv2.boundingRect(cnt)
            orange_boxes.append((x, y, bw, bh))
    
    # Group vertically aligned regions (stripes of same cone)
    grouped = group_vertical_regions(orange_boxes, max_horizontal_gap=60)
    
    # Filter grouped regions by cone-like properties
    cones = []
    for x, y, bw, bh in grouped:
        area = bw * bh
        if area < 1500:  # Too small
            continue
        
        aspect = bh / (bw + 0.001)
        if aspect < 0.4:  # Too wide/flat
            continue
        if aspect > 6:  # Too thin
            continue
        
        cones.append({
            'x': x, 'y': y, 'w': bw, 'h': bh,
            'area': area
        })
    
    # Sort by x position
    cones.sort(key=lambda c: c['x'])
    
    # Remove duplicates/overlaps
    final = []
    for cone in cones:
        overlap = False
        for existing in final:
            # Check overlap
            ox1 = max(cone['x'], existing['x'])
            oy1 = max(cone['y'], existing['y'])
            ox2 = min(cone['x'] + cone['w'], existing['x'] + existing['w'])
            oy2 = min(cone['y'] + cone['h'], existing['y'] + existing['h'])
            
            if ox1 < ox2 and oy1 < oy2:
                ov_area = (ox2 - ox1) * (oy2 - oy1)
                if ov_area > 0.3 * min(cone['area'], existing['area']):
                    overlap = True
                    break
        
        if not overlap:
            final.append(cone)
    
    return final


def process_frame(frame):
    """Process frame and visualize detections"""
    output = frame.copy()
    h, w = frame.shape[:2]
    
    cones = detect_cones(frame)
    
    for i, cone in enumerate(cones):
        x, y, bw, bh = cone['x'], cone['y'], cone['w'], cone['h']
        
        # Padding
        pad = 8
        x = max(0, x - pad)
        y = max(0, y - pad)
        bw = min(w - x, bw + 2*pad)
        bh = min(h - y, bh + 2*pad)
        
        # Draw box
        cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        
        # Label
        label = f'CONE {i + 1}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(output, (x, y - 28), (x + tw + 10, y), (0, 180, 0), -1)
        cv2.putText(output, label, (x + 5, y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Header
    text = f'Traffic Cones Detected: {len(cones)}'
    cv2.rectangle(output, (10, 10), (500, 65), (0, 0, 0), -1)
    cv2.rectangle(output, (10, 10), (500, 65), (0, 255, 0), 3)
    cv2.putText(output, text, (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    return output


@app.route('/')
def index():
    return open('templates/index.html', 'r').read()


@app.route('/api/process-canny/<int:idx>')
def process_canny(idx):
    if idx < 0 or idx >= len(state.image_list):
        return jsonify({'error': 'Invalid index'}), 400
    
    img_path = state.image_list[idx]
    frame = cv2.imread(img_path)
    if frame is None:
        return jsonify({'error': 'Failed to load'}), 500
    
    frame = cv2.resize(frame, (1280, 720))
    output = process_frame(frame)
    
    _, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{img_base64}',
        'index': idx,
        'total': len(state.image_list),
        'filename': os.path.basename(img_path)
    })


@app.route('/api/next-image')
def next_image():
    state.current_image_index = (state.current_image_index + 1) % len(state.image_list)
    return process_canny(state.current_image_index)


@app.route('/api/prev-image')
def prev_image():
    state.current_image_index = (state.current_image_index - 1) % len(state.image_list)
    return process_canny(state.current_image_index)


@app.route('/api/update-canny-threshold', methods=['POST'])
def update_canny_threshold():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("Testing...")
    for img in ['image.png', 'image copy 3.png']:
        path = f'images/{img}'
        if os.path.exists(path):
            f = cv2.imread(path)
            if f is not None:
                f = cv2.resize(f, (1280, 720))
                c = detect_cones(f)
                print(f"  {img}: {len(c)} cones")
    
    print(f"\n{len(state.image_list)} images loaded")
    print("http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
