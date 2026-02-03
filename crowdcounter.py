import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import time
from collections import deque

class EnhancedCrowdCounter:
    def __init__(self, model_path=None):
        self.model = None
        self.frame_history = deque(maxlen=30)  # For smoothing detections
        self.count_history = deque(maxlen=100)  # Store historical counts
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.initialize_detectors()
            
        # Color codes for different density levels
        self.density_colors = {
            'low': (0, 255, 0),      # Green
            'medium': (0, 255, 255),  # Yellow
            'high': (0, 0, 255),      # Red
            'critical': (0, 0, 139)   # Dark Red
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_frames': 0,
            'processing_time': [],
            'average_fps': 0
        }
    
    def initialize_detectors(self):
        """Initialize multiple detection methods"""
        print("Initializing detectors...")
        
        # HOG + SVM for full body detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Upper body detection
        self.upper_body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_upperbody.xml'
        )
        
        # Try to load deep learning model if available
        try:
            self.net = cv2.dnn.readNetFromCaffe(
                'deploy.prototxt', 
                'mobilenet_iter_73000.caffemodel'
            )
            self.use_dnn = True
            print("Deep learning model loaded successfully")
        except:
            self.use_dnn = False
            print("Using traditional detection methods")
        
        print("Detectors initialized successfully")
    
    def create_density_map(self, image, points, sigma=15):
        """Create density map from detection points"""
        h, w = image.shape[:2]
        density_map = np.zeros((h, w), dtype=np.float32)
        
        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < w and 0 <= y < h:
                density_map[y, x] = 1
        
        # Apply Gaussian filter for smooth density
        density_map = gaussian_filter(density_map, sigma)
        
        # Normalize
        if density_map.sum() > 0:
            density_map = density_map / density_map.max()
        
        return density_map
    
    def multi_scale_detection(self, image):
        """Detect people at multiple scales for better accuracy"""
        scales = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        all_boxes = []
        all_weights = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for scale in scales:
            # Resize image
            h, w = image.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
            
            # Detect people using HOG
            boxes, weights = self.hog.detectMultiScale(
                resized,
                winStride=(8, 8),
                padding=(16, 16),
                scale=1.03
            )
            
            # Scale boxes back to original size
            if len(boxes) > 0:
                boxes = (boxes / scale).astype(int)
                all_boxes.extend(boxes)
                all_weights.extend(weights.flatten())
        
        # Apply weighted non-maximum suppression
        if len(all_boxes) > 0:
            boxes = self.weighted_nms(np.array(all_boxes), np.array(all_weights), overlapThresh=0.4)
        else:
            boxes = []
        
        return boxes
    
    def weighted_nms(self, boxes, weights, overlapThresh=0.3):
        """Weighted non-maximum suppression"""
        if len(boxes) == 0:
            return []
        
        boxes = boxes.astype("float")
        pick = []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by weights (confidence scores)
        idxs = np.argsort(weights)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
        
        return boxes[pick].astype("int")
    
    def detect_using_dnn(self, image):
        """Use deep neural network for detection (if available)"""
        if not self.use_dnn:
            return []
        
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, 
                                     (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype("int"))
        
        return boxes
    
    def count_crowd(self, image_path, use_multi_scale=True, save_output=True):
        """Main counting function for single image"""
        start_time = time.time()
        
        # Read and validate image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return 0
        
        # Create output directory
        if save_output:
            output_dir = "crowd_output"
            os.makedirs(output_dir, exist_ok=True)
        
        # Store original image for output
        output_img = img.copy()
        
        # Multiple detection methods
        all_detections = []
        
        # Method 1: Multi-scale HOG
        if use_multi_scale:
            hog_boxes = self.multi_scale_detection(img)
            all_detections.extend([('body', box) for box in hog_boxes])
        else:
            boxes, _ = self.hog.detectMultiScale(img, winStride=(8, 8), 
                                                 padding=(16, 16), scale=1.05)
            all_detections.extend([('body', box) for box in boxes])
        
        # Method 2: Face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        all_detections.extend([('face', box) for box in faces])
        
        # Method 3: Upper body detection
        upper_bodies = self.upper_body_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(50, 100)
        )
        all_detections.extend([('upper_body', box) for box in upper_bodies])
        
        # Method 4: DNN if available
        if self.use_dnn:
            dnn_boxes = self.detect_using_dnn(img)
            all_detections.extend([('dnn', box) for box in dnn_boxes])
        
        # Merge and filter detections
        filtered_detections = self.merge_detections(all_detections)
        
        # Calculate statistics
        body_count = len([d for d in all_detections if d[0] == 'body'])
        face_count = len([d for d in all_detections if d[0] == 'face'])
        upper_body_count = len([d for d in all_detections if d[0] == 'upper_body'])
        
        # Weighted total count
        total_count = int(body_count * 0.7 + face_count * 0.2 + upper_body_count * 0.1)
        
        # Determine density level
        density_level = self.get_density_level(total_count, img.shape[:2])
        
        # Create visualization
        self.visualize_results(output_img, filtered_detections, total_count, 
                              density_level, body_count, face_count)
        
        # Create density map
        points = [(x + w//2, y + h//2) for _, (x, y, w, h) in filtered_detections]
        density_map = self.create_density_map(img, points)
        
        # Create heatmap visualization
        heatmap = self.create_heatmap_visualization(density_map, img.shape)
        
        # Save outputs
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save annotated image
            output_path = os.path.join(output_dir, f"crowd_{timestamp}.jpg")
            cv2.imwrite(output_path, output_img)
            
            # Save heatmap
            heatmap_path = os.path.join(output_dir, f"heatmap_{timestamp}.jpg")
            cv2.imwrite(heatmap_path, heatmap)
            
            # Save analytics
            analytics = {
                'timestamp': timestamp,
                'image_path': image_path,
                'total_count': total_count,
                'body_detections': body_count,
                'face_detections': face_count,
                'upper_body_detections': upper_body_count,
                'density_level': density_level,
                'image_dimensions': img.shape[:2],
                'processing_time': time.time() - start_time
            }
            
            analytics_path = os.path.join(output_dir, f"analytics_{timestamp}.json")
            with open(analytics_path, 'w') as f:
                json.dump(analytics, f, indent=4)
            
            print(f"Results saved to {output_dir}/")
        
        # Display results
        self.display_results(output_img, heatmap, total_count, density_level, 
                            body_count, face_count, time.time() - start_time)
        
        return total_count, analytics if save_output else total_count
    
    def merge_detections(self, detections, overlap_thresh=0.5):
        """Merge overlapping detections from different methods"""
        if not detections:
            return []
        
        boxes = []
        for det_type, (x, y, w, h) in detections:
            boxes.append({
                'box': [x, y, x + w, y + h],
                'type': det_type,
                'area': w * h
            })
        
        # Sort by area (largest first)
        boxes.sort(key=lambda x: x['area'], reverse=True)
        
        merged = []
        while boxes:
            current = boxes.pop(0)
            merged.append(current)
            
            i = 0
            while i < len(boxes):
                box = boxes[i]
                iou = self.calculate_iou(current['box'], box['box'])
                
                if iou > overlap_thresh:
                    boxes.pop(i)
                else:
                    i += 1
        
        # Convert back to original format
        result = []
        for item in merged:
            x1, y1, x2, y2 = item['box']
            result.append((item['type'], (x1, y1, x2 - x1, y2 - y1)))
        
        return result
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection)
    
    def get_density_level(self, count, img_shape):
        """Determine crowd density level based on count and image area"""
        h, w = img_shape[:2]
        total_area = h * w
        density = count / (total_area / 10000)  # People per 10k pixels
        
        if density < 2:
            return 'low'
        elif density < 5:
            return 'medium'
        elif density < 10:
            return 'high'
        else:
            return 'critical'
    
    def visualize_results(self, image, detections, total_count, density_level, 
                         body_count, face_count):
        """Draw visualizations on the image"""
        h, w = image.shape[:2]
        
        # Draw detections with different colors
        for det_type, (x, y, w_box, h_box) in detections:
            if det_type == 'body':
                color = (0, 255, 0)  # Green
                label = "Person"
            elif det_type == 'face':
                color = (255, 0, 0)  # Blue
                label = "Face"
            elif det_type == 'upper_body':
                color = (255, 255, 0)  # Cyan
                label = "Upper Body"
            else:
                color = (255, 0, 255)  # Magenta
                label = "DNN"
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w_box, y + h_box), color, 2)
            
            # Draw label
            cv2.rectangle(image, (x, y - 20), (x + 80, y), color, -1)
            cv2.putText(image, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw statistics panel
        panel_height = 150
        cv2.rectangle(image, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.rectangle(image, (0, 0), (w, panel_height), (255, 255, 255), 2)
        
        # Get color for density level
        density_color = self.density_colors.get(density_level, (255, 255, 255))
        
        # Add text information
        texts = [
            f"CROWD DENSITY: {density_level.upper()}",
            f"TOTAL COUNT: {total_count}",
            f"Body Detections: {body_count}",
            f"Face Detections: {face_count}",
            f"Frame Size: {w}x{h}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        for i, text in enumerate(texts):
            y_pos = 30 + i * 25
            if i == 0:  # Density level
                cv2.putText(image, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, density_color, 2)
            elif i == 1:  # Total count
                cv2.putText(image, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
            else:
                cv2.putText(image, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw density indicator
        indicator_x = w - 150
        indicator_y = 20
        cv2.rectangle(image, (indicator_x, indicator_y), 
                     (indicator_x + 100, indicator_y + 20), density_color, -1)
        cv2.putText(image, density_level.upper(), (indicator_x + 10, indicator_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def create_heatmap_visualization(self, density_map, img_shape):
        """Create heatmap visualization from density map"""
        h, w = img_shape[:2]
        
        # Normalize density map for visualization
        if density_map.max() > 0:
            density_vis = (density_map / density_map.max() * 255).astype(np.uint8)
        else:
            density_vis = np.zeros((h, w), dtype=np.uint8)
        
        # Apply color map
        heatmap = cv2.applyColorMap(density_vis, cv2.COLORMAP_JET)
        
        # Add legend
        legend_height = 30
        legend = np.zeros((legend_height, w, 3), dtype=np.uint8)
        for i in range(w):
            color_val = int(i / w * 255)
            color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), 
                                     cv2.COLORMAP_JET)[0][0]
            legend[:, i] = color
        
        # Add legend labels
        cv2.putText(legend, "Low", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1)
        cv2.putText(legend, "High", (w - 40, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1)
        cv2.putText(legend, "Density Heatmap", (w//2 - 60, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Combine heatmap and legend
        combined = np.vstack([heatmap, legend])
        
        return combined
    
    def display_results(self, image, heatmap, total_count, density_level, 
                       body_count, face_count, processing_time):
        """Display results in a nice format"""
        # Create matplotlib figure for better visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display original image with detections
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Crowd Detection - Count: {total_count}', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Display heatmap
        axes[1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Density Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Add statistics text
        stats_text = f"""
        Detailed Statistics:
        --------------------
        Total People: {total_count}
        Density Level: {density_level.upper()}
        Body Detections: {body_count}
        Face Detections: {face_count}
        Processing Time: {processing_time:.3f} seconds
        """
        
        plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.show()
        
        # Print console output
        print("\n" + "="*50)
        print("CROWD COUNTING RESULTS")
        print("="*50)
        print(f"Total People Detected: {total_count}")
        print(f"Density Level: {density_level.upper()}")
        print(f"Body Detections: {body_count}")
        print(f"Face Detections: {face_count}")
        print(f"Processing Time: {processing_time:.3f} seconds")
        print("="*50)
    
    def process_video(self, video_path, output_path="output_video.mp4", 
                     show_live=True, save_analytics=True):
        """Process video file for crowd counting"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")
        
        # Analytics data
        analytics = []
        frame_count = 0
        start_time = time.time()
        
        # Create output directory for frames
        if save_analytics:
            video_output_dir = "video_analysis"
            os.makedirs(video_output_dir, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                # Perform detection
                boxes, _ = self.hog.detectMultiScale(frame, winStride=(8, 8), 
                                                     padding=(16, 16), scale=1.05)
                
                # Add to history for smoothing
                self.frame_history.append(len(boxes))
                smoothed_count = int(np.mean(self.frame_history))
                self.count_history.append(smoothed_count)
                
                # Draw detections
                for (x, y, w, h) in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add statistics overlay
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Count: {smoothed_count}", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save analytics
                if save_analytics and frame_count % (fps * 5) == 0:  # Every 5 seconds
                    analytics.append({
                        'timestamp': datetime.now().isoformat(),
                        'frame_number': frame_count,
                        'count': smoothed_count,
                        'fps': current_fps
                    })
            
            # Write frame to output video
            out.write(frame)
            
            # Display live if enabled
            if show_live:
                cv2.imshow('Video Processing', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Save analytics
        if save_analytics:
            analytics_path = os.path.join(video_output_dir, "video_analytics.json")
            with open(analytics_path, 'w') as f:
                json.dump(analytics, f, indent=4)
            
            # Create visualization of count over time
            self.plot_video_analytics(analytics)
        
        print(f"\nVideo processing complete!")
        print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")
        print(f"Output saved to: {output_path}")
        if save_analytics:
            print(f"Analytics saved to: {video_output_dir}/")
    
    def plot_video_analytics(self, analytics):
        """Plot analytics data from video processing"""
        if not analytics:
            return
        
        frames = [a['frame_number'] for a in analytics]
        counts = [a['count'] for a in analytics]
        timestamps = [a['timestamp'] for a in analytics]
        
        # Convert timestamps to seconds from start
        start_time = datetime.fromisoformat(timestamps[0])
        times = [(datetime.fromisoformat(ts) - start_time).total_seconds() 
                for ts in timestamps]
        
        plt.figure(figsize=(12, 6))
        
        # Plot count over time
        plt.subplot(2, 1, 1)
        plt.plot(times, counts, 'b-', linewidth=2, marker='o', markersize=4)
        plt.fill_between(times, 0, counts, alpha=0.3, color='blue')
        plt.xlabel('Time (seconds)')
        plt.ylabel('People Count')
        plt.title('Crowd Count Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Plot moving average
        window_size = 5
        if len(counts) > window_size:
            moving_avg = np.convolve(counts, np.ones(window_size)/window_size, mode='valid')
            plt.plot(times[window_size-1:], moving_avg, 'r-', linewidth=2, label=f'{window_size}-frame MA')
            plt.legend()
        
        # Plot histogram of counts
        plt.subplot(2, 1, 2)
        plt.hist(counts, bins=20, edgecolor='black', alpha=0.7, color='green')
        plt.xlabel('People Count')
        plt.ylabel('Frequency')
        plt.title('Distribution of Crowd Counts', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"""
        Statistics:
        Max Count: {max(counts)}
        Min Count: {min(counts)}
        Average Count: {np.mean(counts):.1f}
        Std Dev: {np.std(counts):.1f}
        """
        plt.figtext(0.75, 0.25, stats_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig('video_analysis/crowd_analytics.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def real_time_counting(self, camera_index=0, save_stream=False):
        """Real-time crowd counting from webcam"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize video writer if saving stream
        out = None
        if save_stream:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('realtime_output.avi', fourcc, 20.0, (640, 480))
        
        print("Starting real-time crowd counting...")
        print("Press 'q' to quit, 's' to save snapshot, 'r' to reset counter")
        
        # Performance tracking
        frame_times = deque(maxlen=100)
        start_time = time.time()
        frame_count = 0
        snapshot_count = 0
        
        # Create snapshots directory
        snapshot_dir = "snapshots"
        os.makedirs(snapshot_dir, exist_ok=True)
        
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect people (every 3rd frame for performance)
            if frame_count % 3 == 0:
                boxes, _ = self.hog.detectMultiScale(
                    frame,
                    winStride=(4, 4),
                    padding=(8, 8),
                    scale=1.05
                )
                
                # Update history
                self.frame_history.append(len(boxes))
                smoothed_count = int(np.mean(self.frame_history))
                
                # Draw detections
                for (x, y, w, h) in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Draw center point
                    center_x = x + w // 2
                    center_y = y + h // 2
                    cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            fps = 1.0 / np.mean(frame_times) if frame_times else 0
            
            # Draw UI elements
            elapsed_time = time.time() - start_time
            
            # Status panel
            cv2.rectangle(frame, (0, 0), (640, 100), (0, 0, 0, 180), -1)
            
            # Display information
            cv2.putText(frame, f"Live Crowd Counting", (20, 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, f"Count: {smoothed_count}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Time and frame info
            cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (400, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Frame: {frame_count}", (400, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Instructions
            cv2.putText(frame, "Q: Quit  S: Snapshot  R: Reset", (400, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Density indicator
            density_color = self.density_colors.get(
                self.get_density_level(smoothed_count, frame.shape[:2]), 
                (255, 255, 255)
            )
            cv2.circle(frame, (600, 30), 10, density_color, -1)
            
            # Write to output if saving
            if out is not None:
                out.write(frame)
            
            # Display frame
            cv2.imshow('Real-time Crowd Counting', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save snapshot
                snapshot_count += 1
                snapshot_path = os.path.join(snapshot_dir, 
                                           f"snapshot_{snapshot_count:03d}.jpg")
                cv2.imwrite(snapshot_path, frame)
                print(f"Snapshot saved: {snapshot_path}")
            elif key == ord('r'):
                # Reset counter
                self.frame_history.clear()
                print("Counter reset")
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print(f"\nReal-time counting session ended")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Snapshots saved: {snapshot_count}")
        if save_stream:
            print(f"Video saved: realtime_output.avi")


# Main execution with examples
if __name__ == "__main__":
    # Initialize the enhanced crowd counter
    print("Initializing Enhanced Crowd Counter...")
    counter = EnhancedCrowdCounter()
    
    # Example 1: Process a single image
    print("\n" + "="*50)
    print("EXAMPLE 1: Processing single image")
    print("="*50)
    
    image_path = 'image.png'  # Change this to your image path
    if os.path.exists(image_path):
        count, analytics = counter.count_crowd(
            image_path, 
            use_multi_scale=True, 
            save_output=True
        )
        print(f"✓ Image processed successfully")
    else:
        print(f"✗ Image not found: {image_path}")
        print("Using sample image instead...")
        # Create a sample image for demonstration
        sample_image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.imwrite('sample_image.png', sample_image)
        count, analytics = counter.count_crowd(
            'sample_image.png', 
            use_multi_scale=True, 
            save_output=True
        )
    
    # Example 2: Real-time webcam counting
    print("\n" + "="*50)
    print("EXAMPLE 2: Real-time webcam counting")
    print("="*50)
    
    # Uncomment to enable real-time counting
    # counter.real_time_counting(camera_index=0, save_stream=False)
    
    # Example 3: Process video file
    print("\n" + "="*50)
    print("EXAMPLE 3: Video processing")
    print("="*50)
    
    video_path = 'sample_video.mp4'  # Change this to your video path
    if os.path.exists(video_path):
        counter.process_video(
            video_path, 
            output_path="processed_video.mp4",
            show_live=True,
            save_analytics=True
        )
    else:
        print(f"✗ Video not found: {video_path}")
        print("To test video processing, please provide a video file.")
    
    # Example 4: Batch processing multiple images
    print("\n" + "="*50)
    print("EXAMPLE 4: Batch processing")
    print("="*50)
    
    image_folder = "images"  # Folder containing images to process
    if os.path.exists(image_folder):
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            print(f"Found {len(image_files)} images to process")
            
            results = []
            for i, img_file in enumerate(image_files, 1):
                img_path = os.path.join(image_folder, img_file)
                print(f"Processing {i}/{len(image_files)}: {img_file}")
                
                try:
                    count, analytics = counter.count_crowd(
                        img_path, 
                        use_multi_scale=True, 
                        save_output=True
                    )
                    results.append({
                        'image': img_file,
                        'count': count,
                        **analytics
                    })
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
            
            # Save batch results
            batch_results_path = "batch_results.json"
            with open(batch_results_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Batch results saved to {batch_results_path}")
        else:
            print(f"No images found in {image_folder}")
    else:
        print(f"Image folder not found: {image_folder}")
    
    print("\n" + "="*50)
    print("All examples completed successfully!")
    print("="*50)