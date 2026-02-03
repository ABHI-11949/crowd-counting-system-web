import cv2
import numpy as np
import json
import time
import os
from datetime import datetime
from PIL import Image, ImageDraw
import random

class EnhancedCrowdCounter:
    def __init__(self):
        print("Initializing Enhanced Crowd Counter...")
        self.model_loaded = False
        self.current_frame = None
        self.is_realtime = False
        
    def count_crowd(self, image_path, use_multi_scale=True, save_output=True):
        """Count crowd in an image"""
        print(f"Processing image: {image_path}")
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width = img.shape[:2]
            
            # Simulate crowd counting (replace with actual model)
            # For now, generate random detections based on image size
            max_detections = int((width * height) / 5000)  # Scale with image size
            count = random.randint(max_detections // 2, max_detections)
            
            # Generate random detection points
            detections = []
            for _ in range(count):
                x = random.randint(50, width - 50)
                y = random.randint(50, height - 50)
                detections.append((x, y))
            
            # Create result image with bounding boxes
            result_img = img.copy()
            for (x, y) in detections:
                cv2.circle(result_img, (x, y), 10, (0, 255, 0), 2)
                cv2.putText(result_img, 'P', (x-5, y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Create heatmap (simulated)
            heatmap = self.create_heatmap(img, detections)
            
            # Save outputs if requested
            if save_output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                result_path = f"static/results/result_{timestamp}.jpg"
                heatmap_path = f"static/heatmaps/heatmap_{timestamp}.jpg"
                
                # Ensure directories exist
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
                
                cv2.imwrite(result_path, result_img)
                cv2.imwrite(heatmap_path, heatmap)
                
                print(f"Results saved to: {result_path}, {heatmap_path}")
            
            # Prepare analytics
            analytics = {
                'density_level': self.get_density_level(count),
                'body_detections': count,
                'face_detections': random.randint(0, count // 2),
                'image_dimensions': f"{width}x{height}",
                'inference_time': random.uniform(0.1, 0.5),
                'detection_method': 'simulated' if not use_multi_scale else 'multi_scale_simulated',
                'confidence_score': random.uniform(0.7, 0.95)
            }
            
            return count, analytics
            
        except Exception as e:
            print(f"Error in count_crowd: {str(e)}")
            # Return dummy data for testing
            return 10, {
                'density_level': 'medium',
                'body_detections': 10,
                'face_detections': 5,
                'error': str(e)
            }
    
    def create_heatmap(self, img, detections):
        """Create a heatmap from detections"""
        height, width = img.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for (x, y) in detections:
            # Create Gaussian blob at detection point
            for i in range(-15, 16):
                for j in range(-15, 16):
                    nx, ny = x + i, y + j
                    if 0 <= nx < width and 0 <= ny < height:
                        distance = np.sqrt(i**2 + j**2)
                        if distance <= 15:
                            intensity = np.exp(-distance**2 / (2 * 7**2))
                            heatmap[ny, nx] += intensity
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max() * 255
        
        # Convert to BGR for display
        heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original image
        blended = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
        
        return blended
    
    def get_density_level(self, count):
        """Determine density level based on count"""
        if count < 10:
            return "low"
        elif count < 30:
            return "medium"
        else:
            return "high"
    
    def process_video(self, video_path, output_path, show_live=False, save_analytics=True):
        """Process video for crowd counting"""
        print(f"Processing video: {video_path}")
        
        try:
            # Simulate video processing
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video info: {width}x{height}, {fps} fps")
            
            # For now, just create a dummy output
            # In real implementation, process each frame
            
            # Create a simple output video (just a black frame with text)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write a single frame for demonstration
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, "Video Processing Complete", (50, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Original: {os.path.basename(video_path)}", 
                       (50, height//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            for _ in range(fps * 5):  # 5 seconds of video
                out.write(frame)
            
            cap.release()
            out.release()
            
            print(f"Output video saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return False
    
    def real_time_counting(self, camera_index=0, save_stream=True):
        """Real-time crowd counting from camera"""
        print(f"Starting real-time counting from camera {camera_index}")
        
        self.is_realtime = True
        
        try:
            # Try to open camera
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"Could not open camera {camera_index}, using simulated feed")
                # For simulation, we'll just run for a while
                for i in range(30):  # Run for ~30 seconds
                    if not self.is_realtime:
                        break
                    time.sleep(1)
                return
            
            # Simulate processing for a while
            start_time = time.time()
            while self.is_realtime and (time.time() - start_time) < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Simulate counting on frame
                height, width = frame.shape[:2]
                count = random.randint(1, 20)
                
                # Add detection markers
                for _ in range(count):
                    x = random.randint(50, width - 50)
                    y = random.randint(50, height - 50)
                    cv2.circle(frame, (x, y), 8, (0, 255, 0), 2)
                
                # Add count display
                cv2.putText(frame, f"Count: {count}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show frame (optional)
                cv2.imshow('Real-time Crowd Counting', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error in real-time counting: {str(e)}")
        finally:
            self.is_realtime = False
    
    def stop_realtime(self):
        """Stop real-time counting"""
        self.is_realtime = False
        print("Real-time counting stopped")