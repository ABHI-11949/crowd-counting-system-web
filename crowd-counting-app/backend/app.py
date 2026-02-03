from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import cv2
import numpy as np
import json
import time
from datetime import datetime
import threading
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import sys

# Add the parent directory to path to import crowd_counter
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import your crowd counter
try:
    from backend.crowd_counter import EnhancedCrowdCounter
    print("Successfully imported crowd_counter")
except ImportError as e:
    print(f"Import error: {e}")
    # Create a dummy class for testing
    class EnhancedCrowdCounter:
        def __init__(self):
            pass
        def count_crowd(self, image_path, use_multi_scale=True, save_output=True):
            return 10, {"density_level": "medium", "body_detections": 8, "face_detections": 2}
        def process_video(self, video_path, output_path, show_live=False, save_analytics=True):
            return True
        def real_time_counting(self, camera_index=0, save_stream=True):
            return True

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')

app = Flask(__name__, 
            static_folder=FRONTEND_DIR,
            template_folder=FRONTEND_DIR)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuration
app.config['SECRET_KEY'] = 'crowd-counting-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov', 'mkv'}

# Create necessary directories
def create_directories():
    directories = [
        app.config['UPLOAD_FOLDER'],
        'uploads/images',
        'uploads/videos',
        'static/results',
        'static/heatmaps',
        'static/videos',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

create_directories()

# Initialize crowd counter
counter = EnhancedCrowdCounter()
is_processing = False
processing_thread = None
realtime_active = False

def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crowd Counting System</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 40px; text-align: center; }}
                h1 {{ color: #3498db; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .status {{ padding: 20px; background: #f8f9fa; border-radius: 8px; margin: 20px 0; }}
                .success {{ color: #28a745; }}
                .error {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Crowd Counting System</h1>
                <div class="status">
                    <h2>Backend is running!</h2>
                    <p>Server time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Upload folder: {app.config['UPLOAD_FOLDER']}</p>
                    <p class="error">Note: Frontend files not found at {FRONTEND_DIR}</p>
                    <p>Please ensure your frontend files are in the correct location.</p>
                </div>
                <h3>API Endpoints:</h3>
                <ul style="text-align: left; display: inline-block;">
                    <li><code>GET /api/health</code> - Health check</li>
                    <li><code>POST /api/upload/image</code> - Upload and process image</li>
                    <li><code>POST /api/upload/video</code> - Upload and process video</li>
                    <li><code>POST /api/batch</code> - Batch process images</li>
                    <li><code>POST /api/realtime/start</code> - Start real-time counting</li>
                    <li><code>POST /api/realtime/stop</code> - Stop real-time counting</li>
                </ul>
            </div>
        </body>
        </html>
        """, 200

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from frontend directory"""
    return send_from_directory(app.static_folder, path)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'crowd-counting-api',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'image_upload': '/api/upload/image',
            'video_upload': '/api/upload/video',
            'batch_upload': '/api/batch',
            'realtime_start': '/api/realtime/start',
            'realtime_stop': '/api/realtime/stop',
            'analytics': '/api/analytics',
            'settings': '/api/settings'
        }
    })

@app.route('/api/upload/image', methods=['POST'])
def upload_image():
    """Handle image upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Secure filename
        original_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{original_filename}"
        filepath = os.path.join('uploads', 'images', filename)
        
        # Save file
        file.save(filepath)
        print(f"Image saved: {filepath}")
        
        # Emit processing start
        socketio.emit('processing_start', {
            'message': f'Processing {original_filename}',
            'filename': original_filename,
            'timestamp': datetime.now().isoformat()
        })
        
        # Process image using crowd counter
        count, analytics = counter.count_crowd(
            filepath, 
            use_multi_scale=True, 
            save_output=True
        )
        
        # Generate result and heatmap paths
        result_image = f"result_{timestamp}.jpg"
        heatmap_image = f"heatmap_{timestamp}.jpg"
        
        result_path = os.path.join('static', 'results', result_image)
        heatmap_path = os.path.join('static', 'heatmaps', heatmap_image)
        
        # For demo purposes, create dummy images if they don't exist
        if not os.path.exists(result_path):
            # Create a simple result image
            img = Image.new('RGB', (800, 600), color='white')
            img.save(result_path)
        
        if not os.path.exists(heatmap_path):
            # Create a simple heatmap image
            img = Image.new('RGB', (800, 600), color='red')
            img.save(heatmap_path)
        
        # Prepare response
        result = {
            'success': True,
            'original_filename': original_filename,
            'count': count,
            'analytics': {
                'density_level': analytics.get('density_level', 'medium'),
                'body_detections': analytics.get('body_detections', count),
                'face_detections': analytics.get('face_detections', 0),
                'timestamp': datetime.now().isoformat(),
                'inference_time': analytics.get('inference_time', 0.5)
            },
            'result_image': f"/static/results/{result_image}",
            'heatmap_image': f"/static/heatmaps/{heatmap_image}",
            'timestamp': datetime.now().isoformat(),
            'file_size': os.path.getsize(filepath)
        }
        
        # Emit completion
        socketio.emit('processing_complete', result)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        socketio.emit('processing_error', {'error': str(e)})
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload/video', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Secure filename
        original_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{original_filename}"
        filepath = os.path.join('uploads', 'videos', filename)
        
        # Save file
        file.save(filepath)
        print(f"Video saved: {filepath}")
        
        # Emit processing start
        socketio.emit('processing_start', {
            'message': f'Processing video: {original_filename}',
            'filename': original_filename,
            'timestamp': datetime.now().isoformat()
        })
        
        # Process video in background
        def process_video_task():
            try:
                output_filename = f"processed_{timestamp}.mp4"
                output_path = os.path.join('static', 'videos', output_filename)
                
                # Create output directory if not exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Process video using crowd counter
                counter.process_video(
                    filepath,
                    output_path=output_path,
                    show_live=False,
                    save_analytics=True
                )
                
                # Create analytics file
                analytics_data = {
                    'filename': original_filename,
                    'processing_time': datetime.now().isoformat(),
                    'video_duration': 0,  # You can extract this from video
                    'average_count': 15,  # Example data
                    'peak_count': 25,     # Example data
                    'frames_processed': 100  # Example data
                }
                
                analytics_file = f"video_analytics_{timestamp}.json"
                analytics_path = os.path.join('static', 'results', analytics_file)
                
                with open(analytics_path, 'w') as f:
                    json.dump(analytics_data, f, indent=4)
                
                result = {
                    'success': True,
                    'original_filename': original_filename,
                    'output_video': f"/static/videos/{output_filename}",
                    'analytics_file': f"/static/results/{analytics_file}",
                    'message': 'Video processing complete',
                    'timestamp': datetime.now().isoformat()
                }
                
                socketio.emit('video_processing_complete', result)
                
            except Exception as e:
                print(f"Error in video processing thread: {str(e)}")
                socketio.emit('processing_error', {'error': str(e)})
        
        # Start processing thread
        thread = threading.Thread(target=process_video_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Video processing started in background',
            'filename': original_filename,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error uploading video: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/realtime/start', methods=['POST'])
def start_realtime():
    """Start real-time crowd counting"""
    global realtime_active
    
    try:
        if realtime_active:
            return jsonify({'error': 'Real-time counting already active'}), 400
        
        camera_index = request.json.get('camera_index', 0)
        
        def realtime_task():
            global realtime_active
            realtime_active = True
            
            try:
                # Start real-time counting
                counter.real_time_counting(
                    camera_index=camera_index,
                    save_stream=True
                )
                
                # Example: Send periodic updates
                while realtime_active:
                    # Simulate real-time data
                    data = {
                        'count': np.random.randint(5, 50),
                        'density_level': np.random.choice(['low', 'medium', 'high']),
                        'timestamp': datetime.now().isoformat()
                    }
                    socketio.emit('realtime_update', data)
                    time.sleep(1)  # Update every second
                    
            except Exception as e:
                print(f"Error in real-time counting: {str(e)}")
                socketio.emit('realtime_error', {'error': str(e)})
            finally:
                realtime_active = False
        
        # Start real-time thread
        thread = threading.Thread(target=realtime_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Real-time counting started',
            'camera_index': camera_index,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error starting real-time: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/realtime/stop', methods=['POST'])
def stop_realtime():
    """Stop real-time crowd counting"""
    global realtime_active
    
    realtime_active = False
    
    return jsonify({
        'success': True,
        'message': 'Real-time counting stopped',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/batch', methods=['POST'])
def batch_process():
    """Handle batch image processing"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        results = []
        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                original_filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{timestamp}_{original_filename}"
                filepath = os.path.join('uploads', 'images', filename)
                
                # Save file
                file.save(filepath)
                
                # Emit progress
                socketio.emit('batch_progress', {
                    'current': i + 1,
                    'total': len(files),
                    'filename': original_filename,
                    'progress': ((i + 1) / len(files)) * 100
                })
                
                # Process image (simulated for now)
                count, analytics = counter.count_crowd(filepath, use_multi_scale=True)
                
                results.append({
                    'filename': original_filename,
                    'count': count,
                    'analytics': analytics,
                    'success': True
                })
        
        # Save batch results
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_filename = f"batch_results_{batch_timestamp}.json"
        batch_path = os.path.join('static', 'results', batch_filename)
        
        with open(batch_path, 'w') as f:
            json.dump({
                'timestamp': batch_timestamp,
                'total_files': len(results),
                'results': results
            }, f, indent=4)
        
        return jsonify({
            'success': True,
            'message': f'Successfully processed {len(results)} files',
            'results': results,
            'batch_file': f"/static/results/{batch_filename}",
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """Get or update settings"""
    if request.method == 'GET':
        # Return current settings
        return jsonify({
            'success': True,
            'settings': {
                'confidence_threshold': 0.5,
                'detection_method': 'multi_scale',
                'enable_heatmap': True,
                'enable_density_map': True,
                'max_detections': 100
            },
            'timestamp': datetime.now().isoformat()
        })
    
    else:  # POST
        try:
            settings = request.json
            
            # Here you would update the counter settings
            # For now, just acknowledge receipt
            
            return jsonify({
                'success': True,
                'message': 'Settings updated',
                'settings': settings,
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data"""
    try:
        # Get all analytics files
        analytics_dir = 'static/results'
        analytics_files = []
        
        if os.path.exists(analytics_dir):
            analytics_files = [f for f in os.listdir(analytics_dir) 
                             if f.endswith('.json') and 'analytics' in f]
        
        analytics_data = []
        
        # Read last 10 analytics files
        for file in sorted(analytics_files, reverse=True)[:10]:
            filepath = os.path.join(analytics_dir, file)
            with open(filepath, 'r') as f:
                data = json.load(f)
                analytics_data.append({
                    'file': file,
                    'data': data,
                    'created': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                })
        
        return jsonify({
            'success': True,
            'analytics': analytics_data,
            'count': len(analytics_data),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error getting analytics: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get processing history"""
    try:
        history = []
        
        # Get processed images
        results_dir = 'static/results'
        if os.path.exists(results_dir):
            result_files = [f for f in os.listdir(results_dir) if f.endswith('.jpg')]
            
            for file in sorted(result_files, reverse=True)[:10]:
                filepath = os.path.join(results_dir, file)
                history.append({
                    'type': 'image',
                    'file': file,
                    'url': f"/static/results/{file}",
                    'created': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                })
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error getting history: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static_file(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    emit('connection_response', {
        'status': 'connected',
        'message': 'Welcome to Crowd Counting System',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

@socketio.on('message')
def handle_message(data):
    print(f'Message from client: {data}')
    emit('response', {'data': 'Message received'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Crowd Counting System")
    print("="*50)
    print(f"Frontend directory: {FRONTEND_DIR}")
    print(f"Static files available: {os.path.exists(FRONTEND_DIR)}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Server URL: http://localhost:5500")
    print(f"API Health check: http://localhost:5500/api/health")
    print("="*50 + "\n")
    
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5500, 
        debug=True, 
        allow_unsafe_werkzeug=True
    )