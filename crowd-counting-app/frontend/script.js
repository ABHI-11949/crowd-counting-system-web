class CrowdCountingApp {
    constructor() {
        this.socket = null;
        this.currentFile = null;
        this.results = null;
        this.isProcessing = false;
        this.isRealTime = false;
        
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.setupEventListeners();
        this.checkServerStatus();
    }
    
    connectWebSocket() {
        this.socket = io('http://localhost:5500');
        
        this.socket.on('connect', () => {
            this.showNotification('Connected to server', 'success');
            document.getElementById('statusDot').classList.add('connected');
            document.getElementById('statusText').textContent = 'Connected';
        });
        
        this.socket.on('disconnect', () => {
            this.showNotification('Disconnected from server', 'warning');
            document.getElementById('statusDot').classList.remove('connected');
            document.getElementById('statusText').textContent = 'Disconnected';
        });
        
        this.socket.on('processing_start', (data) => {
            this.showNotification(data.message, 'info');
            this.showLoading();
        });
        
        this.socket.on('processing_complete', (data) => {
            this.hideLoading();
            this.showNotification('Processing complete!', 'success');
            this.displayResults(data);
        });
        
        this.socket.on('video_processing_complete', (data) => {
            this.hideLoading();
            this.showNotification('Video processing complete!', 'success');
            console.log('Video processed:', data);
        });
        
        this.socket.on('batch_progress', (data) => {
            this.showNotification(`Processing ${data.current}/${data.total}: ${data.filename}`, 'info');
        });
        
        this.socket.on('processing_error', (data) => {
            this.hideLoading();
            this.showNotification(`Error: ${data.error}`, 'error');
        });
        
        this.socket.on('realtime_error', (data) => {
            this.hideLoading();
            this.showNotification(`Real-time error: ${data.error}`, 'error');
        });
    }
    
    setupEventListeners() {
        // File upload
        document.getElementById('imageUpload').addEventListener('change', (e) => this.handleFileUpload(e, 'image'));
        document.getElementById('videoUpload').addEventListener('change', (e) => this.handleFileUpload(e, 'video'));
        document.getElementById('batchUpload').addEventListener('change', (e) => this.handleBatchUpload(e));
        
        // Upload buttons
        document.getElementById('uploadImageBtn').addEventListener('click', () => document.getElementById('imageUpload').click());
        document.getElementById('uploadVideoBtn').addEventListener('click', () => document.getElementById('videoUpload').click());
        document.getElementById('uploadBatchBtn').addEventListener('click', () => document.getElementById('batchUpload').click());
        
        // Process buttons
        document.getElementById('processBtn').addEventListener('click', () => this.processImage());
        document.getElementById('processVideoBtn').addEventListener('click', () => this.processVideo());
        document.getElementById('startRealtimeBtn').addEventListener('click', () => this.startRealTime());
        document.getElementById('stopRealtimeBtn').addEventListener('click', () => this.stopRealTime());
        
        // Settings
        document.getElementById('confidenceSlider').addEventListener('input', (e) => {
            document.getElementById('confidenceValue').textContent = `${e.target.value}%`;
        });
        
        // Export buttons
        document.getElementById('exportBtn').addEventListener('click', () => this.exportResults());
        document.getElementById('clearBtn').addEventListener('click', () => this.clearResults());
        
        // Drag and drop
        this.setupDragAndDrop();
    }
    
    setupDragAndDrop() {
        const dropZone = document.querySelector('.preview-container');
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#3498db';
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.style.borderColor = '#ddd';
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#ddd';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleDroppedFile(files[0]);
            }
        });
    }
    
    async handleFileUpload(event, type) {
        const file = event.target.files[0];
        if (!file) return;
        
        this.currentFile = {
            file: file,
            type: type,
            name: file.name,
            size: this.formatFileSize(file.size)
        };
        
        await this.previewFile(file, type);
        
        // Show process button based on file type
        if (type === 'image') {
            document.getElementById('processBtn').disabled = false;
        } else if (type === 'video') {
            document.getElementById('processVideoBtn').disabled = false;
        }
        
        this.showNotification(`${file.name} uploaded successfully`, 'success');
    }
    
    async handleBatchUpload(event) {
        const files = Array.from(event.target.files);
        if (files.length === 0) return;
        
        this.showLoading();
        
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });
        
        try {
            const response = await fetch('/api/batch', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(`Batch processed: ${data.message}`, 'success');
                console.log('Batch results:', data.results);
            } else {
                this.showNotification(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async handleDroppedFile(file) {
        const type = file.type.startsWith('image/') ? 'image' : 
                     file.type.startsWith('video/') ? 'video' : null;
        
        if (!type) {
            this.showNotification('Unsupported file type', 'error');
            return;
        }
        
        this.currentFile = {
            file: file,
            type: type,
            name: file.name,
            size: this.formatFileSize(file.size)
        };
        
        await this.previewFile(file, type);
        
        if (type === 'image') {
            document.getElementById('processBtn').disabled = false;
        }
        
        this.showNotification(`${file.name} dropped successfully`, 'success');
    }
    
    async previewFile(file, type) {
        const preview = document.getElementById('previewImage');
        const placeholder = document.querySelector('.preview-placeholder');
        
        if (type === 'image') {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
                placeholder.style.display = 'none';
            };
            reader.readAsDataURL(file);
        } else if (type === 'video') {
            const videoUrl = URL.createObjectURL(file);
            preview.src = videoUrl;
            preview.style.display = 'block';
            placeholder.style.display = 'none';
        }
    }
    
    async processImage() {
        if (!this.currentFile || this.currentFile.type !== 'image') {
            this.showNotification('Please upload an image first', 'warning');
            return;
        }
        
        this.showLoading();
        
        const formData = new FormData();
        formData.append('file', this.currentFile.file);
        
        try {
            const response = await fetch('/api/upload/image', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.results = data;
                this.displayResults(data);
                this.showNotification('Image processed successfully', 'success');
            } else {
                this.showNotification(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async processVideo() {
        if (!this.currentFile || this.currentFile.type !== 'video') {
            this.showNotification('Please upload a video first', 'warning');
            return;
        }
        
        this.showLoading();
        
        const formData = new FormData();
        formData.append('file', this.currentFile.file);
        
        try {
            const response = await fetch('/api/upload/video', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Video processing started', 'success');
            } else {
                this.showNotification(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
        }
    }
    
    async startRealTime() {
        this.showLoading();
        
        try {
            const response = await fetch('/api/realtime/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    camera_index: 0
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.isRealTime = true;
                document.getElementById('startRealtimeBtn').disabled = true;
                document.getElementById('stopRealtimeBtn').disabled = false;
                this.showNotification('Real-time counting started', 'success');
            } else {
                this.showNotification(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async stopRealTime() {
        try {
            const response = await fetch('/api/realtime/stop', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.isRealTime = false;
                document.getElementById('startRealtimeBtn').disabled = false;
                document.getElementById('stopRealtimeBtn').disabled = true;
                this.showNotification('Real-time counting stopped', 'success');
            } else {
                this.showNotification(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
        }
    }
    
    displayResults(data) {
        // Update statistics
        document.getElementById('totalCount').textContent = data.count || 0;
        document.getElementById('bodyCount').textContent = data.analytics?.body_detections || 0;
        document.getElementById('faceCount').textContent = data.analytics?.face_detections || 0;
        
        // Update density level
        const densityLevel = data.analytics?.density_level || 'low';
        const densityElement = document.getElementById('densityLevel');
        densityElement.textContent = densityLevel.toUpperCase();
        densityElement.className = `density-${densityLevel}`;
        
        // Display result images
        if (data.result_image) {
            document.getElementById('resultImage').src = data.result_image;
            document.getElementById('resultImage').style.display = 'block';
        }
        
        if (data.heatmap_image) {
            document.getElementById('heatmapImage').src = data.heatmap_image;
            document.getElementById('heatmapImage').style.display = 'block';
        }
        
        // Show results section
        document.querySelector('.results-section').style.display = 'block';
        
        // Scroll to results
        document.querySelector('.results-section').scrollIntoView({
            behavior: 'smooth'
        });
    }
    
    async exportResults() {
        if (!this.results) {
            this.showNotification('No results to export', 'warning');
            return;
        }
        
        try {
            const blob = new Blob([JSON.stringify(this.results, null, 2)], {
                type: 'application/json'
            });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `crowd-counting-results-${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.showNotification('Results exported successfully', 'success');
        } catch (error) {
            this.showNotification(`Export error: ${error.message}`, 'error');
        }
    }
    
    clearResults() {
        this.results = null;
        this.currentFile = null;
        
        // Reset preview
        const preview = document.getElementById('previewImage');
        preview.src = '';
        preview.style.display = 'none';
        document.querySelector('.preview-placeholder').style.display = 'flex';
        
        // Reset results
        document.getElementById('totalCount').textContent = '0';
        document.getElementById('bodyCount').textContent = '0';
        document.getElementById('faceCount').textContent = '0';
        
        // Hide result images
        document.getElementById('resultImage').style.display = 'none';
        document.getElementById('heatmapImage').style.display = 'none';
        
        // Hide results section
        document.querySelector('.results-section').style.display = 'none';
        
        // Reset buttons
        document.getElementById('processBtn').disabled = true;
        document.getElementById('processVideoBtn').disabled = true;
        
        this.showNotification('All results cleared', 'info');
    }
    
    async checkServerStatus() {
        try {
        const response = await fetch('/api/health');
        const text = await response.text();

if (!text) return; // backend returned nothing

const data = JSON.parse(text);

            if (data.status === 'healthy') {
                console.log('Server is healthy');
            }
        } catch (error) {
            console.warn('Server not responding:', error);
        }
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${this.getNotificationIcon(type)}"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
    
    getNotificationIcon(type) {
        switch(type) {
            case 'success': return 'check-circle';
            case 'error': return 'exclamation-circle';
            case 'warning': return 'exclamation-triangle';
            default: return 'info-circle';
        }
    }
    
    showLoading() {
        this.isProcessing = true;
        document.getElementById('loadingOverlay').style.display = 'flex';
    }
    
    hideLoading() {
        this.isProcessing = false;
        document.getElementById('loadingOverlay').style.display = 'none';
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    const app = new CrowdCountingApp();
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl + P to process
        if (e.ctrlKey && e.key === 'p') {
            e.preventDefault();
            document.getElementById('processBtn').click();
        }
        
        // Ctrl + U to upload
        if (e.ctrlKey && e.key === 'u') {
            e.preventDefault();
            document.getElementById('uploadImageBtn').click();
        }
        
        // Escape to clear
        if (e.key === 'Escape') {
            app.clearResults();
        }
    });
});