from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from model.model import predict_ocd
import cv2
from functools import wraps
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def error_handler(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    return wrapper

class Camera:
    def __init__(self):
        self.camera = None
    
    def get_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
        return self.camera
    
    def release(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

camera = Camera()

@app.route('/predict', methods=['POST'])
@error_handler
def predict():
    """Predict OCD severity and percentage from the input data."""
    data = request.get_json()
    if not data:
        response = jsonify({'error': 'Invalid input data'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 400
    
    logger.info(f"Received input data: {data}")
    severity, percentage = predict_ocd(data)
    response = jsonify({
        'predicted_severity': severity, 
        'predicted_percentage': percentage
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

def generate_frames():
    cam = camera.get_camera()
    try:
        while True:
            success, frame = cam.read()
            if not success:
                break
            
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   buffer.tobytes() + 
                   b'\r\n')
    except GeneratorExit:
        camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)