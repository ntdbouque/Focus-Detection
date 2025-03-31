from flask import Flask, request, jsonify
import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)
mtcnn = MTCNN()
facenet = load_model('facenet_model.h5')  # Tải mô hình FaceNet (cần có sẵn)


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    face, confidence = extract_face(image)
    if face is None:
        return jsonify({'error': 'No face detected'}), 400
    
    # Trích xuất vector đặc trưng
    face = np.expand_dims(face, axis=0)  # Thêm batch dimension
    embedding = facenet.predict(face)[0]
    
    # Nhận diện sự tập trung
    attention_status = recognize_attention(embedding)
    
    return jsonify({'status': attention_status, 'confidence': confidence})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)