import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine

mtcnn = MTCNN(keep_all=True)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

registered_embeddings = {
    "user_1": np.random.rand(512), 
    "user_2": np.random.rand(512),
}

def get_embedding(face):
    """Trích xuất embedding từ ảnh khuôn mặt"""
    face = torch.tensor(face).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        return facenet(face).squeeze().numpy()

def recognize_face(face):
    """So sánh embedding với danh sách đã đăng ký"""
    embedding = get_embedding(face)
    best_match, best_score = None, float('inf')
    for name, ref_emb in registered_embeddings.items():
        score = cosine(embedding, ref_emb)
        if score < best_score:
            best_match, best_score = name, score
    return best_match if best_score < 0.6 else "Unknown"

def detect_attention(frame):
    """Phát hiện mất tập trung dựa trên hướng mặt & trạng thái mắt"""
    boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
    if boxes is not None:
        for box, landmark in zip(boxes, landmarks):
            x, y, x2, y2 = map(int, box)
            face = frame[y:y2, x:x2]
            user = recognize_face(face)
            
            eye_dist = np.linalg.norm(landmark[0] - landmark[1])
            if eye_dist < 5:  
                status = "Mất tập trung: Nhắm mắt"
            else:
                status = "Tập trung"
            
            cv2.putText(frame, f"{user}: {status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
    return frame

def run_realtime():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_attention(frame)
        cv2.imshow('Attention Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()