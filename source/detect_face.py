import cv2
from mtcnn import MTCNN
import time

def detect(detector, video_path):
    print('Detect in video:', video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error: Could not open video.')
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Original FPS: {original_fps}')

    skip_frames = int(original_fps / 15)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Reached the end of the video or failed to read the frame.')
            break
        
        if frame_count % skip_frames == 0:
            faces = detector.detect_faces(frame)
            

            for face in faces:
                if face['confidence'] > 0.8:
                    x, y, width, height = face['box']
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 155, 255), 2)
                    cv2.putText(frame, f"score: {face['confidence']}", (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow('Frame', frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = MTCNN()
    video_path = r'C:\Users\duy\Desktop\Focus\sample\videoplayback.mp4'
    detect(detector, video_path)
