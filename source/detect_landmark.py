import cv2
import dlip

def detect_landmark(detector, predictor, img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # Hiển thị ảnh
    cv2.imshow("Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = r'C:\Users\duy\Desktop\Focus\model\shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)


    img_path = r'C:\Users\duy\Desktop\Focus\sample\OIP.jpg'
    detect_landmark(detector, predictor, img_path)