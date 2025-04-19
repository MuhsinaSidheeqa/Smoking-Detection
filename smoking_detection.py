import cv2
import mediapipe as mp
import math
from ultralytics import YOLO


model = YOLO("best.pt")

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils


def get_landmark_position(landmarks, index, frame_shape):
    h, w, _ = frame_shape
    x = int(landmarks[index].x * w)
    y = int(landmarks[index].y * h)
    return x, y

def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


cap = cv2.VideoCapture(r'C:\smoking\peaky_blinders.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    yolo_results = model(frame, verbose=False)[0]
    cigarette_detected = any(
        model.names[int(box.cls[0])].lower() in ["smooking"] and float(box.conf[0]) > 0.5
        for box in yolo_results.boxes
    )


    hand_results = mp_hands.process(frame_rgb)
    face_results = mp_face.process(frame_rgb)

    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)


    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1))


    smoking_detected = False

    if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0].landmark
        face_landmarks = face_results.multi_face_landmarks[0].landmark


        hand_tip = get_landmark_position(hand_landmarks, 8, frame.shape)
        mouth = get_landmark_position(face_landmarks, 13, frame.shape)

        distance = euclidean_distance(hand_tip, mouth)

        if cigarette_detected and distance < 60:
            smoking_detected = True


    if smoking_detected:
        cv2.putText(frame, "Smoking Detected", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No Smoking", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    
    cv2.imshow("Smoking Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
