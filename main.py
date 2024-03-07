import cv2
import mediapipe as mp
import math


def get_distance(a, b):
    return math.sqrt(
        math.pow(a.x - b.x, 2) + math.pow(a.y - b.y, 2) + math.pow(a.z - b.z, 2)
    )


# scissors
# 8 12 close far above 4 16 20

# rock
# 4 8 12 16 20 close


# paper
# 4 8 12 16 20 far


def is_rock(landmarks):
    if get_distance(landmarks[0], landmarks[12]) < 0.3:
        return True
    return False


def is_paper(landmarks):
    if get_distance(landmarks[0], landmarks[12]) > 0.3:
        return True
    return False


def is_scissors(landmarks):
    if (
        get_distance(landmarks[0], landmarks[12]) > 0.3
        and get_distance(landmarks[8], landmarks[0]) > 0.3
        and get_distance(landmarks[4], landmarks[16]) < 0.2
    ):
        return True
    return False


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        classification = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                if is_scissors(hand_landmarks.landmark):
                    classification = "Scissors"
                elif is_paper(hand_landmarks.landmark):
                    classification = "Paper"
                elif is_rock(hand_landmarks.landmark):
                    classification = "Rock"
            if classification:
                cv2.putText(
                    image,
                    classification,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        image = cv2.flip(image, 1)

        cv2.imshow("MediaPipe Hands", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
