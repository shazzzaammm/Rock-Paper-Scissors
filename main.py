import cv2
import mediapipe as mp
import math
import keyboard

ROCK = "Rock"
PAPER = "Paper"
SCISSORS = "Scissors"

STATE_COUNTDOWN = 0
STATE_SCORING = 1
current_state = STATE_COUNTDOWN
classifications = []

frames = 0


def get_distance(a, b):
    return math.sqrt(
        math.pow(a.x - b.x, 2) + math.pow(a.y - b.y, 2) + math.pow(a.z - b.z, 2)
    )


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
    max_num_hands=2,
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # For speed
        image.flags.writeable = False

        # Pre processing
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if current_state == STATE_COUNTDOWN:
            frames += 1
            count_num = frames // 10
            text_size = cv2.getTextSize(str(count_num), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[
                0
            ]
            cv2.putText(
                image,
                str(count_num),
                (
                    (image.shape[1] - text_size[0]) // 2,
                    text_size[1],
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_4,
            )
            if count_num == 3:
                current_state = STATE_SCORING

        # Get results
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw stuffs
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                if current_state == STATE_SCORING:
                    # Get classification
                    if is_scissors(hand_landmarks.landmark):
                        classification = SCISSORS
                    elif is_paper(hand_landmarks.landmark):
                        classification = PAPER
                    elif is_rock(hand_landmarks.landmark):
                        classification = ROCK

                    # Display classification
                    if classification:
                        classifications.append(classification)
                        text_size = cv2.getTextSize(
                            classification, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                        )[0]
                        cv2.putText(
                            image,
                            classification,
                            (
                                int(hand_landmarks.landmark[0].x * image.shape[1])
                                - (text_size[0] // 2),
                                int(hand_landmarks.landmark[0].y * image.shape[0])
                                + text_size[1],
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255 * i, 255, 255 * i),
                            2,
                            cv2.LINE_AA,
                        )

        game_result = "error reading both hands"

        if len(classifications) > 1 and current_state == STATE_SCORING:
            if classifications[0] == classifications[1]:
                game_result = "Tie"
            elif (
                (classifications[0] == SCISSORS and classifications[1] == ROCK)
                or (classifications[0] == ROCK and classifications[1] == PAPER)
                or (classifications[0] == PAPER and classifications[1] == SCISSORS)
            ):
                game_result = "White wins"
            elif (
                (classifications[1] == SCISSORS and classifications[0] == ROCK)
                or (classifications[1] == ROCK and classifications[0] == PAPER)
                or (classifications[1] == PAPER and classifications[0] == SCISSORS)
            ):
                game_result = "Green wins"
            text_size = cv2.getTextSize(game_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(
                image,
                game_result,
                (
                    (image.shape[1] - text_size[0]) // 2,
                    text_size[1],
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Show the image
        cv2.imshow("MediaPipe Hands", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

        if keyboard.is_pressed("space"):
            current_state = STATE_COUNTDOWN
            classifications = []
            frames = 0
cap.release()
