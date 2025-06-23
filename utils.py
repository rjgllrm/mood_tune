import numpy as np
import mediapipe as mp

class MediaPipeProcessor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic()
        self.drawing = mp.solutions.drawing_utils
        self.hands = mp.solutions.hands

    def extract_landmarks(self, result):
        lst = []

        if result.face_landmarks:
            for i in result.face_landmarks.landmark:
                lst.append(i.x - result.face_landmarks.landmark[1].x)
                lst.append(i.y - result.face_landmarks.landmark[1].y)

            lst.extend(self.extract_hand_landmarks(result.left_hand_landmarks, 8))
            lst.extend(self.extract_hand_landmarks(result.right_hand_landmarks, 8))

        return lst

    def extract_hand_landmarks(self, hand_landmarks, ref_idx):
        if hand_landmarks:
            ref = hand_landmarks.landmark[ref_idx]
            return [coord for lm in hand_landmarks.landmark for coord in (lm.x - ref.x, lm.y - ref.y)]
        else:
            return [0.0] * 42

    def draw_landmarks(self, frame, result):
        self.drawing.draw_landmarks(frame, result.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)
        self.drawing.draw_landmarks(frame, result.left_hand_landmarks, self.hands.HAND_CONNECTIONS)
        self.drawing.draw_landmarks(frame, result.right_hand_landmarks, self.hands.HAND_CONNECTIONS)