#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import csv
import copy
import argparse
import itertools
from collections import Counter, deque
import threading
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, \
    QProgressBar, QSizePolicy
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QFontDatabase
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


# --- This class handles the background AI processing ---
class GestureThread(QThread):
    # Signals to send data to the main UI thread
    change_pixmap_signal = Signal(np.ndarray)
    update_data_signal = Signal(dict)

    def __init__(self):
        super().__init__()
        self._running = True

    def run(self):
        """
        The engine of the application. Runs in a separate thread.
        """
        args = get_args()
        cap = cv.VideoCapture(args.device)
        if not cap.isOpened():
            print(f"Error: Cannot open camera device {args.device}")
            self.update_data_signal.emit({"error": "Camera not found or in use."})
            return

        # Try to set camera resolution and FPS (some cameras ignore FPS setting)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv.CAP_PROP_FPS, 60) # Request 60 FPS, camera might deliver less

        # Load models and labels
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=args.use_static_image_mode,
            max_num_hands=2,
            model_complexity=0, # Set model complexity to 0 for fastest CPU performance
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        keypoint_classifier = KeyPointClassifier()
        point_history_classifier = PointHistoryClassifier()

        try:
            with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
                keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
            with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
                point_history_classifier_labels = [row[0] for row in csv.reader(f)]
        except FileNotFoundError:
            print("Error: Label CSV files not found. Please ensure they are in the 'model' directory.")
            self.update_data_signal.emit({"error": "Model label files missing."})
            return

        cvFpsCalc = CvFpsCalc(buffer_len=10)
        history_length = 16
        point_history = deque(maxlen=history_length)
        finger_gesture_history = deque(maxlen=history_length)

        current_streak_sign = ""
        current_streak_gesture = ""
        sign_streak_count = 0
        gesture_streak_count = 0
        MIN_STREAK_LENGTH = 5

        while self._running:
            fps = cvFpsCalc.get()
            ret, image = cap.read()
            if not ret:
                print("Warning: Failed to grab frame, attempting to re-open camera...")
                cap.release()
                time.sleep(1)
                cap = cv.VideoCapture(args.device)
                if not cap.isOpened():
                    print("Error: Could not re-open camera. Exiting gesture thread.")
                    self._running = False
                continue

            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            current_hand_sign = "N/A"
            current_finger_gesture = "N/A"
            current_handedness = "N/A"
            detection_confidence = 0.0
            gesture_progress = 0

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                    detection_confidence = handedness.classification[0].score

                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    if hand_sign_id == 2:
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    pre_processed_point_history_list = []
                    if point_history:
                        pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                        gesture_progress = int((len(point_history) / history_length) * 100)

                    finger_gesture_id = 0
                    if len(pre_processed_point_history_list) == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()

                    current_hand_sign = keypoint_classifier_labels[hand_sign_id]
                    current_finger_gesture = point_history_classifier_labels[most_common_fg_id[0][0]]
                    current_handedness = handedness.classification[0].label

                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_point_history(debug_image, point_history)

                    if current_hand_sign == current_streak_sign and current_hand_sign != "N/A":
                        sign_streak_count += 1
                    else:
                        current_streak_sign = current_hand_sign
                        sign_streak_count = 0

                    if current_finger_gesture != "None" and current_finger_gesture != "--" and current_finger_gesture == current_streak_gesture:
                        gesture_streak_count += 1
                    else:
                        current_streak_gesture = current_finger_gesture
                        gesture_streak_count = 0

            else:
                point_history.append([0, 0])
                sign_streak_count = 0
                gesture_streak_count = 0
                current_streak_sign = ""
                current_streak_gesture = ""

            self.change_pixmap_signal.emit(debug_image)
            gesture_data = {
                "fps": fps,
                "hand": current_handedness,
                "sign": current_hand_sign,
                "gesture": "None" if current_finger_gesture == "None" else current_finger_gesture,
                "confidence": int(detection_confidence * 100),
                "sign_streak": sign_streak_count if sign_streak_count >= MIN_STREAK_LENGTH else 0,
                "gesture_streak": gesture_streak_count if gesture_streak_count >= MIN_STREAK_LENGTH else 0,
                "gesture_progress": gesture_progress,
            }
            self.update_data_signal.emit(gesture_data)
            # No time.sleep() here, let the loop run as fast as possible
            # time.sleep(1 / 60) # Removed

        cap.release()
        print("Gesture recognition thread stopped.")

    def stop(self):
        self._running = False
        self.wait()


# --- This class creates the main application window and GUI using PySide6 ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Hand Gesture Recognition")
        self.setStyleSheet(self.get_stylesheet())
        self.setMinimumSize(1280, 720)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)

        title_label = QLabel("Real-time Hand Gesture Recognition")
        title_label.setObjectName("titleLabel")
        left_layout.addWidget(title_label)

        self.video_label = QLabel(self)
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        left_layout.addWidget(self.video_label)
        left_layout.addStretch()

        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        panel_title = QLabel("Live Analysis & Advanced Metrics")
        panel_title.setObjectName("panelTitle")
        right_layout.addWidget(panel_title, alignment=Qt.AlignmentFlag.AlignTop)

        self.labels = {}
        data_grid = QGridLayout()
        data_grid.setSpacing(15)

        self.labels['fps'] = self.create_data_widget("FPS", "--")
        self.labels['hand'] = self.create_data_widget("Detected Hand", "N/A")
        data_grid.addWidget(self.labels['fps']['widget'], 0, 0)
        data_grid.addWidget(self.labels['hand']['widget'], 0, 1)

        self.labels['confidence'] = self.create_data_widget("Detection Confidence", "--%", is_progress=True)
        data_grid.addWidget(self.labels['confidence']['widget'], 1, 0, 1, 2)

        right_layout.addLayout(data_grid)

        self.labels['sign'] = self.create_highlight_widget("Recognized Sign", "--")
        self.labels['gesture'] = self.create_highlight_widget("Dynamic Gesture", "--")
        right_layout.addWidget(self.labels['sign']['widget'])
        right_layout.addWidget(self.labels['gesture']['widget'])

        streak_grid = QGridLayout()
        streak_grid.setSpacing(15)
        self.labels['sign_streak'] = self.create_data_widget("Sign Streak", "0")
        self.labels['gesture_streak'] = self.create_data_widget("Gesture Streak", "0")
        streak_grid.addWidget(self.labels['sign_streak']['widget'], 0, 0)
        streak_grid.addWidget(self.labels['gesture_streak']['widget'], 0, 1)
        right_layout.addLayout(streak_grid)

        self.labels['gesture_progress'] = self.create_data_widget("Gesture Progress", "0%", is_progress=True)
        right_layout.addWidget(self.labels['gesture_progress']['widget'])

        right_layout.addStretch()

        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)

        self.thread = GestureThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_data_signal.connect(self.update_data)
        self.thread.start()

    def create_data_widget(self, label_text, value_text, is_progress=False):
        widget = QWidget()
        widget.setObjectName("dataWidget")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        label = QLabel(label_text)
        label.setObjectName("dataLabel")
        layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)

        if is_progress:
            progress_bar = QProgressBar(self)
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setTextVisible(True)
            progress_bar.setObjectName("dataProgressBar")
            layout.addWidget(progress_bar)
            return {"widget": widget, "progress_bar": progress_bar}
        else:
            value = QLabel(value_text)
            value.setObjectName("dataValue")
            layout.addWidget(value, alignment=Qt.AlignmentFlag.AlignCenter)
            return {"widget": widget, "value": value}

    def create_highlight_widget(self, label_text, value_text):
        widget = QWidget()
        widget.setObjectName("highlightWidget")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(8)

        label = QLabel(label_text)
        label.setObjectName("highlightLabel")
        layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)

        value = QLabel(value_text)
        value.setObjectName("highlightValue")
        value.setWordWrap(True)
        value.setMinimumHeight(50)
        layout.addWidget(value, alignment=Qt.AlignmentFlag.AlignCenter)
        return {"widget": widget, "value": value}

    @Slot(np.ndarray)
    def update_image(self, cv_img):
        if cv_img is None:
            self.video_label.setText("No Camera Feed")
            return

        label_size = self.video_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            return

        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = QPixmap.fromImage(convert_to_Qt_format)

        scaled_p = p.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_p)

    @Slot(dict)
    def update_data(self, data):
        if "error" in data:
            self.video_label.setText(data["error"])
            self.video_label.setStyleSheet("color: #e74c3c; font-size: 20px; font-weight: bold;")
            return

        self.labels['fps']['value'].setText(str(data.get('fps', '--')))
        self.labels['hand']['value'].setText(data.get('hand', 'N/A'))

        self.labels['confidence']['progress_bar'].setValue(data.get('confidence', 0))
        self.labels['confidence']['progress_bar'].setFormat(f"{data.get('confidence', 0)}%")

        sign_text = data.get('sign', '--')
        gesture_text = data.get('gesture', '--')

        if gesture_text != 'None' and gesture_text != '--':
            self.labels['sign']['value'].setText(sign_text)
            self.labels['gesture']['value'].setText(gesture_text)
            self.labels['sign']['value'].setStyleSheet("color: #a8dadc;")
            self.labels['gesture']['value'].setStyleSheet("color: #3498DB;")
        else:
            self.labels['sign']['value'].setText(sign_text)
            self.labels['gesture']['value'].setText('--')
            self.labels['sign']['value'].setStyleSheet("color: #3498DB;")
            self.labels['gesture']['value'].setStyleSheet("color: #a8dadc;")

        self.labels['sign_streak']['value'].setText(str(data.get('sign_streak', 0)))
        self.labels['gesture_streak']['value'].setText(str(data.get('gesture_streak', 0)))

        self.labels['gesture_progress']['progress_bar'].setValue(data.get('gesture_progress', 0))
        self.labels['gesture_progress']['progress_bar'].setFormat(f"{data.get('gesture_progress', 0)}%")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = QPixmap.fromImage(convert_to_Qt_format)
        return p

    def closeEvent(self, event):
        print("Closing application...")
        self.thread.stop()
        event.accept()

    def get_stylesheet(self):
        return """
            QMainWindow {
                background-color: #1a1a2e;
                color: #e0e0e0;
                font-family: 'Poppins', sans-serif;
            }
            #titleLabel {
                font-size: 32px;
                font-weight: 700;
                color: #e0e0e0;
                padding-bottom: 10px;
                border-bottom: 2px solid #0f3460;
                margin-bottom: 15px;
            }
            #videoLabel {
                background-color: #000000;
                border: 3px solid #0f3460;
                border-radius: 20px;
                qproperty-alignment: AlignCenter;
            }
            #panelTitle {
                font-size: 26px;
                font-weight: 600;
                color: #e94560;
                border-bottom: 2px solid #e94560;
                padding-bottom: 8px;
                margin-bottom: 15px;
            }
            #dataWidget, #highlightWidget {
                background-color: #2b2d42;
                border-radius: 12px;
                padding: 12px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            }
            #dataLabel, #highlightLabel {
                font-size: 15px;
                color: #a8dadc;
                font-weight: 500;
            }
            #dataValue {
                font-size: 24px;
                font-weight: 700;
                color: #ffffff;
            }
            #highlightValue {
                font-size: 40px;
                font-weight: 800;
                min-height: 60px;
                color: #3498DB;
                qproperty-alignment: AlignCenter;
            }
            QProgressBar {
                border: 2px solid #0f3460;
                border-radius: 8px;
                text-align: center;
                color: #ffffff;
                background-color: #4a4e69;
            }
            QProgressBar::chunk {
                background-color: #e94560;
                border-radius: 6px;
            }
            #dataProgressBar {
                 height: 25px;
            }
        """


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=int, default=0.5)
    return parser.parse_args()


def calc_landmark_list(image, landmarks):
    h, w, _ = image.shape
    return [[min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)] for lm in landmarks.landmark]


def pre_process_landmark(landmark_list):
    temp_list = copy.deepcopy(landmark_list)
    if not temp_list: return []
    base_x, base_y = temp_list[0][0], temp_list[0][1]
    for i in range(len(temp_list)):
        temp_list[i][0] -= base_x
        temp_list[i][1] -= base_y
    flat_list = list(itertools.chain.from_iterable(temp_list))
    max_val = max(map(abs, flat_list)) if flat_list else 1
    return [n / max_val for n in flat_list] if max_val != 0 else [0.0] * len(flat_list)


def pre_process_point_history(image, point_history):
    h, w, _ = image.shape
    temp_list = copy.deepcopy(point_history)
    if not temp_list: return []
    base_x, base_y = temp_list[0][0], temp_list[0][1]
    for i in range(len(temp_list)):
        temp_list[i][0] = (temp_list[i][0] - base_x) / w
        temp_list[i][1] = (temp_list[i][1] - base_y) / h
    return list(itertools.chain.from_iterable(temp_list))


def draw_landmarks(image, landmark_point):
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                   (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
    for start, end in connections:
        cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (200, 200, 200), 2)
    for i, p in enumerate(landmark_point):
        color = (0, 255, 255) if i in [4, 8, 12, 16, 20] else (100, 255, 100)
        size = 8 if i in [4, 8, 12, 16, 20] else 5
        cv.circle(image, tuple(p), size, color, -1)
    return image


def draw_point_history(image, point_history):
    for i, p in enumerate(point_history):
        if p[0] != 0 and p[1] != 0:
            cv.circle(image, tuple(p), 2 + int(i / 2), (152, 251, 152), -1)
    return image


if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        print(
            f"Warning: Could not add custom font. Falling back to default. Error: {e}")
        pass

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())