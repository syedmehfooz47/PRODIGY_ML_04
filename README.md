# ✨ Advanced Gesture Recognition

> **Real-time AI-powered hand gesture recognition system with modern UI and intelligent feedback**

![Main Demo](https://github.com/syedmehfooz47/Advanced-Gesture-Recognition/blob/master/demo/Demo.gif)

---

## 🚀 Overview

**Advanced Gesture Recognition** is a real-time AI-based system that detects and classifies hand gestures using your webcam. It combines **MediaPipe**, **OpenCV**, and custom-trained models with a modern, dynamic UI. Ideal for accessibility solutions, gesture-controlled interfaces, or next-gen smart apps.

---

## 🌟 Key Features & Capabilities

- 🖐️ **Dual-Hand Real-Time Tracking**  
  Tracks both left and right hands using MediaPipe’s high-fidelity landmark detection.

- ✋ **Static Gesture Classification**  
  Identifies gestures like:
  - Open
  - Close
  - Pointer

- 🔁 **Dynamic Gesture Recognition (Finger Trails)**  
  Recognizes movement-based gestures like:
  - Stop
  - Move
  - Clockwise
  - Anticlockwise  
  > Example:  
  > ![Clockwise Gesture](https://github.com/syedmehfooz47/Advanced-Gesture-Recognition/blob/master/demo/Clockwise.gif)

- 🧠 **Custom Gesture Classifiers**  
  Built using TensorFlow Lite & trained via keypoint/finger history logging.

- 🖥️ **Modern UI (Web & PySide6 Versions)**  
  Displays:
  - Live webcam feed with overlays  
  - FPS and confidence indicators  
  - Detected gestures and hand (Left/Right)  
  - Gesture streak counter and progress bar  

  > Snapshots:  
  > <img src="https://github.com/syedmehfooz47/Advanced-Gesture-Recognition/blob/master/demo/Snapshot.PNG" width="45%"/> <img src="https://github.com/syedmehfooz47/Advanced-Gesture-Recognition/blob/master/demo/Snapshot_1.PNG" width="45%"/>

- 🔄 **Extensible & Modular Design**  
  Easily add new gestures, retrain models, or integrate with other systems.

- ⚙️ **Optimized for Performance**  
  High FPS even on mid-range systems; GPU-accelerated when enabled.

---

## 🎬 More Gesture Demos

> Smooth real-time operation, including detection overlays, streak count, and gesture feedback:

<div align="center">
  <img src="https://github.com/syedmehfooz47/Advanced-Gesture-Recognition/blob/master/demo/Demo-1.gif" width="80%"/>
</div>

---

## 📸 UI Snapshots

> Additional examples of the app running in real-world environments:

<div align="center">
  <img src="https://github.com/syedmehfooz47/Advanced-Gesture-Recognition/blob/master/demo/Snapshot_2.PNG" width="45%"/>
  <img src="https://github.com/syedmehfooz47/Advanced-Gesture-Recognition/blob/master/demo/Snapshot_3.PNG" width="45%"/>
</div>

---

## 🛠️ Technical Stack

**🧠 Core AI/ML**  
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)  
- TensorFlow / TensorFlow Lite  
- Scikit-learn

**🖼️ Computer Vision**  
- [OpenCV](https://opencv.org/)  
- Pillow (for image processing)

**🖥️ GUI Options**  
- PySide6 – for a modern desktop UI  
- Flask – for web-based access

**📦 Utilities**  
- NumPy, itertools, collections, csv  
- Protobuf (MediaPipe), Matplotlib (for visualization)

---

## 🎯 Supported Gestures

### Static Hand Signs  
- 🖐️ Open  
- ✊ Close  
- 👉 Pointer

### Dynamic Finger Gestures  
- ✋ Stop  
- 🧭 Move  
- 🔃 Clockwise  
- 🔄 Anticlockwise

---

## 🧠 Train Your Own Gestures

Use the included Jupyter Notebooks to log data and train your own gestures.

### Static Gesture Training (`keypoint_classification.ipynb`)
1. Press `k` to enable keypoint logging.
2. Press keys `0-9` to log samples.
3. Train using the notebook and export `keypoint_classifier.tflite`.

### Dynamic Gesture Training (`point_history_classification.ipynb`)
1. Press `h` to log fingertip trajectory.
2. Press keys `0-9` for labeling.
3. Train and export `point_history_classifier.tflite`.

---

## 🚀 Getting Started

1. **Clone the Repository**

```bash
git clone https://github.com/syedmehfooz47/Advanced-Gesture-Recognition.git
cd Advanced-Gesture-Recognition
````

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the App**

```bash
python app.py
```

✅ The app will launch automatically in your browser or GUI window.

---

## 💻 Author

Built with ❤️ by
[**Syed Mehfooz C S**](https://github.com/syedmehfooz47)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE)

---

## 🌟 Show Support

If you like this project, **give it a star ⭐** and share it with others!

---
