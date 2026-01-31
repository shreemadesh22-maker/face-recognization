# AI Face, Human & Object Detection using YOLOv8 and OpenCV

This project performs **real-time face detection, human counting, and object detection** using a webcam.  
It combines **YOLOv8 (Ultralytics)** for object & human detection and **OpenCV DNN** for face detection.

---

## 🚀 Features

- 🔍 Real-time object detection using YOLOv8
- 🧍 Human (person) detection and counting
- 🙂 Face detection using OpenCV Deep Neural Network
- 📷 Live webcam feed processing
- 📊 Displays human count and face count on screen

---

## 🛠️ Technologies Used

- Python 3.x  
- OpenCV  
- YOLOv8 (Ultralytics)  
- PyTorch  

---

## 📁 Project Structure

.
├── face.py # Main Python script
├── req.txt # Required Python libraries
├── yolov8n.pt # YOLOv8 pre-trained model
├── deploy.prototxt # Face detection model config
├── res10_300x300_ssd_iter_140000.caffemodel # Face detection weights
└── README.md


---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Install Dependencies
pip install -r req.txt
▶️ How to Run
Make sure your webcam is connected.

python face.py
Press Q to exit the application.

📌 Output
Green bounding boxes → Humans

Blue bounding boxes → Other detected objects

Red bounding boxes → Faces

Live display of:

Human Count

Face Count

📷 Sample Use Cases
Smart surveillance systems

Crowd monitoring

Women safety analysis

AI-based security applications

Computer vision learning projects

📦 Models Used
YOLOv8n – Lightweight and fast object detection model

OpenCV SSD Face Detector – Pre-trained Caffe model

🧠 Future Improvements
Gender and age detection

Face recognition (known vs unknown)

Alert system for crowd threshold

Save detection logs

🙌 Author
shruthi
B.E Computer Science (Cyber Security)



