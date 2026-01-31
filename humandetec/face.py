import cv2
from ultralytics import YOLO

# ===============================
# Load YOLOv8 Model (Object + Human Detection)
# ===============================
model = YOLO("yolov8n.pt")  # Lightweight + Fast

# ===============================
# Load Face Detection Model (OpenCV DNN)
# ===============================
face_model = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# ===============================
# Start Webcam
# ===============================
cap = cv2.VideoCapture(0)

print("✅ AI Detection Started... Press Q to Exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ===============================
    # YOLO Object Detection
    # ===============================
    results = model(frame)

    human_count = 0

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ✅ Count Humans
            if label == "person":
                human_count += 1
                color = (0, 255, 0)  # Green for Human
            else:
                color = (255, 0, 0)  # Blue for Other Objects

            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label Text
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ===============================
    # Face Detection
    # ===============================
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_model.setInput(blob)
    detections = face_model.forward()

    face_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            face_count += 1

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype("int")

            # Draw Face Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.putText(
                frame,
                "Face",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # ===============================
    # Display Human + Face Count
    # ===============================
    cv2.putText(
        frame,
        f"Humans Count: {human_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Faces Count: {face_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    # Show Output Window
    cv2.imshow("AI Face + Human + Object Detection", frame)

    # Exit Key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===============================
# Cleanup
# ===============================
cap.release()
cv2.destroyAllWindows()
print("✅ Detection Stopped")
