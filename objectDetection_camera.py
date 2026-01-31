from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    class_count = {}

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            class_count[label] = class_count.get(label, 0) + 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

    y_text = 25
    for label, count in sorted(class_count.items(), key=lambda x: x[1], reverse=True):
        cv2.putText(frame, f"{label}: {count}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)
        y_text += 30

    cv2.imshow("YOLO USB Webcam - Pi 5", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
