from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

# ใส่ path วิดีโอของคุณ
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    class_count = {}

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            cls = int(box.cls[0])
            label = model.names[cls]

            # นับจำนวน
            class_count[label] = class_count.get(label, 0) + 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

    # แสดงจำนวนด้านบนซ้าย
    y_text = 30
    for label, count in sorted(class_count.items(), key=lambda x: x[1], reverse=True):
        cv2.putText(frame, f"{label}: {count}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)
        y_text += 35

    cv2.imshow("YOLO Video", frame)

    # กด q เพื่อออก
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
