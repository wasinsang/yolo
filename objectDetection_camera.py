from ultralytics import YOLO
import cv2

# โหลดโมเดล YOLO ที่เทรนแล้ว (ฟรี 80 classes) สามารถเปลี่ยนชื่อเป็นไฟล์ .pt ของตัวเองได้
model = YOLO("yolo11n.pt")

# เปิดกล้องเว็บแคม (0 = กล้องหลักของโน้ตบุ๊ค)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # รัน YOLO ตรวจภาพจากกล้อง
    results = model(frame, stream=True)

    # วาดผลลัพธ์ลงบนภาพ
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # พิกัดกรอบ
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # ชื่อคลาส
            cls = int(box.cls[0])
            label = model.names[cls]

            # วาดกรอบ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

    # แสดงผล
    cv2.imshow("YOLO Webcam", frame)

    # กด q เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
