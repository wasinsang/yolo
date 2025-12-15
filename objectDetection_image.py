from ultralytics import YOLO
import cv2

# โหลดโมเดล YOLO
model = YOLO("yolo11n.pt")

# อ่านภาพจากไฟล์
img = cv2.imread("Road.jpg")
if img is None:
    raise FileNotFoundError("ไม่พบไฟล์ภาพ Road.jpg")

# รัน YOLO
results = model(img)

# Dictionary สำหรับนับจำนวนคลาส
class_count = {}

# วาดผลลัพธ์บนภาพ
for r in results:
    for box in r.boxes:
        # พิกัดกล่อง
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # label class
        cls = int(box.cls[0])
        label = model.names[cls]

        # นับจำนวนคลาส
        class_count[label] = class_count.get(label, 0) + 1

        # วาดกรอบและ label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

# ------------------------------
# แสดงจำนวนคลาสมุมซ้ายบนของภาพ
# ------------------------------
y_text = 30
for label, count in class_count.items():
    text = f"{label}: {count}"
    cv2.putText(img, text, (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 255), 2)
    y_text += 30

# แสดงผล
cv2.imshow("YOLO Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
