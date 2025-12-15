import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Tip IDs for fingers (thumb is special case)
fingerTipsIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]  # Landmark list
        bbox = hand["bbox"]      # Bounding box info
        fingers = detector.fingersUp(hand)

        # Count fingers
        totalFingers = fingers.count(1)

        # Show count on image
        cv2.putText(img, f'Fingers: {totalFingers}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)

        # Draw points and lines on fingertips
        for i in fingerTipsIds:
            x, y = lmList[i][:2]
            cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)

        # Draw lines between each fingertip (optional)
        for i in range(1, len(fingerTipsIds)):
            x1, y1 = lmList[fingerTipsIds[i - 1]][:2]
            x2, y2 = lmList[fingerTipsIds[i]][:2]
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)

    cv2.imshow("Finger Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
