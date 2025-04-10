from numpy.random import randint as r
from cvzone.HandTrackingModule import HandDetector
import cv2

def get_non_overlapping_boxes():
    w, h = 180, 180
    w1, h1 = 180, 180

    while True:
        # Random position for box 1
        cx, cy = r(110, 200), r(110, 200)
        # Random position for box 2
        cx1, cy1 = r(1, 100), r(1, 100)

        # Create box corners
        x1_min, y1_min = cx - w // 2, cy - h // 2
        x1_max, y1_max = cx + w // 2, cy + h // 2

        x2_min, y2_min = cx1 - w1 // 2, cy1 - h1 // 2
        x2_max, y2_max = cx1 + w1 // 2, cy1 + h1 // 2

        # Check for NO overlap
        if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
            break  # They don't touch, good to go

    return cx, cy, w, h, cx1, cy1, w1, h1


# Example usage
cx, cy, w, h, cx1, cy1, w1, h1 = get_non_overlapping_boxes()

color = (0, 225, 0)
color1 = (0, 225, 0)
cap = cv2.VideoCapture(0)
cap.set(3, 1400)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.9)
while True:
    suc, img = cap.read()
    cv2.flip(img, 1)
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        lmlist = hand["lmList"]
        l, _, _ = detector.findDistance((lmlist[8][:2]), (lmlist[12][:2]), img=None, scale=10)
        if l < 40:
            corsor = lmlist[8]

            if cx - w // 2 < corsor[0] < cx + w // 2 and cy - h // 2 < corsor[1] < cy + h // 2:
                color = (255, 0, 0)
                cx, cy, _ = corsor

            if cx1 - w1 // 2 < corsor[0] < cx1 + w1 // 2 and cy1 - h1 // 2 < corsor[1] < cy1 + h1 // 2:
                cx1, cy1, _ = corsor
                color1 = (255, 0, 0)

        else:
            color = (255, 255, 255)


    img = cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), color, cv2.FILLED)
    img = cv2.rectangle(img, (cx1 - w1 // 2, cy1 - h1 // 2), (cx1 + w1 // 2, cy1 + h1 // 2), color1, cv2.FILLED)

    cv2.imshow("test", img)

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
