import cv2
from ultralytics import YOLO
import math

# Define class names (ensure they match your model's training)
classname = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'none', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'raccoon', 'dog', 'hair drier', 'toothbrush', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush', 'tie', 'suitcase', 'donut', 'cake', 'cup', 'banana', 'apple', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


# Load YOLO model
model = YOLO('yolo11l.pt')
mask = cv2.imread(r"C:\Users\rezak\OneDrive\Desktop\ML project\Untitled design (2).jpg")
# Initialize webcam
cap = cv2.VideoCapture(r"C:\Users\rezak\OneDrive\Desktop\ML project\27260-362770008_tiny.mp4")
frame_width = int(cap.get(3))  # Frame width
frame_height = int(cap.get(4))  # Frame height
mask = cv2.resize(mask, (frame_width, frame_height))
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    masked = cv2.bitwise_and(frame, mask)
    if not ret:
        print("Error: Unable to capture frame")
        break

    # Perform prediction
    results = model.predict(masked , stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class label
            cls = int(box.cls[0])
            label = classname[cls]  # Get the class name

            # Only detect "person"
            if label in 'car':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 3)
                cv2.putText(frame, f'{label}', (x1, max(35, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                print('NO')

    # Display the frame
    cv2.imshow('For Test', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
