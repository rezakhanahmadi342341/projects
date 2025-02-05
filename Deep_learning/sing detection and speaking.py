import cv2
from ultralytics import YOLO
import pyttsx3
import time

s = pyttsx3.init()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load YOLO model
model = YOLO(r"C:\Users\rezak\OneDrive\Desktop\ML project\last.pt")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection
    results = model(frame)
    
    # Display results
    results = results[0]
    annotated_frame = results.plot()   
    

    for box in results.boxes:
        cls_index = int(box.cls.cpu().numpy()[0])
        cls = model.names[cls_index]
        if cls == 'Me':
            s.say("this is Me")
            s.runAndWait()
            time.sleep(1)
        else:
            pass
        if cls == 'You':
            s.say("You")
            s.runAndWait()
            time.sleep(1)

        if cls == 'Like':
            s.say("I like it.")
            s.runAndWait()
            time.sleep(1)

        if cls == 'sorry':
            s.say("I am really sorry.")
            s.runAndWait()
            time.sleep(1)

        if cls == 'Nice':
            s.say("this is Nice")
            s.runAndWait()
            time.sleep(1)
        
    cv2.imshow('YOLO Detection', annotated_frame)

# Release resources
cap.release()
cv2.destroyAllWindows()

