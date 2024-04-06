from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0) 

model = YOLO('best (2).pt')

classnames = ['Medicinal-Arive Dantu', 'Medicinal-Basale', 'Medicinal-Neem', 'Medicinal-Rose Apple', 'Medicinal-Sandalwood', 'Poisonous-Calatropis', 'Poisonous-Datura_Stramonium', 'Poisonous-Deadly_Nightshade', 'Poisonous-Hemlock', 'Poisonous-Parthenium']

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    
    result = model(frame, stream=True)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            class_index = int(box.cls[0])

            if confidence > 50 and classnames[class_index] in classnames:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.putText(frame, f'{classnames[class_index]} {confidence}%', (x1 + 8, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Plant  Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()