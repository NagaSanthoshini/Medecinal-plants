from ultralytics import YOLO
import cv2
import math
import os
cap = cv2.VideoCapture(0) 

model = YOLO('best (2).pt')

classnames = ['Medicinal-Arive Dantu', 'Medicinal-Basale', 'Medicinal-Neem', 'Medicinal-Rose Apple', 'Medicinal-Sandalwood', 'Poisonous-Calatropis', 'Poisonous-Datura_Stramonium', 'Poisonous-Deadly_Nightshade', 'Poisonous-Hemlock', 'Poisonous-Parthenium']



# Assuming images are stored in a directory
image_dir = 'img'

# List image files in the directory
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('.jpg', '.jpeg', '.png'))]

# Iterate over each image file
for image_file in image_files:
    frame = cv2.imread(image_file)  # Read image from file
    
    if frame is None:
        print(f"Error: Unable to read image {image_file}")
        continue

    frame = cv2.resize(frame, (640, 480))
    
    result = model(frame, stream=True)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            class_index = int(box.cls[0])

            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.putText(frame, f'{classnames[class_index]} {confidence}%', (x1 + 8, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 235, 55), 2)

    cv2.imshow('Plant Detection', frame)
    cv2.waitKey(0)  # Wait for any key press to proceed to the next image

cv2.destroyAllWindows()