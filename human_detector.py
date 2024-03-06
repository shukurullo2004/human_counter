import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('yolov9c.pt')

area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

cap = cv2.VideoCapture("/Users/shukurullomeliboyev2004/Desktop/Machine-learnings/real_ai/real/peoplecount1.mp4")

my_file = open("/Users/shukurullomeliboyev2004/Desktop/Machine-learnings/real_ai/real/peoplecounteryolov8/coco.txt", 'r')
data = my_file.read()
class_list = data.split('\n')

count = 0
tracker = Tracker()

people_enter = {}
entering = set()

people_exit = {}
exiting = set()

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1020, 500))  # Change the filename, frame rate, and resolution

while True:
    success, frame = cap.read()
    if not success:
        break
    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)

    a = results[0].boxes.data
    px = pd.DataFrame(a).astype('float')
    bbox_list = []

    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]

        if 'person' in c:
            bbox_list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(bbox_list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox

        # Entering people
        result = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
        if result >= 0:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            people_enter[id] = (x4, y4)

        if id in people_enter:
            result1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
            if result1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 4, (255, 255, 0), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                entering.add(id)

        # Exiting people
        result2 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
        if result2 >= 0:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            people_exit[id] = (x4, y4)

        if id in people_exit:
            result3 = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
            if result3 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 4, (255, 255, 0), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                exiting.add(id)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    ent_count = len(entering)
    exi_count = len(exiting)
    cv2.putText(frame, f"Enter: {str(ent_count)}", (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
    cv2.putText(frame, f"Exit: {str(exi_count)}", (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 1)

    out.write(frame)  # Write the frame to the output video

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
