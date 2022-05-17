# I Spy by Edward Ng
# User picks an object and says its colour. The computer searches all objects of that colour in view
# and guesses which object was chosen.
# 5/17/2022

import cv2
import numpy as np

def map_colours(colour, frame):
    '''converts colour string to cv colour range.'''
    if colour == "red":
        return cv2.inRange(frame, (0, 0, 50), (50, 50,255))
    if colour == "green":
        return cv2.inRange(frame, (0,50,0), (50, 255, 50))
    if colour == "blue":
        return cv2.inRange(frame, (50,0,0), (255, 50, 50))

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph_coco.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    print("Press q to quit.")

    while True:
        # user inputs colour
        colour = input("I spy something that is...")

        while True:
            _, frame = cap.read()

            # break from loop
            if cv2.waitKey(1) == ord('q'):
                break
            
            # convert to hsv colorspace
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # find the colours specified by user
            mask = map_colours(colour, frame)
            #define kernel size  
            kernel = np.ones((7,7),np.uint8)
            # Remove unnecessary noise from mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Segment only the detected region
            segmented_img = cv2.bitwise_and(frame, frame, mask=mask)

            # Find contours from the mask
            contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

            output = cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

            classIds, confs, bbox = net.detect(frame, confThreshold=0.5)

            if len(classIds) != 0:
                for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(output, box, color=(0,255,0), thickness=2)
                    cv2.putText(output, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # output camera with contour lines
            cv2.imshow("I Spy", output)

        # release camera
        cap.release()
        cv2.destroyAllWindows()

        # play condition
        play = input("Keep playing?")
        if play == "no":
            break
