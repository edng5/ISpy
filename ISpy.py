# I Spy by Edward Ng
# User picks an object and says its colour. The computer searches all objects of that colour in view
# and guesses which object was chosen.
# 5/17/2022

import cv2
import numpy as np
from ColorLabeler import ColorLabeler
import imutils

def map_colours(colour, frame):
    '''converts colour string to cv colour range.'''
    if colour == "red":
        return cv2.inRange(frame, (0, 0, 50), (50, 50,255))
    if colour == "green":
        return cv2.inRange(frame, (0,50,0), (50, 255, 50))
    if colour == "blue":
        return cv2.inRange(frame, (50,0,0), (255, 50, 50))

if __name__ == "__main__":
    # set game state
    game_state = 1

    # create video capture
    cap = cv2.VideoCapture(0)

    # initialize class names using COCO dataset
    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # set config path and weight path
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph_coco.pb'

    # initialize Detection Model
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    print("Press q to quit.")

    while True:
        # objects to guess from
        obj_list = []

        # user inputs colour
        colour = input("I spy something that is...")

        # initialize ColorLabeler
        cl = ColorLabeler()

        while game_state:
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

            # find contours in the frame
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # run detect on frame
            classIds, confs, bbox = net.detect(frame, confThreshold=0.5)

            for contour in cnts:
                if len(classIds) != 0:
                    for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                        # run ColorLabeler on frame
                        obj_colour = cl.label(frame, contour)
                        # if the colour in the frame is the same as the colour chosen by the user
                        if  obj_colour == colour:  
                            # draw and label the frame
                            cv2.rectangle(frame, box, color=(0,255,0), thickness=2)
                            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                            cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                            # add object to object list
                            if classNames[classId-1] not in obj_list:
                                obj_list.append(classNames[classId-1])

            # # guesses objects in the list
            # for obj in obj_list:
            #     guess = input("Is "+obj+" your object?")
            #     # ends game if object was guessed correctly
            #     if guess == "yes":
            #         print("I have guessed your object!")
            #         game_state = 0
            #         break
            #     else:
            #         continue

            # output camera
            cv2.imshow("I Spy", frame)

        # release camera
        cap.release()
        cv2.destroyAllWindows()

        # play condition
        play = input("Keep playing?")
        if play == "no":
            break
