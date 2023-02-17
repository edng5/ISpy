# I Spy by Edward Ng
# User picks an object and says its colour. The computer searches all objects of that colour in view
# and guesses which object was chosen.
# 2/16/2023

import cv2
import numpy as np
import threading

# GLOBAL variables
# objects to guess from
obj_list = []

class myThread (threading.Thread):
    '''Simple thread class.'''
    def __init__(self, threadID, name, counter):
        '''initialize thread'''
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        '''Run thread.'''
        # Get lock to synchronize threads
        threadLock.acquire()

        guess_object()

        # Free lock to release next thread
        threadLock.release()

def guess_object():
    '''Guesses objects in global object list.
    :Returns: None
    '''
    run = 1
    while run:
        for obj in obj_list:
            guess = input("Is "+obj+" your object?")
            # ends game if object was guessed correctly
            if guess == "yes":
                print("I have guessed your object!")
                run = 0
                break
            else:
                continue

def map_colours(colour, frame):
    '''Converts colour string to cv colour range.
    :param colour: string of a colour
    :param frame: the image

    :Returns: cv2 colour range
    '''
    if colour == "red":
        return cv2.inRange(frame, (160,20,70), (190,255,255))
    if colour == "green":
        return cv2.inRange(frame, (50, 20, 20), (100, 255, 255))
    if colour == "blue":
        return cv2.inRange(frame, (101,50,38), (110,255,255))

if __name__ == "__main__":
    # set game state
    game_state = 1
    running = 1
    flag = 0

    # initialize thread
    threadLock = threading.Lock()
    thread = myThread(1, "Thread-1", 1)

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

    while running:
        # user inputs colour
        colour = input("I spy something that is...")

        while game_state:
            _, frame = cap.read()

            # break from loop
            if cv2.waitKey(1) == ord('q'):
                break

            # convert to hsv colorspace
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # find the colors within the boundaries
            mask = map_colours(colour, hsv)

            #define kernel size  
            kernel = np.ones((7,7),np.uint8)

            # Remove unnecessary noise from mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Segment only the detected region
            segmented_img = cv2.bitwise_and(frame, frame, mask=mask)

            # Find contours from the mask
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            try:
                for contour in contours:
                    x,y,w,h = cv2.boundingRect(contour)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cropped_image = frame[y:y+h, x:x+w] # add padding

                    # run detect on cropped portion
                    classIds, confs, bbox = net.detect(cropped_image, confThreshold=0.5)
                    for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                        cv2.rectangle(frame, (x,y),(x+w,y+h), color=(0,255,0), thickness=2)
                        cv2.putText(frame, classNames[classId - 1].upper(), (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        if classNames[classId-1] not in obj_list:
                                obj_list.append(classNames[classId-1])
            except:
                pass
            

            # start guessing thread
            if flag == 0:
                thread.start()
                flag = 1

            # output camera
            cv2.imshow("I Spy", frame)

            # terminate loop because object was guessed
            if not thread.isAlive():
                game_state = 0

        # release camera
        cap.release()
        cv2.destroyAllWindows()

        # play condition
        running = 0
