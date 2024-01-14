import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Variables
width, height = 1280, 720
folderPath = "Presentation/"

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Get the list of presentation images
pathImages = os.listdir(folderPath)

# Variables
imgNumber = 0
ws, hs = 213, 120
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 10
annotations = []
annotationNumber = -1
annotationStart = False
marginX = 150
marginY = 150

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)


def clearAnnotation():
    global annotationStart
    global annotationNumber
    global annotations
    annotationStart = False
    annotationNumber = -1
    annotations = []
    pass


while True:
    # Import Images
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    imgCurrent = cv2.resize(imgCurrent, (1280, 720))

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold),
             (width, gestureThreshold), (0, 255, 0), 10)

    if hands and not buttonPressed:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        # Get the center of hand
        cx, cy = hand['center']

        # LandMark list
        lmList = hand['lmList']

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [
                   marginX, width - marginX], [0, width]))
        yVal = int(np.interp(lmList[8][1], [
                   marginY, height - marginY], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:      # if hand is at the height of the face
            # Gesture 1 - Left
            if fingers == [1, 0, 0, 0, 0]:
                imgNumber = max(0, imgNumber - 1)
                buttonPressed = True
                clearAnnotation()

            # Gesture 2 - Right
            if fingers == [0, 0, 0, 0, 1]:
                imgNumber = min(len(pathImages) - 1, imgNumber + 1)
                buttonPressed = True
                clearAnnotation()

        # Gesture 3 - Show Pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        # Gesture 4 - Annotate
        if fingers == [0, 1, 0, 0, 0]:

            if not annotationStart:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])

            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False

        # Gesture 5 - Erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

    for annotation in annotations:
        for i in range(1, len(annotation)):
            cv2.line(imgCurrent, annotation[i - 1],
                     annotation[i], (0, 0, 200), 12)

    # Button Pressed iterations
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonPressed = False
            buttonCounter = 0

    # Adding webcam image on the slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w-ws:w] = imgSmall

    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
