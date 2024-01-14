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



# Annotation variables
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
colorIndex = 0
pointerSize = 5


# Hand Detector
detector = HandDetector(detectionCon=0.8)

# Left hand actions
ANNOTATION = [[0, 1, 0, 0, 0]]
NAVIGATE = [[0, 1, 1, 1, 0]]
ERASE = [[1, 1, 1, 1, 1]]
QUIT = [[0, 0, 0, 1, 1]]
POINTER_CHANGE = [[1, 1, 1, 0, 0]]

# Right hand actions
CHANGE_COLOR = [[0, 1, 1, 0, 0]]
INCREMENT_SIZE = [[0, 0, 1, 1, 0]]
DECREMENT_SIZE = [[0, 0, 0, 1, 1]]
ANNOTATE = [[0, 1, 0, 0, 0]]


def clearAnnotation():
    global annotationStart
    global annotationNumber
    global annotations
    annotationStart = False
    annotationNumber = -1
    annotations = []


def eraseAnnotation():
    global annotationNumber
    global annotations
    global buttonPressed
    if annotations:
        annotations.pop(-1)
        annotationNumber -= 1
        buttonPressed = True


while True:
    # Import Images
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    imgCurrent = cv2.resize(imgCurrent, (1280, 720))

    # Show the pointer size and color on the top left
    cv2.circle(imgCurrent, (10, 10), pointerSize, COLORS[colorIndex], cv2.FILLED)

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold),
             (width, gestureThreshold), (0, 255, 0), 10)

    indexFinger = None
    fingers = None
    fingersRight = None
    for hand in hands:
        if hand['type'] == 'Left':
            # This is actually the right hand and we will be using this to draw

            # LandMark list
            lmList = hand['lmList']

            # Constrain values for easier drawing
            xVal = int(np.interp(lmList[8][0], [
                marginX, width - marginX], [0, width]))
            yVal = int(np.interp(lmList[8][1], [
                marginY, height - marginY], [0, height]))
            indexFinger = xVal, yVal

            fingersRight = detector.fingersUp(hand)

        else:
            # This is actually the left hand and we will be using this to give control gestures
            fingers = detector.fingersUp(hand)

    if indexFinger:
        cv2.circle(imgCurrent, indexFinger, pointerSize, COLORS[colorIndex], cv2.FILLED)

    if (fingers in ANNOTATION):
        if not annotationStart:
            annotationStart = True
            annotationNumber += 1
            annotations.append([])
        annotations[annotationNumber].append(indexFinger)
    else:
        annotationStart = False

    if (fingers in ERASE) and (not buttonPressed):
        eraseAnnotation()

    if (fingers in POINTER_CHANGE) and (not buttonPressed):
        if fingersRight in CHANGE_COLOR:
            buttonPressed = True
            colorIndex = (colorIndex + 1) % len(COLORS)

        if fingersRight in INCREMENT_SIZE:
            buttonPressed = True
            pointerSize += 1

        if fingersRight in DECREMENT_SIZE:
            buttonPressed = True
            pointerSize -= 1

    if fingers in QUIT:
        break

    for annotation in annotations:
        for i in range(1, len(annotation)):
            cv2.line(imgCurrent, annotation[i - 1],
                     annotation[i], COLORS[colorIndex], pointerSize)

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
