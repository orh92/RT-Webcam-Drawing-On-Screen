import cv2
import numpy as np
import PIL.ImageGrab
from stackImages import stackImages
from getContours import getContours
from findPoints import findPoints

class drawShapes:
    def takeScreenshot():
        im = PIL.ImageGrab.grab()
        im.save(r"resources/screen.jpg")

    def findIfExtist(point, array):
        for p in array:
            if point == p:
                return True
            else:
                return False

    def drawOnCanvas(myPoints, myColorValues,imgContour):
        for point in myPoints:
            cv2.circle(imgContour, (point[0], point[1]), 4, myColorValues[point[2]], cv2.FILLED)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 150)

    #63 115 0 110 255 165
    myColors = [[63 ,115, 0 ,110 ,255 ,165]]
    myColorValues = [[255, 0, 0]]

    myPoints = []  # points array to draw(x,y,colorId)
    myshapes = []
    shapesSize = len(myshapes)
    counter = 0
    while True:
        success, img = cap.read()
        shapesImg = cv2.imread("resources/screen.jpg")
        imgBlank = np.zeros_like(img)
        imgContour=img.copy()
        if shapesImg is None:
            shapesImg = imgBlank.copy()
        shapesImg = cv2.cvtColor(shapesImg, cv2.COLOR_BGR2GRAY)
        shapesImg = shapesImg[520:950, 385:950]
        shapesImg = cv2.GaussianBlur(shapesImg, (5, 5), 1)
        img = cv2.flip(img, 1)
        imgContour = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 1)
        canny = cv2.Canny(blur, 50, 50)

        newPoints = findPoints.findNewPoints(img, myColors, myColorValues)
        getContours.getItemContours(shapesImg)
        if len(newPoints) != 0:
            for newP in newPoints:
                myPoints.append(newP)
        if len(myPoints) != 0:
            drawOnCanvas(myPoints, myColorValues,imgContour)

        if len(myshapes) != 0:
            for shape in myshapes:
                cv2.rectangle(shapesImg, (shape[0][0], shape[0][1]), (shape[1][0], shape[1][1]), (255, 0, 0), 2)
            if shapesSize < len(myshapes):
                myPoints = []
                shapesSize += 1
                print("shapes number: " + str(shapesSize))

        stack = stackImages.stackImages(0.7, [[img, canny], [imgContour, shapesImg]])
        cv2.imshow("Screen", stack)
        cv2.moveWindow("Screen", 300, 50)
        if counter % 10 == 0:
            takeScreenshot()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
