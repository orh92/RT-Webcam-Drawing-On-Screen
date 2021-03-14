import cv2
import numpy as np
from getContours import getContours
class findPoints:
    def findNewPoints(img, myColors, myColorValues):
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        count = 0
        newPoints = []
        for color in myColors:
            lower = np.array(color[0:3])
            upper = np.array(color[3:6])
            mask = cv2.inRange(imgHSV, lower, upper)
            x, y = getContours.getItemContours(mask)
            cv2.circle(img, (x, y), 10, myColorValues[count], cv2.FILLED)
            if x != 0 and y != 0:
                newPoints.append([x, y, count])
            count += 1
        return newPoints