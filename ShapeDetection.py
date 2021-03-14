import cv2
import numpy as np
from stackImages import stackImages
from getContours import getContours

class ShapeDetection:
    def getImgContour(self):
        return self.imgContour

    path = 'resources/shapes.jpg'
    img = cv2.imread(path)
    imgContour = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    getContours.getContours(imgCanny,imgContour)

    imgBlank = np.zeros_like(img)
    imgStack = stackImages.stackImages(0.4, ([img, imgGray, imgBlur], [imgCanny, imgContour, imgBlank]))

    cv2.imshow("Stack", imgStack)
    cv2.waitKey(0)