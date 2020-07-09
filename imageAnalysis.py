import cv2 as cv
import numpy as np
import random as rng

kernel = (5, 5)
path = 'resources/lot.png'
img = cv.imread(path)

if img is None:
    print('Could not open or find the image: ', args.input)
    exit(0)


def getContours(img):
    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    imgContour = cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    return imgContour


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None,
                                               scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

# resizing the image
scale_percent = 30
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

imgCopied = img.copy()
imgBlank = np.ones_like(img)  # all black image
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
grayBlur = cv.GaussianBlur(imgGray, kernel, 0)
grayCanny = cv.Canny(imgGray, 30, 200)
grayCannyCopied = grayCanny.copy()
grayBlurCanny = cv.Canny(grayBlur, 30, 200)
grayDialiate = cv.dilate(grayCanny, kernel, iterations=1)
grayBlurDialiate = cv.dilate(grayCanny, kernel, iterations=1)

# attempting the first contour tutorial i found (link: https://www.youtube.com/watch?v=FbR9Xr0TVdY)

ret, thresh = cv.threshold(grayBlurCanny, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

cv.drawContours(imgCopied, contours, -1, (0, 255, 0), 3)
# Approximate contours to polygons + get bounding rects and circles
contours_poly = [None] * len(contours)
boundRect = [None] * len(contours)
centers = [None] * len(contours)
radius = [None] * len(contours)
for i, c in enumerate(contours):
    contours_poly[i] = cv.approxPolyDP(c, 3, True)
    boundRect[i] = cv.boundingRect(contours_poly[i])
    centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
drawing = np.zeros((grayCanny.shape[0], grayCanny.shape[1], 3), dtype=np.uint8)

# Draw polygonal contour + bonding rects + circles
for i in range(len(contours)):
    color  = (255,0,0)
    #color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv.drawContours(drawing, contours_poly, i, color)
    cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                 (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
    #cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

while True:
    imgStack = stackImages(.4, [img, imgGray, grayBlur])
    cv.imshow('stack', imgStack)
    imgStack = stackImages(.4, ([imgGray, imgBlank, grayBlur],
                                [grayCanny, imgCopied, grayBlurCanny],
                                [grayDialiate, drawing, grayBlurDialiate]))
    cv.imshow("Original Stack", imgStack)

    if cv.waitKey(1) and 0xFF == ('q'):
        break
