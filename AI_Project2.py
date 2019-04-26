import numpy as np
import cv2
import random as rng

def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
    # just create a copy of original image
    img_copy = colored_img.copy()

    # convert the test image to gray image (opencv face detector expects gray images)
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

    # go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_copy


# read image and display it
img = cv2.imread('flower.jpg')
cv2.imshow('image', img)
cv2.waitKey(0)

# split channel blue
b = img.copy()
b[:, :, 1] = 0
b[:, :, 2] = 0
cv2.imshow('split blue', b)
cv2.waitKey(0)

# convert to gray
img_gray = cv2.imread('flower.jpg', 0)
cv2.imshow('gray scale', img_gray)
cv2.waitKey(0)

# smoothing with gaussian filter
blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
cv2.imshow('blur', blur)
cv2.waitKey(0)

# rotate 90 degree
height, width = img.shape[:2]
M = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
rotated90 = cv2.warpAffine(img, M, (height, width))
cv2.imshow('rotated', rotated90)
cv2.waitKey(0)

# resizing
width = width / 2
img_resize = cv2.resize(img, (int(width), int(height)))
cv2.imshow('resized', img_resize)
cv2.waitKey(0)

# detecting edges
edges = cv2.Canny(img, 100, 200)
cv2.imshow('Edges', edges)
cv2.waitKey(0)

# image segmentation
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
imgLaplacian = cv2.filter2D(img, cv2.CV_32F, kernel)
sharp = np.float32(img)
imgResult = sharp - imgLaplacian
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
imgLaplacian = np.clip(imgLaplacian, 0, 255)
imgLaplacian = np.uint8(imgLaplacian)

bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

_, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)

# Dilate a bit the dist image
kernel1 = np.ones((3, 3), dtype=np.uint8)
dist = cv2.dilate(dist, kernel1)

# Create the CV_8U version of the distance image
# It is needed for findContours()
dist_8u = dist.astype('uint8')

# Find total markers
contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create the marker image for the watershed algorithm
markers = np.zeros(dist.shape, dtype=np.int32)

# Draw the foreground markers
for i in range(len(contours)):
    cv2.drawContours(markers, contours, i, (i + 1), -1)

# Draw the background marker
cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)

# Perform the watershed algorithm
cv2.watershed(imgResult, markers)

mark = markers.astype('uint8')
mark = cv2.bitwise_not(mark)
colors = []
for contour in contours:
    colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
# Create the result image
dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
# Fill labeled objects with random colors
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i, j]
        if 0 < index <= len(contours):
            dst[i, j, :] = colors[index - 1]

cv2.imshow('segmentation', dst)
cv2.waitKey(0)

# face detection
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
img_face = cv2.imread('faces2.jpg')
faces_detected_img = detect_faces(haar_face_cascade, img_face)
cv2.imshow('faces detected', faces_detected_img)
cv2.waitKey(0)

# video capturing (first 5 frames with 0.5s delay between them)
cap = cv2.VideoCapture('sample.avi')
cap.set(cv2.CAP_PROP_POS_MSEC, 500)

if not cap.isOpened():
    print("Error opening video stream or file")

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if count == 5:
        break
    if ret:
        count = count + 1
        cv2.imshow('Frame', frame)
        cv2.waitKey(0)
    else:
        break

# When everything done, release the video capture object
cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
