## Table of Contents
* [General](#general)
* [Functionalities (OpenCV)](#functionalities-opencv)
   * [General](#general)
   * [Video Capturing and Waitkey](#video-capturing-and-waitkey)
   * [Drawing Shapes](#drawing-shapes)
   * [Thresholding](#thresholding)
   * [Filtering](#filtering)
   * [Edge detection](#edge-detection)
   * [Template Matching](#template-matching)
   * [Corner Detection](#corner-detection)
   * [Feature Matching](#feature-matching)
   * [Background Subtractor (moving object detection)](#background-subtractor-moving-object-detection)
   * [Object detection with Haar Cascade](#object-detection-with-haar-cascade)
   * [Miscellaneous](#miscellaneous)

## General
This [page](https://heartbeat.fritz.ai/opencv-python-cheat-sheet-from-importing-images-to-face-detection-52919da36433)
gives a more detailed overview.
This [video](https://pythonprogramming.net/loading-images-python-opencv-tutorial/) by sentdex is start of tutorial
series.

## Functionalities (OpenCV)

### General
* General commands
    * cv2.imread(/path, cv2.IMREAD_GRAYSCALE)
    * cv2.imshow("frame", frame)
    * cv2.imwrite("name", obj)
    * cv2.waitKey(0)
    * cv2.DestroyAllWindows()
    * cap = cv2.VideoCapture(0)
    * _, frame = cap.read()

### Video Capturing and Waitkey

```python
cap =  cv2.VideoCapture(<camera number>/<video file name>)

while True:
    ret, frame = cap.read()
    gray = cv2.cvrtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
   
### Drawing shapes

```python
cv2.line(img, (0,0), (150, 150), (255, 255, 255), 15)

cv2.rectangle(img, (15, 25), (200, 150), (0, 255, 0), 5)

cv2.circle(img, (100, 63), 55, (0, 0, 255), -1 <this fills in the circle, ie negative line width>)

pts = np.array([[1,2], [3, 15], [7, 20], [19, 20]])
pts = pts.reshape(-1, 1, 2)
cv2.polylines(img, pts, True <connect final pt to first pt>, (0, 255, 4), 3)


cv2.putText(img, "hello world", (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 255, 255), 1, cv2.LINE_AA)

```

### Thresholding

```python

retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

img_greyscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                                #minimum, maximum
retval2, threshold2 = cv2.threshold(img_greyscaled, 12, 255, cv2.THRESH_BINARY)

gaus = cv2.adaptiveThreshold(img_greyscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

```

### Filtering

```python
    ret, frame = cap.read()

    # hsv hue(color) sat(intensity) value
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower and upper values for color filtering
    lower_red = np.array([90, 0, 0])
    upper_red = np.array([110, 255, 255])

    # filtering
    mask = cv2.inRange(hsv, lower_red, upper_red)
    masked_img = cv2.bitwise_and(frame, frame, mask=mask)
    # blur = cv2.GaussianBlur(masked_img, (15, 15), 0)

    # morphing
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(masked_img, kernel, iterations=1)
    dialation = cv2.dilate(masked_img, kernel, iterations=1)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    median = cv2.medianBlur(masked_img, 15)

    cv2.imshow("video_cap", masked_img)
    cv2.imshow("opening", opening)
    cv2.imshow("closing", closing)

```

### Edge Detection

```python
ret, frame = cap.read()

lap_grad = cv2.Laplacian(frame, cv2.CV_64F)

edges = cv2.Canny(frame, 100, 150)

```

### Template Matching
* Template matching used for matching different small parts within an image.

```python
w, h = to_match.shape[::-1]
res = cv2.matchTemplate(base_img_gs, to_match, cv2.TM_CCOEFF_NORMED)
threshold = 0.8

loc = np.where(res > threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(base_img, pt, (pt[0]+w, pt[1]+h), (0, 255, 255), 2)
```

### Corner Detection

```python
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = np.float32(img_gray)

                                        # how many, min dist, max dist
corners = cv2.goodFeaturesToTrack(img_gray, 100, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)
```

### Feature Matching
The good thing about this is that the object need not have the same rotation, angle, lighting etc.
Homography or brute forcing.
```python
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
```

### Background Subtractor (moving object detection)
Foreground extraction (background reduction) in depth.
This is helpful in detecting objects that are moving.

```python
cap = cv2.VideoCapture("video/people-walking.mp4")

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow("original", frame)
    cv2.imshow("fg", fgmask)

    cv2.waitKey(10000)

cap.release()
```

### Object detection with Haar Cascade

```python
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

    cv2.imshow("frame", frame)
```

### Miscellaneous
* cv2.GrabCut is for manually extracting different areas within an image.
