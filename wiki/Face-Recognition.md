## Table of Contents
* [Viola-Jones Algorithm](#viola-jones-algorithm)
* [Haar-Like Features](#haar-like-features)
* [Integral Image](#integral-image)
* [Training Classifiers](#training-classifiers)
* [Adaptive Boosting (Adaboost)](#adaptive-boosting-adaboost)
* [Cascading](#cascading)

## Viola-Jones Algorithm
Consists of two stages: the training of the algorithm and the detection of the faces in an application. Viola-Jones Algorithm is designed to look for the frontal face. Firstly, it turns an image into grey-scale and then looks for core features of the face. 

It has to identify a set of specific features in order for it to confirm that it is looking at a face. For example: eye brows, then eyes, then nose, then mouth and then cheeks. Once it has identified all of those, it will class it as a face. The features depend on the training of the algorithm.

## Haar-Like Features
This consists of 3 features:

![Features](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/features.png)

Using an example of one of the features, this is how the features are displayed in pixels. 0 being white, 1 being black.

![Face Detection Example](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/face-detection-example.png)

From the pixels it splits it into two categories, white & black. With each category, it calculates the sum of the categories pixels and then divides the average of that category. Grey scale has a range of 0-255.

From the image: White = 0.166. Black = 0.568. It then takes black - white = 0.402. This threshold is there to determine if the feature is the correct one (in this example the nose).

## Integral Image
Integral Image is a hack for the Haar-like Features that is used to speed up calculation processing. An integral image takes a normal image and with every box, it is the sum of the numbers to its left & above it. For example: the box in the integral image is the sum of the digits within the green lines going from left and up.

![Integral Image](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/integral-image.png)

The final result would look like this:

![Integral Image Final](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/final-integral-image.png)

Say we wanted a specific feature within the image which is a set amount of pixels. Using the integral image we can calculate this very effectively, without having to calculate all the numbers up which would take a lot of computer processing speed. This method is effective for when you have large amounts of images. E.g. 1000+ pixels. You will only need to look at 4 values rather than the full 1000.

Here is an example of Integral Images:
* We have our feature and take the bottom right number to start. This corresponds to the large orange space which makes 
  that total number (left image).
* We then take the number just above the top right value which consists of the very top row above the feature and subtract 
  this from our previous value (right image).

![Integral Image Example P1](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/integral-image-example-p1.png)

* We the add the number from just to the left of the top-left corner of the feature (left image).
* We then subtract the number to the left of the bottom-left corner of the feature (right image).

![Integral Image Example P2](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/integral-image-example-p2.png)

We are then left with our feature.

![Totals](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/totals.png)

## Training Classifiers
Training Classifies is an algorithm that shrinks the image to 24px by 24px and then tries to identify which features of that image are one of the Haar-like features. It shrinks the image to make it easier to read the image to locate the necessary features. This is only done through the training stage so that the algorithm can learn the correlations of the features.

On new images the features get scaled up and the image stays the same size. For this to work effectively, you need to supply a lot of frontal face images for the features to be detected and a lot of non-face images (any random picture that doesn't have faces in them) so the algorithm can learn the distinction between the two. The non-face images don't have to be 24px by 24px. They can be the normal size and it will take 24px of each of that image identifying that these are not faces.

## Adaptive Boosting (Adaboost)
In a 24px by 24px image you can have over 180,000+ features. The Adaboost is designed to identify a combination of features that can help increase the accuracy of the face detection within the Viola-Jones Algorithm.

![Adaboost Formula](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/adaboost-formula.png)

Each feature on its own is called a Weak Classifier. When multiple of these are joined together, this makes a Strong Classifier. On the first feature the Adaboost will identify images that are suitable to that feature. It will flag up false positives and false negatives and then adjust the weights of those images in preparation for the second feature. This is done so that it can confirm that they are faces or are not faces. It will then repeat this until it has gone through all the features. 

Adaboost is the first step of improving the algorithm, the second step is Cascading.

## Cascading
* We take a sub-window that we are using to analyse our image (a square box that goes through each pixel).
* We look for the first feature within our list of features, if its not present in that sub-window we reject the sub-
  window.

If there is a feature, we look for the second one in that sub-window. If it's not present, reject that sub-window. Rinse and repeat (T = True, F = False).

![Sub-Window](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/sub-window.png)

If there is not a set amount of features identified, it will skip the remaining features and will flag this image as a non-face image.

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/computer_vision/0.%20face_recognition.py) for an example of face recognition.

```python
# Defining a function that will do the detections
def detect(grey, frame):
    """
    The two arguments that are inside of the detect function consist of:
        - grey - the black and white image
        - frame - the image we want to draw the rectangles on
    
    Faces is a tuple of 4 elements: 
        - x, y - these are the coordinates of the upper left corner
        - w, h - width and height of the rectangle
    
    Arguments are as follows: 
        - grey = black and white image 
        - 1.3 = scale factor, how much of the size of the image is going to be reduced
        - 5 = minimum number of neighbour zones that must be accepted
    """
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    for (x, y, w, h) in faces:
        # Draw the rectangle
        """
        cv2.rectangle() consists of:
            - The image we want to draw the rectangles on
            - The coordinates of the upper left corner of the rectangle
            - The coordinates of the lower right corner of the rectangle
            - Colour of the rectangle
            - The thickness of the edges of the rectangle
        """
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        ## Eyes
        # Zone of interest which is inside the detector rectangle
        # You need the black and white & coloured versions
        roi_grey = grey[y:y+h, x:x+w]
        roi_colour = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_colour, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame
```