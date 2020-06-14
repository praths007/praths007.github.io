## Table of Contents
* [General](#general)
* [Scikit-Learn (Sklearn)](#scikit-learn-sklearn)
   * [General](#general-1)
   * [Data Preprocessing](#data-preprocessing)
   * [Model Selection](#model-selection)
   * [Accuracy & Predictions](#accuracy--predictions)
   * [Models](#models)
* [Keras](#keras)
* [TensorFlow](#tensorflow)

## General

* import [numpy](http://www.numpy.org/) as np
* import [pandas](https://pandas.pydata.org/) as pd
* import [matplotlib.pyplot](https://matplotlib.org/) as plt

## [Scikit-Learn (Sklearn)](http://scikit-learn.org/stable/)

#### General
* .fit() - used to find the internal parameters of a model
* .transform() - used to map new or existing values to data
* .fit_transform() - does both fit and transform
* .predict() - used to make predictions

* from [xgboost](https://xgboost.readthedocs.io/en/latest/) import XGBClassifier - XGBoost gradient boosting software


#### Data Preprocessing
* from sklearn.preprocessing import ...
   * [LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) - used for categorical data
   * [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) - used for dummy variables
   * [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) - used for standardising data
   * [MinMaxScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) - Used for normalising data
   * [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) - Used to replace empty spaces/missing data within a dataset


#### Model Selection
* from sklearn.model_selection import ...
   * [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) - used for splitting data into test sets and training sets
   * [cross_val_score](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) - used for K-Fold Cross Validation
   * [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) - used for grid search (tuning models)


#### Accuracy & Predictions
* from sklearn.metrics import [confusion_matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) - used to identify the accuracy of a trained model


#### Models
* from sklearn.preprocessing import [PolynomialFeatures](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) - Used for creating Polynomial Regressions

* from sklearn.svm import ...
   * [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) - the model class for Support Vector Classification & Kernel SVM
   * [SVR](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) - the model class for Support Vector Regression

* from sklearn.linear_model import ...
   * [LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) - used for Linear Regressions (single and multiple variables).
   * [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - used for Logistic Regressions

* from sklearn.tree import ...
   * [DecisionTreeRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) - used for Decision Tree Regression
   * [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) - used for Decision Tree Classification

* from sklearn.ensemble import ...
   * [RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) - used for Random Forest Regression
   * [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - used for Random Forest Classification

* from sklearn.neighbors import [KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) - K-Neighbours Classification model
* from sklearn.naive_bayes import [GaussianNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) - Naive Bayes model

* import [scipy.cluster.hierarchy](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) as sch - A popular library that can be used for dendrogram creation in Hierarchical Clustering

* from sklearn.cluster import ...
   * [AgglomerativeClustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) - A Hierarchical Clustering Model
   * [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) - K-Means clustering model

* from sklearn.discriminant_analysis import [LinearDiscriminantAnalysis](http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis) as LDA - Linear Discriminant Analysis model

* from sklearn.decomposition import ...
   * [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) - Principal Component Analysis model
   * [KernelPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) - Kernel PCA model


## Keras

* from keras.models import [Sequential](https://keras.io/models/sequential/) - basic building block to creating a model
* from keras.layers import ...
   * [Dense](https://keras.io/layers/core/#dense) - basic function for linear models
   * [Dropout](https://keras.io/layers/core/#dropout) - used to add dropout to layers
   * [Flatten](https://keras.io/layers/core/#flatten) - used to flatten convolutional layers
   * [Conv2D](https://keras.io/layers/convolutional/#conv2d) - a basic convolutional layer
   * [MaxPooling2D](https://keras.io/layers/pooling/#maxpooling2d) - used to apply max pooling to a convolutional layer
* from keras.wrappers.scikit_learn import [KerasClassifier](https://keras.io/scikit-learn-api/) - used to wrap a sequential model to allow the model to be fit to datasets

## TensorFlow

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_nodes_hl1, n_nodes_hl2, n_nodes_hl3 = 500, 500, 500
n_classes = 10
batch_size = 100

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")


def neural_network_model(data):

    # height * width
    hidden_1_layer = {"weights": tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    "biases": tf.Variable(tf.random_normal([n_classes]))}

    # model = (input_data * weight) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer["weights"]), hidden_1_layer["biases"])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer["weights"]), hidden_2_layer["biases"])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer["weights"]), hidden_3_layer["biases"])
    l3 = tf.nn.relu(l3)

    op = tf.matmul(l3, output_layer["weights"]) + output_layer["biases"]
    # op = tf.nn.sigmoid(op)
    return op



def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x_epoch, y_epoch = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: x_epoch, y: y_epoch})
                epoch_loss += c
            print('Epoch', epoch, "completed out of", epochs, "loss:", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)
```

## OpenCV
This [page](https://heartbeat.fritz.ai/opencv-python-cheat-sheet-from-importing-images-to-face-detection-52919da36433)
gives a more detailed overview.

This [video](https://pythonprogramming.net/loading-images-python-opencv-tutorial/) by sentdex is start of tutorial
series.
* 1st video, initial commands:
    cv2.imread(/path, cv2.IMG_GREYSCALE), cv2.imshow, cv2.imwrite, cv2.waitKey, cv2.DestroyAllWindows()
    
* 2nd video, video capturing: 

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
   
* 3rd video, drawing things:

```python
cv2.line(img, (0,0), (150, 150), (255, 255, 255), 15)

cv2.rectangle(img, (15, 25), (200, 150), (0, 255, 0), 5)

cv2.circle(img, (100, 63), 55, (0, 0, 255), -1 <this fills in the circle, ie negative line width>)

pts = np.array([[1,2], [3, 15], [7, 20], [19, 20]])
pts = pts.reshape(-1, 1, 2)
cv2.polylines(img, pts, True <connect final pt to first pt>, (0, 255, 4), 3)


cv2.putText(img, "hello world", (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 255, 255), 1, cv2.LINE_AA)

```

* Canny edge detection and laplacian edge detection

```python
cv2.Canny(frame,,)
```

* Template matching used for matching different small parts within an image.

```python
w, h = to_match.shape[::-1]
res = cv2.matchTemplate(base_img_gs, to_match, cv2.TM_CCOEFF_NORMED)
threshold = 0.8

loc = np.where(res > threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(base_img, pt, (pt[0]+w, pt[1]+h), (0, 255, 255), 2)
```

* cv2.GrabCut is for manually extracting different areas within an image.

* Corner detection

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

* Feature matching
The good thing about this is that the object need not have the same rotation, angle, lighting etc.

```python
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
```

* Foreground extraction (background reduction) in depth
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
* Object detection with Haar Cascade

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