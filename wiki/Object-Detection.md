## Table of Contents
* [Single Shot Detection (SSD)](#single-shot-detection-ssd)
* [Multi-Box Concept](#multi-box-concept)
* [Predicting Object Positions](#predicting-object-positions)
* [The Scale Problem](#the-scale-problem)

## Single Shot Detection (SSD)
There is a technique called Object Proposal that uses colours. For example: if an area has the same colours then chances are it is not an object. However, Single Shot Detection does not use the Object Proposal technique. Instead it looks at the image once and makes it's predictions from there.

![Single Shot Detection](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/ssd.png)

## Multi-Box Concept
Multi-Box Concept is used to break an image down into segments. Within these segments it creates boxes, think of each box as its own image. Single Shot Detection uses a Convolutional Neural Network that tries to identify if the image contains an object. In the below example, the red highlighted boxes have detected that there is a person there.

![Multi-Box Concept](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/multi-box.png)

When training the Single Shot Detection you have to input a rectangle with a tag to state that there is a person/object there.

## Predicting Object Positions
Using the ground truth and the training of images using the Single Shot Detection, the algorithm can predict where objects are located. Another factor to help this is in relation to the boxes that overlap each other. These show that there is a high probability of an object being there. 

Ground truth means that boxes are placed on the image to help with training the Single Shot Detection.

![Predicting Objects](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/predicting-objects.png)

Through training, two things happen:
* Each box will learn to better classify objects inside it, this uses the ground truth to assess this.
* The Single Shot Detection gets better at identifying the exact final output rectangle that should be created that 
  matches the ground truth.

## The Scale Problem
Sometimes there will be an object that is extremely large in the image and the Single Shot Detection may not pick this up within the largest size. However, as the Single Shot Detection goes through multiple sizes of the image and remembers the previous state, it may not detect the object within the largest size but it will detect it in smaller sizes.

The Single Shot Detection uses a Convolutional Neural Network to break down the image into layers and will use it through the Multi-Box Concept to determine the ground truth's position. All of this is done within one Convolutional Neural Network.

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/computer_vision/1.%20object_detection.py) for an example of object detection.

```python
# Defining a function that will do the detections
def detect(frame, net, transform):
    """
    Frame = the image
    net = the network
    transform = transformations applied to the image. Making the images compatible with the network
    """
    # Getting the height and width of the image
    height, width = frame.shape[:2]
    # Tranformed frame after transformation
    frame_t = transform(frame)[0]
    # NumPy array to Torch Tensor
    # Invert red, blue, green to green, red, blue (was trained on these colours in that order)
    # Permute does this, 2 being green, 0 being red and 1 being blue
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    # Add new dimension (unsqueeze), should always be first index - NN is only able to accept batches of data (Pytorch)
    # Variable - torch variable that contains a tensor and a gradient
    x = Variable(x.unsqueeze(0))
    # Feeds x to our neural network
    y = net(x)
    # Retrieve the values of output
    detections = y.data
    # First width & height = top left corner
    # Second width & height = bottom right corner
    # Used to normalize the scaled values between 0 & 1
    scale = torch.Tensor([width, height, width, height])
    
    # The detections Tensor consists of:
    # [batch, number of classes/objects, number of occurence, (score, x0, y0, x1, y1)]
    
    # Number of classes = detections.size()
    for i in range(detections.size(1)):
        # Occurence of the objects
        j = 0
        # While the score (last 0) of the occurence (j) is detected (i) is greater than 0.6, continue loop
        # first 0 = batch
        while detections[0, i, j, 0] >= 0.6:
            # Point (pt) is a torch Tensor
            # 1: = last 4 elements in the tuple mentioned above within the Tensor (x0, y0, x1, y1)
            pt = (detections[0, i, j, 1:] * scale).numpy()
            # pt[0] -> pt[3] = (x0, y0, x1, y1)
            # Creates the rectangle
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            # Creates the label
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            # Update the occurence of the objects
            j += 1
    return frame
```