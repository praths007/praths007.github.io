## Table of Contents
* [Deep Convolutional Q-Learning](#deep-convolutional-q-learning)
* [Eligibility Trace (n-step Q-Learning)](#eligibility-trace-n-step-q-learning)

## Deep Convolutional Q-Learning
Deep Convolutional Q-Learning is when the environment is a group of images. We then pass these images through a CNN. This is more realistic then using a vector because humans use their sight as one of their primary senses.

![Deep Convolutional Q-Learning](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/deep-convolutional-q-learning.png)

## Eligibility Trace (n-step Q-Learning)
Eligibility Tracy is when the agent takes a set of steps, rather than just 1 step at a time, before calculating the reward it has achieved. 

![Eligibility Trace](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/eligibility-trace.png)

During the process the agent keeps a trace of eligibility. For example: if there is a step that provides a negative reward, it keeps a track of that step and tries to avoid it.

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/artificial_intelligence/1.%20deep_convolutional_q_learning/ai.py) for an example of Deep Convolutional Q-Learning.

```python
# Part 1 - Building the AI

# Making the brain
class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        # in_channels = 1 for black and white, 3 for coloured
        # out_channels = 32 processed images
        # kernel_size = 5x5 feature detector
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, 80, 80)), out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)
        
    def count_neurons(self, image_dim):
        # * converts the image_dim to a list that passes the values through the function
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Making the body
class SoftmaxBody(nn.Module):
    def __init__(self, T): # T = Temperature
        super(SoftmaxBody, self).__init__()
        self.T = T
    
    def forward(self, outputs):    
        probs = F.softmax(outputs * self.T) # probs = probabilities
        actions = probs.multinomial()
        return actions
        
# Making the AI
class AI():
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
    
    # Used to call other functions from another class
    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        # Convert back to a numpy array
        return actions.data.numpy()
```