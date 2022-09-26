A3C is currently the most powerful AI algorithm to date (as of early 2018). This algorithm uses Deep Convolutional Q-Learning.

## Table of Contents
* [Actor-Critic](#actor-critic)
* [Asynchronous](#asynchronous)
* [Advantage](#advantage)
* [Long Short-term Memory (LSTM) Layer](#long-short-term-memory-lstm-layer)

## Actor-Critic
This adds a second set of outputs to the neural network.
* Top one being the original one - set of actions, also known as the actor. This is the part where the agent chooses what 
  it wants to do.
* Bottom one being the new one - this is the value of the state that the agent is in, this is known as the critic.

![Actor-Critic](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/actor-critic.png)

The Q values in the actor are known as the policy. Sometimes represented by a Greek letter P.

## Asynchronous
This adds an element where there are multiple agents within an environment at once. These are all initialised differently, all agents start at different points within the environment. The agents provide each other with knowledge on how to overcome the environment, preventing the agent from getting stuck on a set part within the environment.

The agents share one neural network that feeds into separate actors to perform different actions. The neural network also feeds into one critic that connects them together. The agents provide each other with knowledge of the environment through the critic. The agents share the weights within the network and when the network gets updated, all agents weights are updated. Each agent has its own actions output and the value of the state they ended in is sent to the critic. The critic then shares the information to all agents.

![Asynchronous](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/asynchronous.png)

## Advantage
This consists of two values being backpropagated through the critic to the agents. These are:
* Value Loss (related to the critic)
* Policy Loss (related to the actor)

To help calculate the policy loss we input an advantage equation.

![Advantage Equation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/advantage-equation.png)

The critic knows the value of state but it doesn't know how much better the Q-value is that is being selected compared to the current value of state. This is where the advantage comes in. The higher the advantage, the more the agents will look at doing those actions. 

The policy loss helps to improve the agent's behaviour by making them do more of the positive actions rather than the negative impacting ones. 

## Long Short-term Memory (LSTM) Layer
A Long Short-term Memory Layer is a layer within the neural network that allows the agent to have a memory. This allows it to remember what their last few actions and states were. This can either be added before the hidden layer or replace it. This can be used to replace the hidden layer because the LSTM is so powerful that it doesn't require one.

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/artificial_intelligence/2.%20a3c/model.py) for an example of an A3C.

```python
# Making the A3C brain
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1) # Output = V(s)
        self.actor_linear = nn.Linear(256, num_outputs) # Output = Q(s,a)
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train() # Puts into train mode
    
    def forward(self, inputs):
        inputs, (hx, cx) = inputs # hx = Hidden states; cx = Cell states
        x = F.elu(self.conv1(inputs)) # Propagates images to first layer; elu = Exponential Linear Unit
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32 * 3 * 3) # Flatten vector
        (hx, cx) = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
```