## Table of Contents
* [Deep Q-Learning](#deep-q-learning)
* [Experience Relay](#experience-relay)
* [Action Selection Policies](#action-selection-policies)

## Deep Q-Learning
Deep Q-Learning is Q-Learning but with the information passed through an Artificial Neural Network.

![Deep Q-Learning](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/deep-q-learning.png)

The neural network predicts numbers equal to the amount of actions, this is the output layer. The input within the neural network consists of the position of the state the agent is currently going to (this is passed in as a vector). Each output layer has a target which consists of the state the agent is moving to via the action it is taking.

From there we take the target from the output values, square it and add them up. This provides us with the loss which needs to be as close to 0 as possible. We then backpropagate these values through the network and update the weights of the synapses for the agent to understand the environment better.

![Deep Q-Learning - Learning](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/deep-q-learning-learning.png)

Once the agent has completed the learning process, we pass the Q values through a SoftMax function which helps select the optimal Q value to take as its action. This then ends up in the next state.

![Deep Q-Learning - Acting](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/deep-q-learning-acting.png)

## Experience Relay
Experience Relay refers to when a set of states are not input into the neural network straight away. Instead, they are saved into memory of the agent. When the agent reaches a set state within the environment it randomly selects a uniform distributed sample from a batch of experiences it has saved within its memory.

Each experience consists of 4 characteristics: the state it was in, the action that it took, the state it ended up in and the reward it achieved through the action it took within that specific state.

It takes those experiences, passes them through the network and learns from them.

## Action Selection Policies
Action Selection Policies revolve around the concept of Exploration vs Exploitation. Using this concept, we exploit the good action selections while also exploring new actions to understand which are best to fit the environment. The most commonly used action selections are:
* ε-greedy - you select the good action most of the time (exploitation). However, there is a random chance it will choose 
  a different action (exploration).
* ε-soft (1-ε) - this is the opposite of ε-greedy. You are more likely to choose an action at random (exploration) with a 
  small chance of using the optimal action (exploitation).
* SoftMax - provides a probability for each action, all actions must add up to 1. The best action is used most of the time 
  (exploitation) with a small probability of the other actions being chosen (exploration). These actions are not chosen at 
  random.

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/artificial_intelligence/0.%20deep_q_learning/ai.py) for an example of Deep Q-Learning.

```python
# Implementing Deep Q-Learning
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    # Which action to play at each state
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True))*100) # Temperature = 100
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        # ReplyMemory => memory attribute
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.) # Cannot be 0 or it will crash the game
    
    def save(self):
        torch.save({'state_dict' : self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict()
                    }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('=> loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Done!')
        else:
            print('No checkpoint found...')
```