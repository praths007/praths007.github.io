## Table of Contents
* [The Bellman Equation](#the-bellman-equation)
* [Deterministic Search](#deterministic-search)
* [Markov Process](#markov-process)
* [Markov Decision Process (MDP)](#markov-decision-process-mdp)
* [Policy vs Plan](#policy-vs-plan)
* [Living Penalty](#living-penalty)
* [Q-Learning](#q-learning)
* [Temporal Difference](#temporal-difference)

## The Bellman Equation

![Bellman Equation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/bellman-equation.png)

Equation breakdown:
* s - State: the state in which our agent (AI) is in
* a - Action: an action that an agent can take. The agent has a list of actions it can take, these are very important when 
  they are looked at in a state
* R - Reward: the reward an agent gets for entering a certain state
* y - Discount: (Gamma symbol) used to discount the value of the state as the agent is further away from the reward state
* s' - (s Prime) is the following state, that you will end up in by taking a certain action
* V - value of the new state
* max - used to identify the reward of all actions

Each square is a different state, reaching the state with a value of 1 the AI knows that it only has to move right once to achieve a reward.

## Deterministic Search
Deterministic Search is when the agent decides to go up and there is a 100% probability it goes up. 

Non-Deterministic Search is when the agent decides to go up but there is an 80% probability it goes up, 10% it goes left & 10% it goes right.

![Deterministic Search](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/deterministic-search.png)

Deterministic Search is unrealistic. Whereas, Non-Deterministic search is used to make the Markov Decision Process more realistic. The agent cannot go backwards.

## Markov Process
A Markov Property consists of an agents future state (including the agents choice and environment) resulting in the action it takes in it's environment but it doesn't depend on where it is now and it doesn't matter how it got there. This is the Markov Process. In the image example below, the Markov Process doesn't care how the agent got to where it is now. It only cares about the actions it can take next, the probabilities of those actions will always be the same in that state (box) it is in now.

![Markov Process](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/markov-process.png)

## Markov Decision Process (MDP)
The Markov Decision Process provides a mathematical framework for modelling decision making in situations where outcomes are partly random and partly under the control of a decision maker. The MDP is the framework in which the agent will use to understand what to do in this environment.

MDP uses the Bellman Equation but as there are random probabilities through the use of non-deterministic search it has to break the expected value into the different probabilities and take the average to determine where to go next. The bellman equation with the additional probability states looks like this:

![MDP & Bellman Equation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/mdp-and-bellman.png)

Here is another example of the equation, both of these equations are the same displayed differently.

![MDP & Bellman Equation 2](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/mdp-and-bellman2.png)

P stands for probability and the Sigma symbol stands for sum.

## Policy vs Plan
A plan is when you are using deterministic search where you have a set destination and know where to go to reach it.

![Plan](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/plan.png)

A policy is when you are using non-deterministic search where you have multiple options that can be taken within one state. For example: in order for the agent to prevent itself from losing, it has chosen some unique options that humans wouldn't normally think of. Facing away from the negative reward - it would rather face the grey wall and bump into it as it has a 10% probability of going left or right than risk not receiving a reward.

![Policy](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/policy.png)

## Living Penalty
Living Penalty is when there is a negative reward in every state except for the positive one, the agent will only receive a reward when they enter the positive state.

![Living Penalty Example](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/living-penalty-example.png)

This gives the agent an incentive to reach the reward as quickly as possible. Below are 4 environments that have different examples of living penalty values per state:

![Living Penalty Scores](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/living-penalty-scores.png)

## Q-Learning

![Q-Learning Equation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/q-learning-equation.png)

The example on the left uses the bellman equation incorporated with markov decision process. The example on the right uses the Q-Learning technique which uses the value of each action over the value of each state.

![Q-Learning Examples](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/q-learning-examples.png)

The purpose of Q-Learning is to look at each action the agent can take within a state, rather than the state the agent is in. It finds the optimal action within that state and proceeds with that action.

## Temporal Difference

![Temporal Difference Equation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/temporal-difference-equation.png)

Temporal Difference is used so that the agent can understand the difference between its current state and the state it has moved to.

![TD Before and After](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/td-before-after.png)

Using the before state and the new state we calculate the Temporal Difference. This includes the time of the previous state.

![Q-Learning Equation 1](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/q-learning-equation1.png)

We then use this to calculate the new state. This is done by taking the before state and times a learning rate version of the Temporal Difference.

![Q-Learning Equation 2](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ai/q-learning-equation2.png)

α (Alpha) - this is the learning rate

Random events can happen due to the probability of taking a different action which means Temporal Difference needs to be calculated step by step.