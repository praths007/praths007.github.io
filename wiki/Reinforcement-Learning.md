Reinforcement Learning consists of different environments in which the AI takes an action and then returns a state where they can be rewarded based on the state it takes. The state taken doesn't always provide it with a reward.

A real life example consists of training a dog. If it obeys a command, you give it a treat. If it doesn't, you don't give it a treat. Through that process it learns what action it needs to take in certain states to receive the reward. The state are the commands you are giving it. Desired outcomes provide the AI with reward (1) and undesired with punishment (0). The models learn through trial and error.

## Table of Contents
* [Multi-Armed Bandit Problem](#multi-armed-bandit-problem)
* [Upper Confidence Bound (UCB)](#upper-confidence-bound-ucb)
* [Thompson Sampling](#thompson-sampling)

## Multi-Armed Bandit Problem
This is common reinforcement learning example where a person is challenged with deciding on which machine, out of multiple machines, provides you with the maximum number of return out of a total number of tries (e.g. 100 times).

We would need to assume that each machine has a different result in terms of winning. However, we wouldn't know which machine is best and need to work this out.
<!---
![Multi-Armed Bandit Problem](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/multi-armed-bandit-problem.png)
--->
Using the image above as an example, lets say that we know that machine D5 is the best machine. Two factors come into play when trying to determine which machine is the best one: Exploration and Expectation.

The idea is to explore the machines to find out which is the best one and at the same time, exploit the machines to create the maximum return.
<!---
![Multi-Armed Bandit Problem Process](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/multi-armed-bandit-problem-process.png)
--->
## Upper Confidence Bound (UCB)
This is a popular algorithm to solve the Multi-Armed Bandit Problem.
<!---
![Upper Confidence Bound](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/upper-confidence-bound.png)
--->
Using the ad example from the Multi-Armed Bandit Problem, here is a step by step process:
* Step 1 - Provide each machine with the same value and Confidence Bound.
* Step 2 - Choose machine at random for testing.
* Step 3 - Identify if the user has clicked on the ad displayed or not. If they haven't, the value decreases. If the user 
           has clicked on the ad, the value increases.
* Step 4 - Confidence Bound shrinks as we are becoming more confident with each prediction. Repeat from Step 2.
* FIN - One ad is given highest value and is constantly being flagged by algorithm.
<!---
See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/5.%20reinforcement_learning/0.%20upper_confidence_bound.py) for an example of Upper Confidence Bound. 
--->
```python
# Implementing UCB
import math
N = 10000
d = 10
ads_selected = []

# Step 1
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

# Step 2
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]

            # Confidence interval
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        # Step 3
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
```

## Thompson Sampling
This algorithm creates distributions based off your data. These distributions provide us with an area of where we think the expected value might be. Thompson Sampling uses the Bayesian Inference to achieve this and is another way to tackle the Multi-Armed Bandit Problem.
<!---
![Bayesian Inference](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/bayesian-inference.png)

![Thompson Sampling Process](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/thompson-sampling-process.png)
--->
Using the image below as an example, Thompson Sampling works as the following:
* Step 1 - We take a value at random from each distribution. The algorithm is more likely to pull a value from the center 
           of the curve rather than the sides.
* Step 2 - We pick the highest value of each machine as it has the highest expected return out of the three.
* Step 3 - As the highest value is within the green distribution, the algorithm moves the value within the green 
           distribution close to the expected value line.
* Step 4 - The algorithm adapts to the new information and the curve of the green machine increases in size. Raising the 
           point slightly higher.
* Step 5 - Step 1 to 4 is then repeated.
<!---
![Thompson Sampling](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/thompson-sampling.png)

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/5.%20reinforcement_learning/1.%20thompson_sampling.py) for an example of Thompson Sampling.
--->
```python
# Implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []

# Step 1
numbers_of_rewards_1 = [1] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0

# Step 2
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        # Step 2
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        # Step 3
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_reward += reward
```

### UCB vs Thompson Sampling

UCB:
* Is a deterministic algorithm
* Requires an update at every round

Thompson Sampling:
* Is a probabilistic algorithm
* Can accommodate delayed feedback
* Better empirical evidence than UCB