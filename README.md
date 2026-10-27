# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```python

import gymnasium as gym
import numpy as np
# Create FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
# Access the unwrapped environment to use the transition model
env = env.unwrapped
# Number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n
     
# Parameters
gamma = 0.99
theta = 1e-8
# Random policy: each action has equal probability
policy = np.ones((n_states, n_actions)) / n_actions
def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    iterations = 0
    
    while True:
        delta = 0
        iterations += 1
        for s in range(n_states):
            v = V[s]
            new_value = 0
            for a in range(env.action_space.n):
                prob_a = policy[s][a]
                for prob, next_state, reward, done in env.P[s][a]:
                    new_value += prob_a * prob * (reward + gamma * V[next_state])
            V[s] = new_value
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
            
    return V, iterations

V, iterations = policy_evaluation(env, policy, gamma, theta)

print("Name: Chiiradeep R  ")
print("Register Number: 212224240028 ")
print("Number of iterations:", iterations)
print("\nState-Value Function:")
print(V)

print("\nName: Chiiradeep R ")
print("Register Number: 212224240028 ")
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()
```

## Output

## Gamma = 0.99
<img width="1061" height="498" alt="Screenshot 2026-10-27 105452" src="https://github.com/user-attachments/assets/ec13e1ed-e516-442f-80b8-a07cac46e4e3" />

## Gamma = 0.6
<img width="774" height="447" alt="Screenshot 2026-10-27 110012" src="https://github.com/user-attachments/assets/e4d1d028-e1c4-4bd1-b784-a8b24505ffa2" />


## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

~~~


1. Importing Libraries: The program imports `gymnasium` to create the FrozenLake environment and `numpy` to perform numerical calculations and arrange state values in a matrix.

2. Creating the Environment: The `gym.make()` function creates a 4 × 4 FrozenLake environment with `is_slippery=True`. The agent can move left, right, up or down, but its movement may differ from the intended direction.

3. Defining Parameters: The program sets `gamma = 0.99` as the discount factor and `theta = 1e-8` as the convergence threshold. These parameters determine the importance of future rewards and when the algorithm should stop.

4. Initializing the Random Policy: The statement `policy = np.ones((n_states, n_actions)) / n_actions` creates a policy in which each of the four actions has an equal probability of 25% in every state.

5. Initializing State Values: The `np.zeros(n_states)` function initializes the values of all 16 states to zero. The iteration counter is also initialized to zero.

6. Evaluating Each State: The `while True` loop repeatedly visits all states. For each state, the program stores its old value and calculates a new value based on the possible actions and transitions.

7. Calculating Expected Rewards: The program uses the Bellman expectation equation to calculate each state's value by considering the action probability, transition probability, immediate reward, discount factor and next-state value.

8. Updating State Values: The newly calculated value replaces the old value of each state. The variable `delta` stores the maximum difference between the old and new values during an iteration.

9. Checking Convergence: The algorithm compares `delta` with `theta`. If `delta` becomes smaller than `theta`, the loop stops because the state values have sufficiently converged.

10. Displaying Results: The program prints the number of iterations and the final state-value function. It then uses `reshape(4, 4)` to display the values as a grid and `np.round()` to round them to four decimal places.

11. Interpreting the Output: The algorithm converges after 54 iterations. Hole states and the terminal goal state have zero values, while other states have positive values depending on their expected discounted future rewards under the random policy.

12. Closing the Environment: The `env.close()` statement closes the FrozenLake environment after the calculations are complete.

~~~


