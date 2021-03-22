# CACB - Contextual Bandit with Continuous Actions

## About

Contextual Bandit (1-step reinforcement learning) model with
continuous action space.

CACB learns a policy from (context, action, cost) triplets to choose
an action so as to minimize the expected cost given the context. The
model discretizes the provided action space between the minimum and
the maximum and reduces to a supervised regression task where the cost
is predicted separately for each action; With the expected cost of
each action, the policy returns either the least costly action (exploit)
or one of its neighbours (explore).

## Installation

```
pip install git+https://github.com/veicco/cacb.git
```

## Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from cacb.cacb import ContinuousActionContextualBanditModel

def get_cost(action, context):
    target = 0
    if context == 1:
        target = 60
    if context == 2:
        target = 70
    return (action - target) ** 2

def plot(costs_1, costs_2, actions_1, actions_2):
    fig, ax = plt.subplots(ncols=2, figsize=(12,4))    
    ax[0].plot(costs_1, 'g', label="context = 1")
    ax[0].plot(costs_2, 'b', label="context = 2")
    ax[0].legend()
    ax[0].set_title("Cost")
    ax[1].plot(actions_1, 'g', label="context = 1")
    ax[1].plot(actions_2, 'b', label="context = 2")
    ax[1].legend()
    ax[1].set_title("Action")

cacb = ContinuousActionContextualBanditModel(
    min_value=50,
    max_value=100,
    action_width=1,
    initial_action=50,
    regression_model=GradientBoostingRegressor()
)

costs_1 = []
costs_2 = []
actions_1 = []
actions_2 = []
for i in range(1000):
    context = np.array(np.random.choice([1, 2]))
    epsilon = max(1/(i + 1), 0.1)
    action, prob = cacb.predict(context, epsilon, exploration_width=1)
    cost = get_cost(action, context)
    if context == 1:
        costs_1.append(cost)
        actions_1.append(action)
    if context == 2:
        costs_2.append(cost)
        actions_2.append(action)
    cacb.learn(context, action, cost, prob)
plot(costs_1, costs_2, actions_1, actions_2)
```

![Costs and chosen actions over time.](https://github.com/veicco/cacb/blob/master/img/plot.png?raw=true)

## Development

Run unit tests:

```python
pytest test/
```
