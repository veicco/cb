# CB: Contextual Bandits with Python

## About

Contextual Bandit (1-step reinforcement learning) implementation with Python.

CB learns a policy from (context, action, cost) triplets to choose
an action so as to minimize the expected cost given the context. The model 
reduces to a supervised regression task where the cost is predicted separately 
for each action, choosing either the least costly action (exploit) or one
of the suboptimal actions in order to explore.

This package includes a model for both discrete actions and continuous actions.
The continuous actions model discretizes the provided action space between the 
minimum and the maximum, and it explores by choosing one of the neighbours
of the optimal action.

## Installation

```bash
pip install git+https://github.com/veicco/cb.git@v0.2.0
```

## Examples

### Continuous Actions

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from cb.continuous_actions import ContextualBanditContinuousActionsModel

def get_cost(action, context):
    target = 0
    if context == 1:
        target = 60
    if context == 2:
        target = 70
    return (action - target) ** 2

def plot(costs_1, costs_2, actions_1, actions_2):
    _, ax = plt.subplots(ncols=2, figsize=(12, 4))
    ax[0].plot(costs_1, "g", label="context = 1")
    ax[0].plot(costs_2, "b", label="context = 2")
    ax[0].legend()
    ax[0].set_title("Cost")
    ax[1].plot(actions_1, "g", label="context = 1")
    ax[1].plot(actions_2, "b", label="context = 2")
    ax[1].legend()
    ax[1].set_title("Action")

model = ContextualBanditContinuousActionsModel(
    min_value=50,
    max_value=100,
    action_width=1,
    initial_action=50,
    regression_model=GradientBoostingRegressor(),
)

costs_1 = []
costs_2 = []
actions_1 = []
actions_2 = []
for i in range(1000):
    context = np.array(np.random.choice([1, 2]))
    epsilon = max(1 / (i + 1), 0.1)
    action, prob = model.predict(context, epsilon, exploration_width=1)
    cost = get_cost(action, context)
    if context == 1:
        costs_1.append(cost)
        actions_1.append(action)
    if context == 2:
        costs_2.append(cost)
        actions_2.append(action)
    model.learn(context, action, cost, prob)
plot(costs_1, costs_2, actions_1, actions_2)
```

![Costs and chosen actions over time.](https://github.com/veicco/cacb/blob/master/img/plot_continuous_actions.png?raw=true)

### Discrete Actions

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from cb.discrete_actions import ContextualBanditDiscreteActionsModel

ROCK = 'ROCK'
PAPER = 'PAPER'
SCISSORS = 'SCISSORS'

ACTIONS = [ROCK, PAPER, SCISSORS]

CONTEXTS = {
    ROCK: 0,
    PAPER: 1,
    SCISSORS: 2,
}

def get_cost(action, context):
    target = ROCK
    if context == CONTEXTS[ROCK]:
        target = PAPER
    if context == CONTEXTS[PAPER]:
        target = SCISSORS
    if context == CONTEXTS[SCISSORS]:
        target = ROCK
    return -1 if action == target else 1

def plot(costs):
    _, ax = plt.subplots(ncols=1, figsize=(12, 4))
    ax.plot(costs, "g")
    ax.legend()
    ax.set_title("Cumulative cost")

model = ContextualBanditDiscreteActionsModel(
    actions=ACTIONS,
    initial_action=ROCK,
    regression_model=GradientBoostingRegressor(),
)

costs = [0]
for i in range(50):
    context = np.array(np.random.choice([
        CONTEXTS[ROCK],
        CONTEXTS[PAPER],
        CONTEXTS[SCISSORS],
    ]))
    epsilon = max(1 / (i + 1), 0.05)
    action, prob = model.predict(context, epsilon)
    cost = get_cost(action, context)
    costs.append(costs[-1] + cost)
    model.learn(context, action, cost, prob)
plot(costs)
```

![Cumulative cost over time.](https://github.com/veicco/cacb/blob/master/img/plot_discrete_actions.png?raw=true)

## Development

Run unit tests:

```python
pytest test/
```
