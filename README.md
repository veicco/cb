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

TODO

## Example

TODO

## Development

Run unit tests:

```python
pytest test/
```
