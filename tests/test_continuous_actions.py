import numpy as np
import pandas as pd
import os
from numpy.testing import assert_array_equal
from cb.continuous_actions import ContextualBanditContinuousActionsModel

mock_log_data = pd.DataFrame(
    [
        [0.90, 115, -3, 0, 0, 0],
        [0.90, 100, 0, 0, 0, 1],
        [0.90, 105, -1, 0, 1, 0],
        [0.90, 110, 2, 0, 1, 1],
        [0.0333, 120, 4, 1, 1, 0],
    ]
)

mock_file = "./tests/mock.csv"

np.random.seed(123)


def test_get_actions():
    actions1 = ContextualBanditContinuousActionsModel(
        min_value=10,
        max_value=15,
        action_width=1,
    )._get_actions()
    assert actions1 == [10, 11, 12, 13, 14, 15]

    actions2 = ContextualBanditContinuousActionsModel(
        min_value=-10,
        max_value=-5,
        action_width=1,
    )._get_actions()
    assert actions2 == [-10, -9, -8, -7, -6, -5]

    actions3 = ContextualBanditContinuousActionsModel(
        min_value=10,
        max_value=15,
        action_width=2,
    )._get_actions()
    assert actions3 == [10, 12, 14]


def test_log_example():
    context = np.array([1, 2, 3])
    action = 11
    cost = 100
    prob = 0.75

    # uncategorized actions
    model_1 = ContextualBanditContinuousActionsModel(
        min_value=10, max_value=15, action_width=1, categorize_actions=False
    )
    model_1._log_example(context, action, cost, prob)
    assert_array_equal(
        model_1.logged_data, np.array([0.75, 100, 11, 1, 2, 3]).reshape(1, -1)
    )

    # categorized actions
    model_2 = ContextualBanditContinuousActionsModel(
        min_value=10, max_value=15, action_width=1, categorize_actions=True
    )
    model_2._log_example(context, action, cost, prob)
    assert_array_equal(
        model_2.logged_data,
        np.array([0.75, 100, 0, 1, 0, 0, 0, 0, 1, 2, 3]).reshape(1, -1),
    )

    # log another example
    model_2._log_example(context, 10, 90, 0.90)
    assert_array_equal(
        model_2.logged_data,
        np.array(
            [
                np.array([0.75, 100, 0, 1, 0, 0, 0, 0, 1, 2, 3]),
                np.array([0.90, 90, 1, 0, 0, 0, 0, 0, 1, 2, 3]),
            ]
        ),
    )


def test_exploit():
    model = ContextualBanditContinuousActionsModel(
        min_value=10,
        max_value=15,
        action_width=1,
    )

    # lowest cost exists
    costs_per_action = {
        10.0: 100.0,
        11.0: 100.0,
        12.0: 90.0,
        13.0: 100.0,
        14.0: 100.0,
        15.0: 100.0,
    }
    epsilon = 0.10
    action, prob = model._exploit(costs_per_action, epsilon)
    assert action == 12.0
    assert prob == 0.90

    # no clear winner => should choose the first of the best
    costs_per_action = {
        10.0: 100.0,
        11.0: 90.0,
        12.0: 100.0,
        13.0: 90.0,
        14.0: 90.0,
        15.0: 100.0,
    }
    epsilon = 0.10
    action, prob = model._exploit(costs_per_action, epsilon)
    assert action == 11.0
    assert prob == 0.90


def test_explore():
    model = ContextualBanditContinuousActionsModel(
        min_value=10,
        max_value=15,
        action_width=1,
    )
    # the best action is 12
    costs_per_action = {
        10.0: 100.0,
        11.0: 100.0,
        12.0: 90.0,
        13.0: 100.0,
        14.0: 100.0,
        15.0: 100.0,
    }

    # exploration width = 1
    epsilon = 0.10
    exploration_width = 1
    action, prob = model._explore(costs_per_action, epsilon, exploration_width)
    assert prob == 0.05
    assert action in [11, 13]

    # exploration direction = left
    epsilon = 0.10
    exploration_width = 1
    direction = "left"
    action, prob = model._explore(
        costs_per_action, epsilon, exploration_width, direction
    )
    assert prob == 0.10
    assert action == 11

    # exploration width = 2
    epsilon = 0.10
    exploration_width = 2
    action, prob = model._explore(costs_per_action, epsilon, exploration_width)
    assert prob == 0.025
    assert action in [10, 11, 13, 14]

    # exploration width = 1, optimum in the end
    costs_per_action = {
        10.0: 90.0,
        11.0: 100.0,
        12.0: 100.0,
        13.0: 100.0,
        14.0: 100.0,
        15.0: 100.0,
    }
    epsilon = 0.10
    exploration_width = 1
    action, prob = model._explore(costs_per_action, epsilon, exploration_width)
    assert prob == 0.10
    assert action == 11

    # exploration width = 1, optimum in the left end, left direction
    costs_per_action = {
        10.0: 90.0,
        11.0: 100.0,
        12.0: 100.0,
        13.0: 100.0,
        14.0: 100.0,
        15.0: 100.0,
    }
    epsilon = 0.10
    exploration_width = 1
    action, prob = model._explore(
        costs_per_action, epsilon, exploration_width, direction="left"
    )
    assert prob == 0.10
    assert action == 10


def test_existing_data():
    mock_log_data.to_csv(mock_file, header=None, index=None)
    model = ContextualBanditContinuousActionsModel(
        min_value=10,
        max_value=15,
        action_width=1,
        data_file=mock_file,
    )
    assert model.logged_data.shape[0] == 5
    model.learn(np.array([0, 0, 1]), 0, 100, 0.90)
    log_file_data = pd.read_csv(mock_file, header=None).values  # type: ignore
    assert log_file_data.shape[0] == 6
    os.remove(mock_file)


def test_existing_data_and_memory():
    mock_log_data.to_csv(mock_file, header=None, index=None)
    model = ContextualBanditContinuousActionsModel(
        min_value=10, max_value=15, action_width=1, data_file=mock_file, memory=10
    )
    assert model.logged_data.shape[0] == 5

    model = ContextualBanditContinuousActionsModel(
        min_value=10, max_value=15, action_width=1, data_file=mock_file, memory=3
    )
    assert model.logged_data.shape[0] == 3
    assert list(model.logged_data[-1]) == [0.0333, 120, 4, 1, 1, 0]
    model._log_example(np.array([0, 1, 1]), 1, 105, 0.0333)
    assert model.logged_data.shape[0] == 3
    assert list(model.logged_data[-1]) == [0.0333, 105, 1, 0, 1, 1]
    os.remove(mock_file)


def test_get_previous_move():
    mock_log_data.to_csv(mock_file, header=None, index=None)
    model = ContextualBanditContinuousActionsModel(
        min_value=10,
        max_value=15,
        action_width=1,
        data_file=mock_file,
    )
    assert model._get_previous_move(0.1) == (True, 10.0, 2.0)
    os.remove(mock_file)
