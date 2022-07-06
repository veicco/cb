import os
import abc
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from collections import deque
from io import StringIO


Action = Union[int, float]
Cost = Union[int, float]
Prob = float


class ContextualBanditBase:
    def __init__(
        self,
        memory: int = None,
        initial_action: Action = None,
        data_file: str = None,
        regression_model: RegressorMixin = None,
        decay_rate: float = 1.0,
    ):
        self.memory = memory
        self.initial_action = initial_action
        self.data_file = data_file
        self.regression_model = regression_model
        self.decay_rate = decay_rate
        self.logged_data = (
            self._read_logged_data_file(data_file, memory)
            if data_file
            else np.array([])
        )
        self.reg = None

    def _read_logged_data_file(
        self, data_file: str, memory: Optional[int]
    ) -> np.ndarray:
        if not os.path.exists(data_file):
            open(data_file, "w").close()
            return np.array([])
        if os.path.getsize(data_file) == 0:
            return np.array([])
        if memory is None:
            return pd.read_csv(data_file, header=None).values  # type: ignore
        with open(data_file, "r") as f:
            q = deque(f, memory)
        return pd.read_csv(StringIO("".join(q)), header=None).values  # type: ignore

    @abc.abstractmethod
    def _get_actions(self) -> List[Action]:
        pass

    def _get_actions_one_hot(self, action: Action = None) -> np.ndarray:
        actions = self._get_actions()
        actions_one_hot = np.zeros(shape=len(actions))
        if action is not None:
            actions_one_hot[actions.index(action)] = 1
        return actions_one_hot

    def _init_regressor(self, context: np.ndarray):
        if self.regression_model is not None:
            self.reg = self.regression_model
        else:
            self.reg = LinearRegression()
        action = self._get_actions_one_hot()
        x = np.append(action, context).reshape(1, -1)
        cost = np.array([1])
        self.reg.fit(x, cost)

    def _log_example(self, context: np.ndarray, action: Action, cost: Cost, prob: Prob):
        data = self.logged_data
        a = self._get_actions_one_hot(action)
        x = np.append(a, context)
        example = np.append([prob, cost], x)
        if self.data_file:
            with open(self.data_file, "a") as f:
                new_row = ",".join(np.char.mod("%f", example))
                f.write("\n" + new_row)
        if data.shape[0] == 0:
            self.logged_data = np.hstack([data, example]).reshape(1, -1)
        else:
            data = np.vstack([data, example])
            if self.memory is not None:
                data = data[-self.memory :]
            self.logged_data = data

    def _exploit(
        self, costs_per_action: Dict[Action, Cost], epsilon: Prob
    ) -> Tuple[Action, Prob]:
        best_action = min(costs_per_action, key=costs_per_action.get)  # type: ignore
        prob = 1 - epsilon
        return best_action, prob

    def _explore(
        self, costs_per_action: Dict[Action, Cost], epsilon: Prob
    ) -> Tuple[Action, Prob]:
        return self._sample_action(costs_per_action, epsilon)

    def _sample_action(
        self, costs_per_action: Dict[Action, Cost], epsilon: Prob
    ) -> Tuple[Action, Prob]:
        actions = list(costs_per_action.keys())
        costs = list(costs_per_action.values())
        max_cost = max(np.abs(costs))
        rewards_scaled = [-cost / max_cost for cost in costs]
        pmf = np.exp(rewards_scaled) / sum(np.exp(rewards_scaled))
        draw = np.random.random()
        sum_prob = 0.0
        for idx, prob in enumerate(pmf):
            sum_prob += prob
            if sum_prob > draw:
                return actions[idx], prob * epsilon
        raise ValueError("Invalid pmf: could not sample action.")

    def _get_previous_move(self, epsilon: Prob) -> Tuple[bool, Cost, Action]:
        if self.logged_data.shape[0] < 2:
            return (False, 0, 0)
        last_2 = self.logged_data[-2:]
        explored = last_2[-1][0] != (1 - epsilon)
        cost_diff = last_2[-1][1] - last_2[-2][1]
        action_diff = last_2[-1][2] - last_2[-2][2]
        return explored, cost_diff, action_diff

    def get_costs_per_action(self, context: np.ndarray) -> Dict[Action, Cost]:
        """
        Get the predicted cost for each of the actions given the
        provided context.

        Parameters
        ----------
        context : np.array([...])
            Context/feature set that action-wise costs are predicted for.

        Returns
        ----------
        costs_per_action : Dict({action: cost})
            Dictionary with actions as keys and costs as values.
        """
        costs_per_action = {}
        for action in self._get_actions():
            action_one_hot = self._get_actions_one_hot(action)
            x = np.append(action_one_hot, context)
            costs_per_action[action] = self.reg.predict(x.reshape(1, -1)).reshape(-1)[0]
        return costs_per_action

    def predict(self, context: np.ndarray, epsilon: Prob = 0.05) -> Tuple[Action, Prob]:
        """
        Predict an action given a context.

        Parameters
        ----------
        context : np.array([...])
            Context/feature set that an action is predicted for.

        epsilon : float between 0.0 and 1.0
            Probability of exploration, that is, the probability that a
            suboptimal action is returned instead of the best known action.

        Returns
        ----------
        (action, prob) : tuple
            The predicted action with the probability that it was selected.
        """

        if self.reg is None:
            self._init_regressor(context)
        costs_per_action = self.get_costs_per_action(context)
        if np.random.random() < epsilon:
            return self._explore(costs_per_action, epsilon)
        return self._exploit(costs_per_action, epsilon)

    def learn(self, context: np.ndarray, action: Action, cost: Cost, prob: Prob):
        """
        Write a new training example in the logged data and re-train
        the regression model using the accumulated training data.

        Parameters
        ----------
        context : numpy.array([...])
            Context/feature set of the training example.

        action : int of float
            Action of the training example.

        cost : int or float
            Cost of the training example.

        prob : float
            Logged probability that the given action was chosen when it was applied.
            Needed in order to do IPS weighting when learning the policy.
        """
        if self.reg is None:
            self._init_regressor(context)
        self._log_example(context, action, cost, prob)
        data = self.logged_data
        probs = data[:, 0]
        ips = 1 / probs
        weights = ips * (np.linspace(0, 1, len(ips) + 1) ** self.decay_rate)[1:]
        costs = data[:, 1]
        x = data[:, 2:]
        self.reg.fit(x, costs, sample_weight=weights)

    def get_logged_data_df(self) -> pd.DataFrame:
        """
        Get the logged training data as a Pandas DataFrame.

        Returns
        ----------
        logged_data : pandas.DataFrame
        """
        data = self.logged_data
        cols = ["prob", "cost"]
        for action in self._get_actions():
            cols.append(f"action__{action}")
        for i in range(data.shape[1] - len(cols)):
            cols.append(f"context__{i}")
        return pd.DataFrame(self.logged_data, columns=cols)
