import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Tuple, Optional
from sklearn.base import RegressorMixin
from cb.base import ContextualBanditBase, Action, Cost, Prob


class ContextualBanditContinuousActionsModel(ContextualBanditBase):
    """
    Contextual Bandit (1-step reinforcement learning) model with
    continuous action space. Learns a policy from (context, action, cost)
    triplets to choose an action so as to minimize the expected cost given the
    context.

    The model discretizes the provided action space between the minimum and
    the maximum and reduces to a supervised regression task where the cost
    is predicted separately for each action; With the expected cost of
    each action, the policy returns either the least costly action (exploit)
    or one of its neighbours (explore).

    Parameters
    ----------
    min_value : int or float
        Minimum value of the action space.

    max_value : int or float
        Maximum value of the action space.

    action_width : int or float
        Distance between two discretized actions.

    memory : int
        Maximum number of logged data used for learning the policy and kept in memory.

    initial_action : int or float
        The first action to start with when no training data has yet
        been logged.

    data_file : str
        Path to the file where the logged data is stored. If no file is provided, the data exceeding
        the memory limit will be forgotten. The provided file can be empty or it can contain
        existing logged data (for warm start) in the following csv format:
            prob,
            cost,
            ...action(s) (1 column if self.categorize_actions == False, otherwise N of actions columns),
            ...context : (one-hot encoded features)

    regression_model : default sklearn.linear_model.LinearRegression()
        Regression model used for learning to predict costs given action +
        context. Must be an instance of a regression model with the Scikit
        learn API, such as sklearn.linear_model.GradientBoostingRegressor().

    categorize_actions : boolean
        An option to define whether the actions in the feature array are
        treated as numerical/continuous values or one-hot encoded dummies.
         => True: one-hot encoded values.
         => False: continuous feature.

    decay_rate : float
        Exponent factor to control how quickly the logged data decays causing
        earlier samples to have less weight.
    """

    def __init__(
        self,
        min_value: Action,
        max_value: Action,
        action_width: Action,
        memory: int = None,
        initial_action: Action = None,
        data_file: str = None,
        regression_model: RegressorMixin = None,
        categorize_actions: bool = False,
        decay_rate: float = 1.0,
    ):
        super(ContextualBanditContinuousActionsModel, self).__init__(
            memory, initial_action, data_file, regression_model, decay_rate
        )
        self.min_value = min_value
        self.max_value = max_value
        self.action_width = action_width
        self.categorize_actions = categorize_actions

    def _get_actions(self) -> List[Action]:
        num_actions = int((self.max_value - self.min_value) / self.action_width + 1)
        return [self.min_value + i * self.action_width for i in range(num_actions)]

    def _init_regressor(self, context: np.ndarray):
        if self.regression_model is not None:
            self.reg = self.regression_model
        else:
            self.reg = LinearRegression()
        action = 0
        if self.categorize_actions:
            action = self._get_actions_one_hot(self._get_actions())
        x = np.append(action, context).reshape(1, -1)
        cost = np.array([1])
        self.reg.fit(x, cost)

    def _log_example(self, context: np.ndarray, action: Action, cost: Cost, prob: Prob):
        data = self.logged_data
        a = action
        if self.categorize_actions:
            a = self._get_actions_one_hot(self._get_actions(), action)
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
        self,
        costs_per_action: Dict[Action, Cost],
        epsilon: Prob,
        exploration_width: int,
        direction: str = None,
    ) -> Tuple[Action, Prob]:
        actions = self._get_actions()
        best_action = min(costs_per_action, key=costs_per_action.get)  # type: ignore
        best_idx = actions.index(best_action)
        if direction == "left":
            neighbours_idx = np.arange(best_idx - exploration_width, best_idx)
        elif direction == "right":
            neighbours_idx = np.arange(best_idx + 1, best_idx + exploration_width + 1)
        else:
            neighbours_idx = np.append(
                np.arange(best_idx - exploration_width, best_idx),
                np.arange(best_idx + 1, best_idx + exploration_width + 1),
            )
        possible_idx = neighbours_idx[
            (neighbours_idx >= 0) & (neighbours_idx < len(actions))
        ]
        if len(possible_idx) == 0:
            return best_action, epsilon
        possible_actions = [actions[idx] for idx in possible_idx]
        costs_per_possible_action = {
            key: costs_per_action[key] for key in possible_actions
        }
        return self._sample_action(costs_per_possible_action, epsilon)

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
            if self.categorize_actions:
                action_one_hot = self._get_actions_one_hot(self._get_actions(), action)
                x = np.append(action_one_hot, context)
            else:
                x = np.append(action, context)
            costs_per_action[action] = self.reg.predict(x.reshape(1, -1)).reshape(-1)[0]
        return costs_per_action

    def predict(
        self,
        context: np.ndarray,
        epsilon: Prob = 0.05,
        exploration_width: int = 1,
        exploration_strategy: str = "smart",
    ) -> Tuple[Action, Prob]:
        """
        Predict an action given a context.

        Parameters
        ----------
        context : np.array([...])
            Context/feature set that an action is predicted for.

        epsilon : float between 0.0 and 1.0
            Probability of exploration, that is, the probability that a
            suboptimal action is returned instead of the best known action.

        exploration_width : int between 1 and N of actions
            Defines the maximum deviation from the optimum when exploring.
            For instance, exploration_width=2 allows exploring with actions
            that are not smaller the optimal action - 2 * action_width and
            not larger than the optimal action + 2 * action_width.

        exploration_strategy : str (smart | aggressive | None)
            If 'smart' exploration strategy is used, the model keeps track if
            the previous exploration improved the result and if so it keeps
            exploring to the same direction until no improvement happens.

            'aggressive' stragegy is similar to 'smart' except that it explores
            after every step where the cost improved, whether the previous step
            was exploration or exploitation.

        Returns
        ----------
        (action, prob) : tuple
            The predicted action with the probability that it was selected.
        """

        def _get_direction(action_change: Action) -> Optional[str]:
            if action_change < 0:
                return "left"
            elif action_change > 0:
                return "right"

        assert exploration_strategy in [
            "smart",
            "aggressive",
            None,
        ], "Exploration strategy must be 'smart', 'aggressive' or None"

        if self.reg is None:
            self._init_regressor(context)
            if self.initial_action:
                closest_action = min(
                    self._get_actions(), key=lambda x: abs(x - self.initial_action)  # type: ignore
                )
                return closest_action, 1.0
        costs_per_action = self.get_costs_per_action(context)
        if exploration_strategy == "smart":
            explored, cost_change, action_change = self._get_previous_move(epsilon)
            if explored and cost_change < 0:
                direction = _get_direction(action_change)
                return self._explore(
                    costs_per_action, 1.0, exploration_width, direction
                )
        if exploration_strategy == "aggressive":
            explored, cost_change, action_change = self._get_previous_move(epsilon)
            if cost_change < 0:
                direction = _get_direction(action_change)
                return self._explore(
                    costs_per_action, 1.0, exploration_width, direction
                )
        if np.random.random() < epsilon:
            return self._explore(costs_per_action, epsilon, exploration_width)
        return self._exploit(costs_per_action, epsilon)

    def get_logged_data_df(self) -> pd.DataFrame:
        """
        Get the logged training data as a Pandas DataFrame.

        Returns
        ----------
        logged_data : pandas.DataFrame
        """
        data = self.logged_data
        cols = ["prob", "cost"]
        if self.categorize_actions:
            for action in self._get_actions():
                cols.append(f"action__{action}")
        else:
            cols.append("action")
        for i in range(data.shape[1] - len(cols)):
            cols.append(f"context__{i}")
        return pd.DataFrame(self.logged_data, columns=cols)
