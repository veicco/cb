from typing import List
from sklearn.base import RegressorMixin
from cb.base import ContextualBanditBase, Action


class ContextualBanditDiscreteActionsModel(ContextualBanditBase):
    """
    Contextual Bandit (1-step reinforcement learning) model with
    discrete action space. Learns a policy from (context, action, cost) triplets to
    choose an action so as to minimize the expected cost given the context.

    The model reduces to a supervised regression task where the cost
    is predicted separately for each action; With the expected cost of
    each action, the policy returns either the least costly action (exploit)
    or a randomly chosen action (explore).

    Parameters
    ----------
    actions : array of int or float
        Values of possible actions.

    memory : int
        Maximum number of logged data used for learning the policy and kept in memory.

    initial_action : int
        The first action to start with when no training data has yet
        been logged.

    data_file : str
        Path to the file where the logged data is stored. If no file is provided, the data exceeding
        the memory limit will be forgotten. The provided file can be empty or it can contain
        existing logged data (for warm start) in the following csv format:
            prob,
            cost,
            ...actions (N of actions columns),
            ...context : (one-hot encoded features)

    regression_model : default sklearn.linear_model.LinearRegression()
        Regression model used for learning to predict costs given action +
        context. Must be an instance of a regression model with the Scikit
        learn API, such as sklearn.linear_model.GradientBoostingRegressor().

    decay_rate : float
        Exponent factor to control how quickly the logged data decays causing
        earlier samples to have less weight.
    """

    def __init__(
        self,
        actions: List[Action],
        memory: int = None,
        initial_action: Action = None,
        data_file: str = None,
        regression_model: RegressorMixin = None,
        decay_rate: float = 1.0,
    ):
        self.actions = actions
        super(ContextualBanditDiscreteActionsModel, self).__init__(
            memory, initial_action, data_file, regression_model, decay_rate
        )

    def _get_actions(self) -> List[Action]:
        return self.actions
