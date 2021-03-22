import unittest
import numpy as np
from numpy.testing import assert_array_equal
from cacb.cacb import ContinuousActionContextualBanditModel


np.random.seed(123)


class ContinuousActionContextualBanditModelTest(unittest.TestCase):
    def test_get_actions(self):
        actions1 = ContinuousActionContextualBanditModel(
            min_value=10,
            max_value=15,
            action_width=1,
        )._get_actions()
        self.assertListEqual(actions1, [10, 11, 12, 13, 14, 15])

        actions2 = ContinuousActionContextualBanditModel(
            min_value=-10,
            max_value=-5,
            action_width=1,
        )._get_actions()
        self.assertListEqual(actions2, [-10, -9, -8, -7, -6, -5])

        actions3 = ContinuousActionContextualBanditModel(
            min_value=10,
            max_value=15,
            action_width=2,
        )._get_actions()
        self.assertListEqual(actions3, [10, 12, 14])

    def test_get_actions_one_hot(self):
        actions_one_hot = ContinuousActionContextualBanditModel(
            min_value=10,
            max_value=15,
            action_width=1,
        )._get_actions_one_hot(12)
        assert_array_equal(actions_one_hot, np.array([0, 0, 1, 0, 0, 0]))

    def test_log_example(self):
        context = np.array([1, 2, 3])
        action = 11
        cost = 100
        prob = 0.75
        
        # uncategorized actions
        cacb_1 = ContinuousActionContextualBanditModel(
            min_value=10,
            max_value=15,
            action_width=1,
            categorize_actions=False
        )
        cacb_1._log_example(context, action, cost, prob)
        assert_array_equal(
            cacb_1.logged_data,
            np.array([0.75, 100, 11, 1, 2, 3]).reshape(1, -1)
        )

        # categorized actions
        cacb_2 = ContinuousActionContextualBanditModel(
            min_value=10,
            max_value=15,
            action_width=1,
            categorize_actions=True
        )
        cacb_2._log_example(context, action, cost, prob)
        assert_array_equal(
            cacb_2.logged_data,
            np.array([0.75, 100, 0, 1, 0, 0, 0, 0, 1, 2, 3]).reshape(1, -1)
        )

        # log another example
        cacb_2._log_example(context, 10, 90, 0.90)
        assert_array_equal(
            cacb_2.logged_data,
            np.array([
                np.array([0.75, 100, 0, 1, 0, 0, 0, 0, 1, 2, 3]),
                np.array([0.90, 90, 1, 0, 0, 0, 0, 0, 1, 2, 3])
            ])
            
        )

    def test_exploit(self):
        cacb = ContinuousActionContextualBanditModel(
            min_value=10,
            max_value=15,
            action_width=1,
        )

        # lowest cost exists
        costs_per_action = {
            10: 100,
            11: 100,
            12: 90,
            13: 100,
            14: 100,
            15: 100
        }
        epsilon = 0.10
        action, prob = cacb._exploit(costs_per_action, epsilon)
        self.assertEqual(action, 12)
        self.assertEqual(prob, 0.90)

        # no clear winner => should choose the first of the best
        costs_per_action = {
            10: 100,
            11: 90,
            12: 100,
            13: 90,
            14: 90,
            15: 100
        }
        epsilon = 0.10
        action, prob = cacb._exploit(costs_per_action, epsilon)
        self.assertEqual(action, 11)
        self.assertEqual(prob, 0.90)

    def test_explore(self):
        cacb = ContinuousActionContextualBanditModel(
            min_value=10,
            max_value=15,
            action_width=1,
        )
        costs_per_action = {
            10: 100,
            11: 100,
            12: 90,
            13: 100,
            14: 100,
            15: 100
        }
        
        # exploration width = 1
        epsilon = 0.10
        exploration_width=1
        action, prob = cacb._explore(costs_per_action, epsilon, exploration_width)
        self.assertEqual(prob, 0.05)
        self.assertEqual(action, 13)

        # exploration width = 2
        epsilon = 0.10
        exploration_width=2
        action, prob = cacb._explore(costs_per_action, epsilon, exploration_width)
        self.assertEqual(prob, 0.025)
        self.assertEqual(action, 11)

        # exploration width = 1, optimum in the end
        costs_per_action = {
            10: 90,
            11: 100,
            12: 100,
            13: 100,
            14: 100,
            15: 100
        }
        epsilon = 0.10
        exploration_width=1
        action, prob = cacb._explore(costs_per_action, epsilon, exploration_width)
        self.assertEqual(prob, 0.10)
        self.assertEqual(action, 11)

