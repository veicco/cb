from cb.discrete_actions import ContextualBanditDiscreteActionsModel


def test_get_actions():
    actions = ContextualBanditDiscreteActionsModel(actions=[0, 1, 2])._get_actions()
    assert actions == [0, 1, 2]
