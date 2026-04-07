import pytest

from grasp_gui_v2 import GUIState, StateMachine


EXPECTED_RULES = {
    GUIState.STARTUP: {"init": False, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": False},
    GUIState.IDLE: {"init": True, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": True},
    GUIState.INITIALIZING: {"init": False, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": False},
    GUIState.READY: {"init": True, "pre_grasp": True, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": True},
    GUIState.PREVIEW: {"init": False, "pre_grasp": False, "confirm": True, "replan": True, "cancel_preview": True, "pause": False, "resume": False, "stop": False, "object_select": False},
    GUIState.GRASPING: {"init": False, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": True, "resume": False, "stop": True, "object_select": False},
    GUIState.PAUSED: {"init": False, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": True, "stop": True, "object_select": False},
    GUIState.STOPPING: {"init": False, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": False},
    GUIState.FAULT: {"init": True, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": False},
    GUIState.CLOSING: {"init": False, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": False},
}

_EXPECTED_KEYS = {"init", "pre_grasp", "confirm", "replan", "cancel_preview", "pause", "resume", "stop", "object_select"}


def test_state_enum_matches_required_workflow():
    assert [state.name for state in GUIState] == [
        "STARTUP",
        "IDLE",
        "INITIALIZING",
        "READY",
        "PREVIEW",
        "GRASPING",
        "PAUSED",
        "STOPPING",
        "FAULT",
        "CLOSING",
    ]


@pytest.mark.parametrize("state", list(GUIState))
def test_button_state_rules_exact(state):
    machine = StateMachine(initial_state=state)
    button_states = machine.get_button_states()
    assert set(button_states.keys()) == _EXPECTED_KEYS
    assert button_states == EXPECTED_RULES[state]


def test_transition_and_previous_state_tracking():
    machine = StateMachine(initial_state=GUIState.STARTUP)
    assert machine.transition_to(GUIState.IDLE) is True
    assert machine.previous_state == GUIState.STARTUP
    assert machine.current_state == GUIState.IDLE

    assert machine.transition_to(GUIState.READY) is True
    assert machine.previous_state == GUIState.IDLE
    assert machine.current_state == GUIState.READY


def test_idle_cannot_transition_directly_to_grasping():
    machine = StateMachine(initial_state=GUIState.IDLE)
    assert machine.transition_to(GUIState.GRASPING) is False
    assert machine.current_state == GUIState.IDLE


def test_invalid_transition_rejected():
    machine = StateMachine(initial_state=GUIState.STARTUP)
    assert machine.transition_to(GUIState.GRASPING) is False
    assert machine.current_state == GUIState.STARTUP


def test_force_state_always_updates():
    machine = StateMachine(initial_state=GUIState.STARTUP)
    machine.force_state(GUIState.CLOSING)
    assert machine.current_state == GUIState.CLOSING
