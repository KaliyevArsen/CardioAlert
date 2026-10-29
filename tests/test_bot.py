import pytest

from cardioalert.bot import Conversation
from cardioalert.schema import FEATURES


def test_complete_conversation_clears_data_and_shows_limits(artifact, record):
    flow = Conversation(artifact)
    assert "fictional" in flow.message(1, "/start")
    for name in FEATURES:
        result = flow.message(1, str(record[name]))
    assert "Research model score" in result
    assert "not a diagnosis" in result
    assert 1 not in flow.sessions


def test_users_are_isolated_and_cancel_is_local(artifact):
    flow = Conversation(artifact)
    flow.message(1, "/start")
    flow.message(2, "/start")
    flow.message(1, "55")
    assert flow.sessions[2].values == {}
    flow.message(1, "/cancel")
    assert 1 not in flow.sessions and 2 in flow.sessions


def test_expiry_boundary_and_active_session_refresh(artifact):
    now = [0.0]
    flow = Conversation(artifact, ttl_seconds=10, clock=lambda: now[0])
    flow.message(1, "/start")
    now[0] = 9
    flow.message(1, "55")
    now[0] = 18
    flow.expire()
    assert 1 in flow.sessions
    now[0] = 19
    flow.expire()
    assert not flow.sessions


@pytest.mark.parametrize("value", ["hello", "NaN", "inf", "-5", "130"])
def test_invalid_input_does_not_advance(artifact, value):
    flow = Conversation(artifact)
    flow.message(1, "/start")
    assert "Invalid input" in flow.message(1, value)
    assert flow.sessions[1].values == {}


def test_prediction_failure_still_clears_session(artifact, record, monkeypatch):
    def fail(*args):
        raise RuntimeError("failed inference")
    monkeypatch.setattr("cardioalert.bot.predict_record", fail)
    flow = Conversation(artifact)
    flow.message(1, "/start")
    for name in FEATURES[:-1]:
        flow.message(1, str(record[name]))
    with pytest.raises(RuntimeError):
        flow.message(1, str(record[FEATURES[-1]]))
    assert not flow.sessions
