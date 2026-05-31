# ============================================================
# tests/test_fsm.py - Unit tests for RakaatFSM
# Run: python -m pytest tests/ -v
# ============================================================

import sys
import time
sys.path.insert(0, "..")

from rakaat_fsm import RakaatFSM
from config import RAKAAT_SEQUENCE, POSTURE_HOLD_TIME


def feed_posture(fsm, posture, duration=None):
    """Simulate holding a posture for long enough to register."""
    if duration is None:
        duration = POSTURE_HOLD_TIME + 0.1
    fsm.current_posture    = posture
    fsm.posture_start_time = time.time() - duration
    fsm.last_confirmed     = None
    return fsm.update(posture)


def complete_one_rakaat(fsm):
    """Feed a full valid sequence."""
    completed = False
    for posture in RAKAAT_SEQUENCE:
        result = feed_posture(fsm, posture)
        if result:
            completed = True
    return completed


class TestRakaatFSM:

    def test_initial_state(self):
        fsm = RakaatFSM()
        assert fsm.rakaat_count == 0
        assert fsm.seq_index == 0

    def test_one_full_rakaat(self):
        fsm = RakaatFSM("Test", 4)
        completed = complete_one_rakaat(fsm)
        assert completed is True
        assert fsm.rakaat_count == 1

    def test_two_full_rakaat(self):
        fsm = RakaatFSM("Test", 4)
        complete_one_rakaat(fsm)
        complete_one_rakaat(fsm)
        assert fsm.rakaat_count == 2

    def test_wrong_sequence_does_not_count(self):
        fsm = RakaatFSM("Test", 4)
        # Feed out-of-order postures
        feed_posture(fsm, "RUKU")    # Wrong — expected QIYAM first
        feed_posture(fsm, "SUJUD")
        assert fsm.rakaat_count == 0
        assert fsm.seq_index == 0

    def test_posture_hold_too_short_does_not_register(self):
        fsm = RakaatFSM("Test", 4)
        # Feed with hold time shorter than threshold
        fsm.current_posture    = "QIYAM"
        fsm.posture_start_time = time.time() - (POSTURE_HOLD_TIME * 0.3)
        result = fsm.update("QIYAM")
        assert result is False
        assert fsm.seq_index == 0

    def test_sequence_index_advances(self):
        fsm = RakaatFSM("Test", 4)
        for i, posture in enumerate(RAKAAT_SEQUENCE[:-1]):
            feed_posture(fsm, posture)
            assert fsm.seq_index == i + 1

    def test_fajr_two_rakaat_completes(self):
        fsm = RakaatFSM("Fajr", 2)
        complete_one_rakaat(fsm)
        complete_one_rakaat(fsm)
        assert fsm.is_complete is True
        assert fsm.rakaat_count == 2

    def test_extra_rakaat_after_complete_ignored(self):
        fsm = RakaatFSM("Fajr", 2)
        complete_one_rakaat(fsm)
        complete_one_rakaat(fsm)
        complete_one_rakaat(fsm)   # Extra
        assert fsm.rakaat_count == 2   # Still 2

    def test_reset_clears_state(self):
        fsm = RakaatFSM("Test", 4)
        complete_one_rakaat(fsm)
        fsm.reset("Zuhr", 4)
        assert fsm.rakaat_count == 0
        assert fsm.seq_index == 0
        assert fsm.prayer_name == "Zuhr"

    def test_session_log_populated(self):
        fsm = RakaatFSM("Test", 4)
        complete_one_rakaat(fsm)
        events = [e["event"] for e in fsm.session_log]
        assert "POSTURE_CONFIRMED" in events
        assert "RAKAAT_COMPLETE"   in events

    def test_summary_structure(self):
        fsm = RakaatFSM("Zuhr", 4)
        summary = fsm.get_summary()
        assert "prayer_name" in summary
        assert "rakaat_done" in summary
        assert "duration"    in summary
        assert "completed"   in summary


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])