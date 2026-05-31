# ============================================================
# rakaat_fsm.py — Prayer State Machine
# Counts rakaat only when the correct posture SEQUENCE
# is completed in order. Handles edge cases robustly.
# ============================================================

import time
from config import RAKAAT_SEQUENCE, POSTURE_HOLD_TIME


class RakaatFSM:
    """
    Finite State Machine that tracks prayer sequences.

    A rakaat is counted ONLY when the full sequence:
      QIYAM -> RUKU -> QIYAM -> SUJUD -> TASHAHHUD -> SUJUD
    is completed with each posture held for POSTURE_HOLD_TIME seconds.

    The final SUJUD transitions back to QIYAM (start of next rakaat).
    """

    def __init__(self, prayer_name="Custom", total_rakaat=99):
        self.frame_log = []   # per-frame predictions for evaluation
        self.prayer_name    = prayer_name
        self.total_rakaat   = total_rakaat

        # Core state
        self.rakaat_count   = 0
        self.seq_index      = 0          # position in RAKAAT_SEQUENCE

        # Posture hold tracking
        self.current_posture    = None
        self.posture_start_time = None
        self.last_confirmed     = None   # last posture that was locked in

        # Session log
        self.session_log = []            # list of {event, posture, time, rakaat}
        self.session_start = time.time()
        self.is_complete    = False

    # ── Public API ───────────────────────────────────────────

    def update(self, detected_posture):
        """
        Call every frame with the current detected posture.
        Returns True if a rakaat was just completed.
        """
        if self.is_complete:
            return False

        now = time.time()
        just_completed = False

        # If posture changed, reset hold timer
        if detected_posture != self.current_posture:
            self.current_posture    = detected_posture
            self.posture_start_time = now
            return False

        # Check hold duration
        held = now - self.posture_start_time
        if held < POSTURE_HOLD_TIME:
            return False

        # Already confirmed this posture — don't re-process
        if detected_posture == self.last_confirmed:
            return False

        # Check if this matches the next expected step
        expected = RAKAAT_SEQUENCE[self.seq_index]
        if detected_posture == expected:
            self.last_confirmed = detected_posture
            self._log("POSTURE_CONFIRMED", detected_posture)
            self.seq_index += 1

            # Full rakaat completed
            if self.seq_index >= len(RAKAAT_SEQUENCE):
                self.rakaat_count += 1
                self.seq_index = 0
                self.last_confirmed = None
                just_completed = True
                self._log("RAKAAT_COMPLETE", detected_posture)

                if self.rakaat_count >= self.total_rakaat:
                    self.is_complete = True
                    self._log("PRAYER_COMPLETE", detected_posture)

        return just_completed

    def get_state(self):
        """Snapshot of current FSM state for the UI."""
        progress_pct = (self.seq_index / len(RAKAAT_SEQUENCE)) * 100
        return {
            "rakaat":           self.rakaat_count,
            "total_rakaat":     self.total_rakaat,
            "prayer_name":      self.prayer_name,
            "current_posture":  self.current_posture or "UNKNOWN",
            "next_expected":    RAKAAT_SEQUENCE[self.seq_index],
            "seq_index":        self.seq_index,
            "seq_total":        len(RAKAAT_SEQUENCE),
            "progress_pct":     progress_pct,
            "hold_progress":    self._hold_progress(),
            "is_complete":      self.is_complete,
            "elapsed":          int(time.time() - self.session_start),
        }

    def get_summary(self):
        """Post-prayer summary for display."""
        elapsed = time.time() - self.session_start
        mins, secs = divmod(int(elapsed), 60)
        return {
            "prayer_name":  self.prayer_name,
            "rakaat_done":  self.rakaat_count,
            "target":       self.total_rakaat,
            "completed":    self.is_complete,
            "duration":     f"{mins:02d}:{secs:02d}",
            "log":          self.session_log,
            "frame_log": self.frame_log,
        }

    def reset(self, prayer_name=None, total_rakaat=None):
        """Full reset, optionally changing prayer type."""
        self.__init__(
            prayer_name  = prayer_name  or self.prayer_name,
            total_rakaat = total_rakaat or self.total_rakaat,
        )

    # ── Private ──────────────────────────────────────────────

    def _hold_progress(self):
        """0.0–1.0 progress of current posture hold."""
        if self.posture_start_time is None:
            return 0.0
        held = time.time() - self.posture_start_time
        return min(held / POSTURE_HOLD_TIME, 1.0)

    def _log(self, event, posture):
        self.session_log.append({
            "event":   event,
            "posture": posture,
            "rakaat":  self.rakaat_count,
            "time":    round(time.time() - self.session_start, 2),
        })