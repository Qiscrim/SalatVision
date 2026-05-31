# ============================================================
# dashboard.py — OpenCV UI Overlay  (redesigned)
# Improvements:
#   - Cleaner layout with better spacing
#   - Sequence bar shows full posture names, not single letters
#   - Debug panel shows ALL values including view mode
#   - Camera angle warning when view is unknown or top-down
#   - Hold bar moved inside posture pill for cleaner look
#   - TRANSITION state shown distinctly in amber
#   - Summary screen improved
# ============================================================

import cv2
import numpy as np
from config import (
    POSTURE_COLORS,
    COLOR_WHITE, COLOR_DARK, COLOR_GOLD,
    COLOR_GREEN, COLOR_GRAY,
    RAKAAT_SEQUENCE,
)

# Full posture labels
POSTURE_LABEL = {
    "QIYAM":      "Qiyam",
    "RUKU":       "Ruku",
    "SUJUD":      "Sujud",
    "TASHAHHUD":  "Tashahhud",
    "TRANSITION": "Moving...",
    "UNKNOWN":    "---",
}

# Subtitle for each posture
POSTURE_SUB = {
    "QIYAM":      "Standing",
    "RUKU":       "Bowing",
    "SUJUD":      "Prostration",
    "TASHAHHUD":  "Sitting",
    "TRANSITION": "Transition",
    "UNKNOWN":    "",
}

# Short sequence step labels (wider — 2 chars)
SEQ_SHORT = ["QI", "RU", "QI", "SU", "TA", "SU"]
SEQ_FULL  = ["Qiyam", "Ruku", "Qiyam", "Sujud", "Tashahhud", "Sujud"]


def _rect(img, x1, y1, x2, y2, color, alpha=0.65, radius=12):
    """Draw a semi-transparent rounded rectangle."""
    overlay = img.copy()
    r = radius
    cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(overlay, (cx,cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)


def _text(img, text, x, y, scale, color, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_dashboard(frame, state, debug_angles=None):
    h, w = frame.shape[:2]

    posture      = state["current_posture"]
    rakaat       = state["rakaat"]
    next_exp     = state["next_expected"]
    seq_idx      = state["seq_index"]
    hold_prog    = state["hold_progress"]
    prayer_name  = state["prayer_name"]
    total_rak    = state["total_rakaat"]
    elapsed      = state["elapsed"]
    mins, secs   = divmod(elapsed, 60)
    p_color      = POSTURE_COLORS.get(posture, COLOR_GRAY)
    total_str    = str(total_rak) if total_rak < 99 else "∞"

    # ── TOP BAR ─────────────────────────────────────────────
    _rect(frame, 0, 0, w, 64, (12, 12, 20), alpha=0.82, radius=0)
    _text(frame, "SALAT TRACKER", 18, 44,
          0.95, COLOR_GOLD, 2, cv2.FONT_HERSHEY_DUPLEX)
    # Prayer name — centred
    label_w, _ = cv2.getTextSize(f"{prayer_name} Prayer",
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    _text(frame, f"{prayer_name} Prayer",
          w//2 - label_w//2, 44, 0.9, COLOR_WHITE, 2)
    # Timer — right
    _text(frame, f"{mins:02d}:{secs:02d}",
          w - 110, 44, 0.85, COLOR_GRAY, 2)

    # ── RAKAAT COUNTER (top-left block) ─────────────────────
    _rect(frame, 18, 80, 220, 220, (12, 12, 20), alpha=0.82, radius=16)
    rak_str = str(rakaat)
    (rw, _), _ = cv2.getTextSize(rak_str, cv2.FONT_HERSHEY_DUPLEX, 4.0, 8)
    cv2.putText(frame, rak_str, (18 + (202-rw)//2, 192),
                cv2.FONT_HERSHEY_DUPLEX, 4.0, COLOR_GREEN, 8, cv2.LINE_AA)
    cv2.putText(frame, rak_str, (18 + (202-rw)//2, 192),
                cv2.FONT_HERSHEY_DUPLEX, 4.0, (210, 255, 220), 2, cv2.LINE_AA)
    _text(frame, f"of {total_str} rakaat", 30, 215, 0.55, COLOR_GRAY, 1)

    # ── CURRENT POSTURE BLOCK (top-centre) ──────────────────
    px1, px2 = 240, 240 + 340
    _rect(frame, px1, 80, px2, 175, p_color, alpha=0.75, radius=14)

    main_lbl = POSTURE_LABEL.get(posture, posture)
    sub_lbl  = POSTURE_SUB.get(posture, "")
    (mw, _), _ = cv2.getTextSize(main_lbl, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    mid = px1 + (px2-px1)//2
    cv2.putText(frame, main_lbl, (mid - mw//2, 130),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (10,10,10), 2, cv2.LINE_AA)
    if sub_lbl:
        (sw, _), _ = cv2.getTextSize(sub_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        _text(frame, sub_lbl, mid - sw//2, 162, 0.7, (30,30,30), 1)

    # ── HOLD BAR (below posture block) ──────────────────────
    bx1, bx2 = px1, px2
    by1, by2 = 182, 202
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30,30,40), -1)
    fill = int((bx2-bx1) * hold_prog)
    if fill > 0:
        cv2.rectangle(frame, (bx1, by1), (bx1+fill, by2), p_color, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (70,70,80), 1)
    pct = int(hold_prog * 100)
    _text(frame, f"Hold  {pct}%", bx1+6, by2-4, 0.45, COLOR_WHITE, 1)

    # ── NEXT POSTURE (top-right) ─────────────────────────────
    nx1 = w - 290
    _rect(frame, nx1, 80, w-18, 175, (12,12,20), alpha=0.80, radius=14)
    _text(frame, "Next:", nx1+12, 110, 0.6, COLOR_GRAY, 1)
    next_color = POSTURE_COLORS.get(next_exp, COLOR_GRAY)
    next_main  = POSTURE_LABEL.get(next_exp, next_exp)
    next_sub   = POSTURE_SUB.get(next_exp, "")
    _text(frame, next_main,  nx1+12, 142, 0.85, next_color, 2)
    if next_sub:
        _text(frame, f"({next_sub})", nx1+12, 168, 0.6, next_color, 1)

    # ── SEQUENCE BAR (bottom) ────────────────────────────────
    # One pill per step, full short name, evenly spaced
    n_steps  = len(SEQ_FULL)
    bar_y1   = h - 70
    bar_y2   = h - 18
    pill_w   = (w - 40) // n_steps
    gap      = 6

    for i, (short, full) in enumerate(zip(SEQ_SHORT, SEQ_FULL)):
        px  = 20 + i * pill_w
        pw  = pill_w - gap
        pmid = px + pw // 2
        pmy  = (bar_y1 + bar_y2) // 2

        if i < seq_idx:
            # Completed — solid green
            _rect(frame, px, bar_y1, px+pw, bar_y2,
                  COLOR_GREEN, alpha=0.85, radius=8)
            _text(frame, short, pmid-14, pmy+6, 0.55, (10,10,10), 2)
        elif i == seq_idx:
            # Current — posture colour + gold border
            _rect(frame, px, bar_y1, px+pw, bar_y2,
                  POSTURE_COLORS.get(RAKAAT_SEQUENCE[i], COLOR_GRAY),
                  alpha=0.9, radius=8)
            cv2.rectangle(frame, (px, bar_y1), (px+pw, bar_y2),
                          COLOR_GOLD, 2)
            _text(frame, short, pmid-14, pmy+6, 0.6, COLOR_WHITE, 2)
            # Full name below pill
            (fw,_),_ = cv2.getTextSize(full, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            _text(frame, full, pmid-fw//2, bar_y2+14, 0.42, COLOR_GOLD, 1)
        else:
            # Future — dim
            _rect(frame, px, bar_y1, px+pw, bar_y2,
                  (35,35,45), alpha=0.7, radius=8)
            _text(frame, short, pmid-14, pmy+6, 0.5, (90,90,100), 1)

    # ── CAMERA ANGLE WARNING ─────────────────────────────────
    if debug_angles:
        view = debug_angles.get("view", "")
        if view == "unknown":
            _rect(frame, 18, 230, 430, 268, (0,30,80), alpha=0.85, radius=8)
            _text(frame, "⚠  Camera angle unclear — try side view",
                  28, 255, 0.52, (0,200,255), 1)

    # ── DEBUG PANEL ──────────────────────────────────────────
    if debug_angles:
        # Panel sits on the LEFT side below the rakaat counter
        # so it does not overlap the Next posture block on the right.
        # Larger font, wider panel, more row spacing for readability.
        panel_x  = 18          # left edge
        panel_y  = 230         # below rakaat counter
        row_h    = 28          # pixels per row
        font_sc  = 0.58        # readable at 1280px width
        col_w    = 320         # panel width

        rows = []
        for k, v in debug_angles.items():
            if isinstance(v, float):
                rows.append((k, f"{v:.1f}"))
            elif isinstance(v, bool):
                rows.append((k, "yes" if v else "no"))
            elif isinstance(v, str):
                rows.append((k, v))

        panel_h = 36 + len(rows) * row_h
        _rect(frame, panel_x, panel_y,
              panel_x + col_w, panel_y + panel_h,
              (12, 12, 20), alpha=0.85, radius=10)

        # Header
        _text(frame, "DEBUG", panel_x + 10, panel_y + 22,
              0.65, COLOR_GOLD, 2)

        # Rows — key in gray, value in white for easy scanning
        for j, (key, val) in enumerate(rows):
            y = panel_y + 44 + j * row_h
            _text(frame, f"{key}", panel_x + 10, y,
                  font_sc, COLOR_GRAY, 1, cv2.FONT_HERSHEY_SIMPLEX)
            _text(frame, val, panel_x + 190, y,
                  font_sc, COLOR_WHITE, 1, cv2.FONT_HERSHEY_SIMPLEX)

    # ── PRAYER COMPLETE BANNER ───────────────────────────────
    if state.get("is_complete"):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2-80), (w, h//2+80), (8,35,8), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        msg = "Prayer Complete  —  Alhamdulillah"
        (mw,_),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)
        cv2.putText(frame, msg, (w//2 - mw//2, h//2+18),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, COLOR_GREEN, 3, cv2.LINE_AA)

    # ── FPS + AUDIO drawn by main.py ─────────────────────────
    return frame


def draw_summary_screen(frame, summary):
    """Full-screen post-prayer summary overlay."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,h), (8,12,8), -1)
    cv2.addWeighted(overlay, 0.93, frame, 0.07, 0, frame)

    # Title
    _text(frame, "Prayer Summary",
          w//2 - 210, 90, 1.8, COLOR_GOLD, 3, cv2.FONT_HERSHEY_DUPLEX)

    # Divider
    cv2.line(frame, (w//2-300, 110), (w//2+300, 110), COLOR_GOLD, 1)

    # Stats
    status_color = COLOR_GREEN if summary["completed"] else (60,60,220)
    status_text  = "Complete ✓" if summary["completed"] else "Incomplete"
    lines = [
        ("Prayer",   summary["prayer_name"],                  COLOR_WHITE),
        ("Rakaat",   f"{summary['rakaat_done']} / {summary['target']}", COLOR_GREEN),
        ("Duration", summary["duration"],                     COLOR_WHITE),
        ("Status",   status_text,                             status_color),
    ]
    for i, (lbl, val, col) in enumerate(lines):
        y = 190 + i * 80
        _text(frame, f"{lbl}:", w//2-280, y, 0.9, COLOR_GRAY, 1)
        _text(frame, val,       w//2-60,  y, 1.05, col, 2)

    cv2.line(frame, (w//2-300, h-100), (w//2+300, h-100), (60,60,70), 1)
    _text(frame, "Q — exit      R — new prayer",
          w//2-220, h-55, 0.8, COLOR_GRAY, 1)

    return frame