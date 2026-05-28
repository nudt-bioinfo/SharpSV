import math
import os
import shutil
import sys
import textwrap
import time


_MIN_FRAME_WIDTH = 64
_FALLBACK_FRAME_WIDTH = 48
_MAX_FRAME_WIDTH = 126
_COMPACT_BREAKPOINT = 104
_DENSE_BREAKPOINT = 82

_STAGE_META = {
    "pipeline": {"icon": "◆", "label": "PIPELINE", "compact": "PIPE", "theme": "pipeline"},
    "stage-1/features": {"icon": "◫", "label": "STAGE-1/FEATURES", "compact": "S1/FEAT", "theme": "stage-1/features"},
    "stage-1": {"icon": "◩", "label": "STAGE-1/SCORE", "compact": "S1/SCORE", "theme": "stage-1"},
    "stage-2": {"icon": "◪", "label": "STAGE-2/REFINE", "compact": "S2/REF", "theme": "stage-2"},
    "stage-2/cpu": {"icon": "◨", "label": "STAGE-2/CPU", "compact": "S2/CPU", "theme": "stage-2/cpu"},
    "stage-3": {"icon": "◬", "label": "STAGE-3/VALIDATE", "compact": "S3/VAL", "theme": "stage-3"},
    "stage-3/sort": {"icon": "◭", "label": "STAGE-3/SORT", "compact": "S3/SORT", "theme": "stage-3"},
    "stage-3/assemble": {"icon": "◮", "label": "STAGE-3/ASM", "compact": "S3/ASM", "theme": "stage-3"},
    "stage-3/merge": {"icon": "◰", "label": "STAGE-3/MERGE", "compact": "S3/MRG", "theme": "stage-3"},
    "stage-3/validate": {"icon": "◱", "label": "STAGE-3/FILTER", "compact": "S3/FILT", "theme": "stage-3"},
    "stage-4": {"icon": "◲", "label": "STAGE-4/VCF", "compact": "S4/VCF", "theme": "stage-4"},
    "stage-4/export": {"icon": "◳", "label": "STAGE-4/EXPORT", "compact": "S4/EXP", "theme": "stage-4"},
    "stage-4/realign": {"icon": "◴", "label": "STAGE-4/REALIGN", "compact": "S4/RLN", "theme": "stage-4"},
}

_TITLE_META = {
    "stage-1 feature synthesis": {
        "icon": "◫",
        "theme": "stage-1/features",
        "subtitle": "Genome-scale NPZ feature synthesis from aligned read evidence",
        "compact_subtitle": "NPZ feature synthesis from aligned reads",
    },
    "stage-1 candidate scoring": {
        "icon": "◩",
        "theme": "stage-1",
        "subtitle": "GPU-aware ranking over compressed feature windows",
        "compact_subtitle": "GPU-aware candidate scoring",
    },
    "stage-2 breakpoint refinement": {
        "icon": "◪",
        "theme": "stage-2",
        "subtitle": "CPU image builders feeding a sequence-aware breakpoint refiner",
        "compact_subtitle": "CPU builders feeding GPU breakpoint refinement",
    },
    "stage-2 refinement": {
        "icon": "◪",
        "theme": "stage-2",
        "subtitle": "CPU image builders feeding a sequence-aware breakpoint refiner",
        "compact_subtitle": "CPU builders feeding GPU breakpoint refinement",
    },
    "stage-3 assembly validation": {
        "icon": "◬",
        "theme": "stage-3",
        "subtitle": "Interval decoding, local assembly, merge, and adaptive validation",
        "compact_subtitle": "Assembly-backed adaptive SV validation",
    },
    "stage-4 vcf finalization": {
        "icon": "◲",
        "theme": "stage-4",
        "subtitle": "VCF export and DEL realignment with all variant types preserved",
        "compact_subtitle": "VCF export and DEL realignment",
    },
}

_PALETTES = {
    "pipeline": {
        "border": "1;38;5;45",
        "title": "1;38;5;51",
        "subtitle": "38;5;117",
        "detail": "38;5;252",
        "log": "38;5;117",
    },
    "stage-1/features": {
        "border": "1;38;5;75",
        "title": "1;38;5;81",
        "subtitle": "38;5;117",
        "detail": "38;5;251",
        "log": "38;5;111",
    },
    "stage-1": {
        "border": "1;38;5;42",
        "title": "1;38;5;48",
        "subtitle": "38;5;84",
        "detail": "38;5;251",
        "log": "38;5;85",
    },
    "stage-2": {
        "border": "1;38;5;214",
        "title": "1;38;5;221",
        "subtitle": "38;5;222",
        "detail": "38;5;252",
        "log": "38;5;222",
    },
    "stage-2/cpu": {
        "border": "1;38;5;178",
        "title": "1;38;5;186",
        "subtitle": "38;5;223",
        "detail": "38;5;251",
        "log": "38;5;186",
    },
    "stage-3": {
        "border": "1;38;5;141",
        "title": "1;38;5;147",
        "subtitle": "38;5;183",
        "detail": "38;5;252",
        "log": "38;5;183",
    },
    "stage-4": {
        "border": "1;38;5;204",
        "title": "1;38;5;210",
        "subtitle": "38;5;218",
        "detail": "38;5;252",
        "log": "38;5;218",
    },
    "default": {
        "border": "1;38;5;244",
        "title": "1;38;5;250",
        "subtitle": "38;5;248",
        "detail": "38;5;252",
        "log": "38;5;250",
    },
}


def _supports_color():
    force = os.environ.get("SHARPSV_COLOR")
    if force == "1":
        return True
    if force == "0" or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR") or os.environ.get("CLICOLOR_FORCE") == "1":
        return True
    return sys.stdout.isatty() and os.environ.get("TERM", "").lower() != "dumb"


def _paint(text, style=None):
    if not style or not _supports_color():
        return text
    return f"\033[{style}m{text}\033[0m"


def _terminal_width():
    try:
        return shutil.get_terminal_size((118, 20)).columns
    except OSError:
        return 118


def _frame_width():
    terminal_width = _terminal_width()
    if terminal_width < _MIN_FRAME_WIDTH:
        return max(terminal_width, _FALLBACK_FRAME_WIDTH)
    return min(terminal_width, _MAX_FRAME_WIDTH)


def _layout_mode():
    width = _frame_width()
    if width <= _DENSE_BREAKPOINT:
        return "dense"
    if width <= _COMPACT_BREAKPOINT:
        return "compact"
    return "standard"


def _content_width():
    return _frame_width() - 4


def _label_width():
    mode = _layout_mode()
    if mode == "dense":
        return 8
    if mode == "compact":
        return 12
    return 18


def _detail_key_width(details):
    if not details:
        return 0
    mode = _layout_mode()
    max_key = max(len(str(key)) for key, _ in details)
    if mode == "dense":
        return min(max_key, 12)
    if mode == "compact":
        return min(max_key, 16)
    return min(max_key, 18)


def _box_top(hero=False):
    width = _frame_width()
    if hero:
        return "╔" + ("═" * (width - 2)) + "╗"
    return "╭" + ("─" * (width - 2)) + "╮"


def _box_mid(hero=False):
    width = _frame_width()
    if hero:
        return "╠" + ("═" * (width - 2)) + "╣"
    return "├" + ("─" * (width - 2)) + "┤"


def _box_bottom(hero=False):
    width = _frame_width()
    if hero:
        return "╚" + ("═" * (width - 2)) + "╝"
    return "╰" + ("─" * (width - 2)) + "╯"


def _pad(text, width):
    text = str(text)
    if len(text) > width:
        text = text[: max(width - 1, 0)] + "…"
    return text.ljust(width)


def _row(text):
    return f"│ {_pad(text, _content_width())} │"


def _wrap_text(text, width):
    lines = []
    for block in str(text).splitlines() or [""]:
        wrapped = textwrap.wrap(
            block,
            width=max(width, 1),
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines or [""]


def _stage_meta(stage):
    meta = _STAGE_META.get(stage)
    if meta:
        return meta
    return {
        "icon": "•",
        "label": str(stage).upper().replace("_", "-"),
        "compact": str(stage).upper().replace("_", "-")[:8],
        "theme": "default",
    }


def _palette(theme):
    return _PALETTES.get(theme, _PALETTES["default"])


def _banner_theme(title):
    normalized = str(title).strip().lower()
    if normalized == "structural variant discovery pipeline":
        return "pipeline"
    return _TITLE_META.get(normalized, {}).get("theme", "default")


def _banner_title_lines(title):
    normalized = str(title).strip()
    mode = _layout_mode()
    if normalized.lower() == "structural variant discovery pipeline":
        if mode == "dense":
            return [
                ("title", "◢██◣ SharpSV"),
                ("subtitle", "SV Discovery Pipeline"),
            ]
        if mode == "compact":
            return [
                ("title", "◢██◣ SharpSV"),
                ("subtitle", "Structural Variant Discovery Pipeline"),
                ("subtitle", "Candidate discovery  ·  GPU-aware refinement"),
            ]
        return [
            ("title", "  ◢██◣  SharpSV"),
            ("title", "  ◥██◤  Structural Variant Discovery Pipeline"),
            ("subtitle", "  ╱██╲  Candidate discovery  ·  breakpoint refinement  ·  GPU-aware inference"),
        ]

    meta = _TITLE_META.get(normalized.lower())
    if meta:
        subtitle = meta["compact_subtitle"] if mode != "standard" else meta["subtitle"]
        lines = [("title", f"{meta['icon']}  SharpSV {normalized}")]
        if subtitle:
            lines.append(("subtitle", f"   {subtitle}"))
        return lines
    return [("title", "◢◤  SharpSV"), ("subtitle", f"◥◣  {normalized}")]


def _detail_lines(details):
    if not details:
        return []

    mode = _layout_mode()
    rows = []
    if mode == "standard":
        key_width = _detail_key_width(details)
        value_width = max(_content_width() - key_width - 5, 20)
        for key, value in details:
            prefix = f"{str(key):<{key_width}} │ "
            wrapped = _wrap_text(value, value_width)
            rows.append(prefix + wrapped[0])
            continuation = " " * len(prefix)
            for line in wrapped[1:]:
                rows.append(continuation + line)
        return rows

    for key, value in details:
        summary = f"{key}: {value}"
        rows.extend(_wrap_text(summary, _content_width()))
    return rows


def emit_banner(title, details=None):
    hero = str(title).strip().lower() == "structural variant discovery pipeline"
    palette = _palette(_banner_theme(title))

    print(_paint(_box_top(hero=hero), palette["border"]), file=sys.stdout, flush=True)
    for style_key, raw_line in _banner_title_lines(title):
        style = palette["title"] if style_key == "title" else palette["subtitle"]
        for line in _wrap_text(raw_line, _content_width()):
            print(_paint(_row(line), style), file=sys.stdout, flush=True)

    detail_rows = _detail_lines(details or [])
    if detail_rows:
        print(_paint(_box_mid(hero=hero), palette["border"]), file=sys.stdout, flush=True)
        for line in detail_rows:
            print(_paint(_row(line), palette["detail"]), file=sys.stdout, flush=True)

    print(_paint(_box_bottom(hero=hero), palette["border"]), file=sys.stdout, flush=True)


def _emit_lines(stage, lines):
    meta = _stage_meta(stage)
    palette = _palette(meta["theme"])
    timestamp = time.strftime("%H:%M:%S")
    label = meta["compact"] if _layout_mode() != "standard" else meta["label"]
    prefix = f"{timestamp} │ {meta['icon']} {label:<{_label_width()}} │ "
    continuation = " " * len(prefix)
    message_width = max(_content_width() - len(prefix), 16)

    for line_index, raw_line in enumerate(lines):
        wrapped_lines = _wrap_text(raw_line, message_width)
        for wrapped_index, line in enumerate(wrapped_lines):
            leader = prefix if line_index == 0 and wrapped_index == 0 else continuation
            style = palette["log"] if line_index == 0 else palette["detail"]
            print(_paint(_row(leader + line), style), file=sys.stdout, flush=True)


def emit(stage, message):
    _emit_lines(stage, [str(message)])


def format_count(value):
    if value is None:
        return "n/a"
    try:
        if isinstance(value, float) and not value.is_integer():
            return f"{value:,.2f}"
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _compact_count(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    absolute = abs(numeric)
    if absolute >= 1_000_000:
        return f"{numeric / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{numeric / 1_000:.1f}k"
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.1f}"


def format_duration(seconds):
    if seconds is None:
        return "n/a"
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return str(seconds)
    if not math.isfinite(seconds):
        return "n/a"

    seconds = max(int(round(seconds)), 0)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)

    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_rate(rate):
    if rate is None:
        return None
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        return str(rate)
    if not math.isfinite(rate) or rate < 0:
        return None
    if rate >= 1_000_000:
        return f"{rate / 1_000_000:.1f}M/s"
    if rate >= 1_000:
        return f"{rate / 1_000:.1f}k/s"
    return f"{rate:.1f}/s"


def render_bar(current, total, width=20):
    width = max(int(width), 4)
    try:
        current = float(current)
        total = float(total)
    except (TypeError, ValueError):
        return "░" * width
    if total <= 0:
        return "░" * width

    ratio = max(0.0, min(current / total, 1.0))
    filled = min(width, int(round(ratio * width)))
    return ("█" * filled) + ("░" * (width - filled))


def _progress_width(width):
    mode = _layout_mode()
    default = 22 if mode == "standard" else 16 if mode == "compact" else 10
    upper = 28 if mode == "standard" else 18 if mode == "compact" else 12
    return max(8, min(width if width is not None else default, upper))


def emit_progress(stage, current, total, speed=None, eta_seconds=None, extra=None, width=20):
    try:
        current_value = float(current)
    except (TypeError, ValueError):
        current_value = 0.0
    try:
        total_value = float(total)
    except (TypeError, ValueError):
        total_value = 0.0

    ratio = 0.0 if total_value <= 0 else max(0.0, min(current_value / total_value, 1.0))
    mode = _layout_mode()
    bar = render_bar(current_value, total_value, width=_progress_width(width))
    counts = (
        f"{format_count(current)}/{format_count(total)}"
        if mode == "standard"
        else f"{_compact_count(current)}/{_compact_count(total)}"
    )

    if mode == "standard":
        primary = f"progress  {ratio * 100:5.1f}%  ▐{bar}▌  {counts}"
    else:
        primary = f"{ratio * 100:5.1f}%  ▐{bar}▌  {counts}"

    secondary_parts = []
    rate_text = _format_rate(speed)
    if rate_text:
        secondary_parts.append(f"throughput {rate_text}" if mode == "standard" else rate_text)

    eta_text = format_duration(eta_seconds)
    if eta_text != "n/a":
        secondary_parts.append(f"eta {eta_text}")

    if extra:
        secondary_parts.append(str(extra))

    if secondary_parts:
        separator = "  ·  " if mode == "standard" else " · "
        _emit_lines(stage, [primary, separator.join(secondary_parts)])
        return
    _emit_lines(stage, [primary])
