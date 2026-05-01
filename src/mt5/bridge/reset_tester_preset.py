#!/usr/bin/env python3
"""reset_tester_preset — reset MT5 Tester .set/.ini files to compiled defaults.

MT5 Strategy Tester auto-saves "last used input parameters" to .set/.ini
files in MQL5/Profiles/Tester/ at every Start. These files override the
compiled defaults next time you open the tester. After changing default
values in source code (e.g. Inp_SymbolSuffix or Inp_MacroSourceMode) and
recompiling, the saved presets keep the OLD values — and re-write them
each time MT5 saves, defeating any manual cleanup.

This utility forces specific input values into the saved tester preset
files. Run with MT5 closed so the change sticks (MT5 caches presets in
memory; reopening reads fresh from disk).

Usage:
    # Standard reset for FxMultiSleeve on this user's broker
    python reset_tester_preset.py

    # Or override values
    python reset_tester_preset.py --suffix .c --mode AUTO

    # Or list current state without changing
    python reset_tester_preset.py --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Maps EMacroSourceMode enum names to their integer value (must match
# src/mt5/Include/FxCommon.mqh)
MACRO_SOURCE_MODE = {
    "FILE":    0,
    "NATIVE":  1,
    "HYBRID":  2,
    "HISTORY": 3,
    "AUTO":    4,
}
MACRO_SOURCE_MODE_MAX = max(MACRO_SOURCE_MODE.values())

DEFAULT_TESTER_DIR = (
    Path.home() / "AppData/Roaming/MetaQuotes/Terminal"
    / "D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Profiles/Tester"
)
DEFAULT_EA = "FxMultiSleeve"
DEFAULT_SUFFIX = ".c"
DEFAULT_MODE = "AUTO"


def find_preset_files(tester_dir: Path, ea: str) -> list[Path]:
    """Find all .set / .ini files belonging to the given EA."""
    return sorted([
        *tester_dir.glob(f"{ea}.set"),
        *tester_dir.glob(f"{ea}.*.ini"),
    ])


def update_preset(path: Path, suffix: str, mode_int: int,
                  check_only: bool) -> bool:
    """Read .set/.ini (UTF-16), patch SymbolSuffix and MacroSourceMode,
    write back. Returns True if the file needed changes (value diff OR
    corrupted line endings)."""
    raw = path.read_bytes()
    # Detect the previous version of this script that wrote \r\r\n line
    # terminators (Windows universal-newline translation bug). MT5 keeps the
    # trailing \r as part of the string value, breaking symbol lookup.
    bad_eol = b"\r\x00\r\x00\n\x00" in raw
    text = raw.decode("utf-16")
    lines = text.splitlines(keepends=False)

    new_lines = []
    value_changed = False
    for ln in lines:
        if ln.startswith("Inp_SymbolSuffix="):
            new = f"Inp_SymbolSuffix={suffix}"
            if new != ln:
                print(f"  - {ln}")
                print(f"  + {new}")
                value_changed = True
            new_lines.append(new)
        elif ln.startswith("Inp_MacroSourceMode="):
            new = (f"Inp_MacroSourceMode={mode_int}||{mode_int}"
                   f"||0||{MACRO_SOURCE_MODE_MAX}||N")
            if new != ln:
                print(f"  - {ln}")
                print(f"  + {new}")
                value_changed = True
            new_lines.append(new)
        else:
            new_lines.append(ln)

    if bad_eol and not value_changed:
        print(f"  (no value diff, but fixing corrupted \\r\\r\\n line endings)")

    needs_write = value_changed or bad_eol
    if check_only or not needs_write:
        return needs_write

    # MT5 expects UTF-16 LE with BOM and CRLF line terminators.
    # IMPORTANT: write_bytes (not write_text) to bypass Windows universal-newline
    # translation, which would turn our '\n' into '\r\n' and produce '\r\r\n'.
    new_text = "\r\n".join(new_lines) + "\r\n"
    path.write_bytes(new_text.encode("utf-16"))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ea", default=DEFAULT_EA,
                        help=f"EA name (default {DEFAULT_EA})")
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX,
                        help=f"Inp_SymbolSuffix value (default '{DEFAULT_SUFFIX}')")
    parser.add_argument("--mode", default=DEFAULT_MODE,
                        choices=list(MACRO_SOURCE_MODE.keys()),
                        help=f"Inp_MacroSourceMode (default {DEFAULT_MODE})")
    parser.add_argument("--tester-dir", type=Path, default=DEFAULT_TESTER_DIR,
                        help="MT5 Profiles/Tester directory")
    parser.add_argument("--check", action="store_true",
                        help="Show diff without writing")
    args = parser.parse_args()

    if not args.tester_dir.exists():
        print(f"ERROR: tester dir not found: {args.tester_dir}", file=sys.stderr)
        return 1

    files = find_preset_files(args.tester_dir, args.ea)
    if not files:
        print(f"No .set/.ini files for {args.ea} in {args.tester_dir}")
        return 0

    mode_int = MACRO_SOURCE_MODE[args.mode]
    print(f"Target: SymbolSuffix='{args.suffix}'  MacroSourceMode={args.mode}={mode_int}")
    print(f"Found {len(files)} preset file(s):")

    n_changed = 0
    for f in files:
        print(f"\n[{f.name}]")
        if update_preset(f, args.suffix, mode_int, args.check):
            n_changed += 1

    if args.check:
        print(f"\n=== CHECK ONLY — {n_changed} files would be modified ===")
    else:
        print(f"\n=== {n_changed} files updated ===")
        if n_changed:
            print("WARNING: make sure MT5 is closed BEFORE running this, then "
                  "reopen MT5 to read the corrected presets. Otherwise MT5 "
                  "will overwrite from its in-memory cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
