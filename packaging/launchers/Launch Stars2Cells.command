#!/usr/bin/env bash
# ============================================================
#  Optional launcher for the PACKAGED macOS build.
#
#  Ships in the release folder next to Stars2Cells.app.
#  Double-clicking Stars2Cells.app works on its own; use this
#  instead if you want the app to relaunch automatically after
#  a crash (same behavior as the old launch_s2c.bat), with the
#  exit code shown in a Terminal window.
# ============================================================

MAX_RESTARTS=5
RESTARTS=0

cd "$(dirname "$0")"

APP_BIN="./Stars2Cells.app/Contents/MacOS/Stars2Cells"
if [ ! -x "$APP_BIN" ]; then
    echo "ERROR: Stars2Cells.app not found next to this launcher."
    read -r -p "Press Enter to close..."
    exit 1
fi

while true; do
    echo
    echo "============================================"
    if [ "$RESTARTS" -eq 0 ]; then
        echo "  Stars2Cells Launcher"
    else
        echo "  Stars2Cells Restarting [attempt $RESTARTS/$MAX_RESTARTS]"
    fi
    echo "============================================"
    echo

    "$APP_BIN"
    EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo
        echo "  Stars2Cells exited cleanly."
        break
    fi

    RESTARTS=$((RESTARTS + 1))
    if [ "$RESTARTS" -gt "$MAX_RESTARTS" ]; then
        echo
        echo "  Max restarts [$MAX_RESTARTS] reached. Giving up."
        echo "  Last exit code: $EXIT_CODE"
        break
    fi

    echo
    echo "  Crashed with exit code $EXIT_CODE. Restarting in 3 seconds..."
    sleep 3
done

echo
read -r -p "Press Enter to close..."
