#!/usr/bin/env bash
# ============================================================
#  Stars2Cells launcher for macOS (source / conda install)
#
#  Double-clickable equivalent of launch_s2c.bat: activates the
#  "s2c" conda env, runs stars2cells.py, and relaunches it
#  automatically (up to 5 times) if it crashes.
#
#  One-time setup so Finder will run it:  chmod +x launch_s2c.command
#
#  NOT needed for the packaged Stars2Cells.app from the Releases
#  page - that app is self-contained and launches directly.
# ============================================================

SCRIPT=stars2cells.py
MAX_RESTARTS=5
RESTARTS=0

cd "$(dirname "$0")"

# Locate conda
CONDA_PATH=""
for candidate in "$HOME/anaconda3" "$HOME/miniconda3" \
                 "/opt/anaconda3" "/opt/miniconda3" \
                 "/opt/homebrew/Caskroom/miniconda/base" \
                 "/usr/local/Caskroom/miniconda/base"; do
    if [ -d "$candidate" ]; then
        CONDA_PATH="$candidate"
        break
    fi
done

if [ -z "$CONDA_PATH" ]; then
    echo "ERROR: Could not find anaconda3 or miniconda3 in the usual locations"
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

    # Reactivate s2c env
    # shellcheck disable=SC1091
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    if conda activate s2c 2>/dev/null; then
        echo "  Conda env: s2c [OK]"
    else
        echo "  WARNING: Could not activate s2c env, trying anyway..."
    fi

    echo "  Launching: python $SCRIPT"
    echo
    python "$SCRIPT"
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
