#!/bin/sh
# Install (or reinstall) the weekly lookahead-model retrain as a launchd agent.
# Runs scripts/retrain_lookahead.py every Sunday 06:00 local; output appends to
# data/logs/retrain_lookahead.log. Re-run this script after moving the repo.
#   sh scripts/install_retrain_job.sh            # install / update
#   sh scripts/install_retrain_job.sh --remove   # uninstall
set -eu

LABEL="com.kalshibot.retrain-lookahead"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [ "${1:-}" = "--remove" ]; then
    rm -f "$PLIST"
    echo "removed $LABEL"
    exit 0
fi

mkdir -p "$ROOT/data/logs" "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>WorkingDirectory</key><string>$ROOT</string>
    <key>ProgramArguments</key>
    <array>
        <string>$ROOT/venv/bin/python</string>
        <string>$ROOT/scripts/retrain_lookahead.py</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key><integer>0</integer>
        <key>Hour</key><integer>6</integer>
        <key>Minute</key><integer>0</integer>
    </dict>
    <key>StandardOutPath</key><string>$ROOT/data/logs/retrain_lookahead.log</string>
    <key>StandardErrorPath</key><string>$ROOT/data/logs/retrain_lookahead.log</string>
    <key>LowPriorityIO</key><true/>
    <key>Nice</key><integer>10</integer>
</dict>
</plist>
EOF
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL (Sundays 06:00) -> $PLIST"
echo "run now: launchctl kickstart gui/$(id -u)/$LABEL"
