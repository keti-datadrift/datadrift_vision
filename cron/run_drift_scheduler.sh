#!/bin/bash

# Shell wrapper script for running drift scheduler as a cron job
# This script ensures proper environment setup and error handling

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set up Python environment
# Uncomment and modify if using virtual environment:
# source "$PROJECT_ROOT/venv/bin/activate"

# Change to the cron directory
cd "$SCRIPT_DIR" || exit 1

# Run the drift scheduler with Python
python3 drift_scheduler.py

# Capture exit code
EXIT_CODE=$?

# Log the execution
if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Drift scheduler completed successfully" >> cron_execution.log
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Drift scheduler failed with exit code: $EXIT_CODE" >> cron_execution.log
fi

exit $EXIT_CODE
