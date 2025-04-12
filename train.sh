#!/bin/bash

# This script will run the train.py Python script and log the output.

# Create a log file (optional: add a timestamp to the log filename)
LOG_FILE="training_$(date +'%Y%m%d_%H%M%S').log"

# Run the Python script and redirect output to the log file
echo "Starting training at $(date)" > $LOG_FILE
python3 train_pytorch.py >> $LOG_FILE 2>&1

# Print message when done
echo "Training finished at $(date)" >> $LOG_FILE
