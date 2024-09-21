#!/bin/bash

# Run car.py
echo "Running car.py..."
python3 car.py

# Check if car.py ran successfully
if [ $? -eq 0 ]; then
    echo "car.py executed successfully!"
else
    echo "car.py encountered an error." >&2
    exit 1
fi

# Run bank.py
echo "Running bank.py..."
python3 bank.py

# Check if bank.py ran successfully
if [ $? -eq 0 ]; then
    echo "bank.py executed successfully!"
else
    echo "bank.py encountered an error." >&2
    exit 1
fi

echo "Both scripts executed successfully!"
