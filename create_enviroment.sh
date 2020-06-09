#! /bin/bash

#Create virtualenv
DIR="$(pwd)/gaze_detection/"

if [ ! -d "$DIR" ] 
then
    python3 -m venv gaze_detection
    echo "Created virtualenv"
    source gaze_detection/bin/activate
    pip install -r requirements.txt
    pip freeze
    echo "Virtual env READY to work"
fi

echo "Venv created!!"