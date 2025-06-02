#!/bin/bash

# Simple script to test the transcription API
# Usage: ./test_api.sh <audio_file>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <audio_file>"
    echo "Example: $0 test1.wav"
    exit 1
fi

AUDIO_FILE=$1

if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: File '$AUDIO_FILE' not found"
    exit 1
fi

echo "Testing transcription API with file: $AUDIO_FILE"
echo "API Response:"

curl -X POST "http://localhost:8000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@$AUDIO_FILE" \
     | jq '.'

echo
