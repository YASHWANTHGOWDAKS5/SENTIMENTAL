#!/bin/bash
# start.sh - start script for the Flask Emotion Chatbot app

# Run the Flask app with Gunicorn on port 8000, bind to all IPs
gunicorn --bind 0.0.0.0:8000 app:app
