#!/bin/bash
# Script to start React app on local network
echo "Starting P4P Frontend on local network..."
echo "Your local IP: $(ifconfig | grep 'inet ' | grep -v 127.0.0.1 | awk '{print $2}' | head -1)"
echo "Access the app at: http://$(ifconfig | grep 'inet ' | grep -v 127.0.0.1 | awk '{print $2}' | head -1):3000"
echo ""
echo "Make sure your phone/other devices are on the same WiFi network!"
echo ""
HOST=0.0.0.0 npm start
