#!/bin/bash
# Script to start both frontend and backend for development

echo "Starting P4P Development Environment..."
echo "======================================"

# Get local IP for network access
LOCAL_IP=$(ifconfig | grep 'inet ' | grep -v 127.0.0.1 | awk '{print $2}' | head -1)

echo "Local IP: $LOCAL_IP"
echo "Frontend: http://$LOCAL_IP:3000"
echo "Backend:  http://$LOCAL_IP:5001"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

# Set up cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting Flask backend..."
cd backend
python app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "Starting React frontend..."
cd ../frontend
HOST=0.0.0.0 npm start &
FRONTEND_PID=$!

echo ""
echo "Both servers are starting up..."
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for both processes
wait
