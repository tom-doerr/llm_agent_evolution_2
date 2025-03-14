#!/bin/bash
# Run the Streamlit dashboard or control center on a non-standard port

# Default port (non-standard)
PORT=8765
APP="dashboard"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --app)
      APP="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./run_streamlit.sh [--port PORT] [--app dashboard|control]"
      exit 1
      ;;
  esac
done

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing now..."
    pip install streamlit pandas
fi

# Run the selected app
if [ "$APP" == "dashboard" ]; then
    echo "Starting Streamlit dashboard on port $PORT..."
    streamlit run streamlit_dashboard.py --server.port $PORT
elif [ "$APP" == "control" ]; then
    echo "Starting Streamlit control center on port $PORT..."
    streamlit run streamlit_control.py --server.port $PORT
else
    echo "Unknown app: $APP"
    echo "Usage: ./run_streamlit.sh [--port PORT] [--app dashboard|control]"
    exit 1
fi
