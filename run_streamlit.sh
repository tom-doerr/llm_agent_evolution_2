#!/bin/bash
# Run the Streamlit dashboard on a non-standard port

# Default port (non-standard)
PORT=8765
if [ ! -z "$1" ]; then
    PORT=$1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing now..."
    pip install streamlit pandas
fi

# Run the dashboard
echo "Starting Streamlit dashboard on port $PORT..."
streamlit run streamlit_dashboard.py --server.port $PORT
