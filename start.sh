#!/bin/bash
echo "Starting Streamlit dashboard..."
streamlit run src/dashboard.py --server.port $PORT --server.address 0.0.0.0
