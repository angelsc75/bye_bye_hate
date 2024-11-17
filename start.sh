#!/bin/bash
Xvfb :99 -screen 0 1024x768x16 &
sleep 1
streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0 --browser.gatherUsageStats=false