#!/bin/bash
apt-get update
apt-get install -y libgl1
uvicorn main:app --host 0.0.0.0 --port 8000