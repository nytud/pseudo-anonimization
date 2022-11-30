#!/usr/bin/env bash

bash /app/install.sh

uvicorn main:app --host 0.0.0.0 --port 80 --app-dir /app/src --reload --reload-dir /app/src

exit 0