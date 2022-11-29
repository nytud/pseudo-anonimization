#!/usr/bin/env bash

bash /app/install.sh

uvicorn main:app --host 0.0.0.0 --port "${PORT}"