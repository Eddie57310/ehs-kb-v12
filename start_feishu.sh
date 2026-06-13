#!/bin/bash
cd ~/doc_parser_v12
nohup venv/bin/python feishu_ws_server_v12.py > logs/feishu_ws_server_v12_$(date +%Y-%m-%d).log 2>&1 &
echo "飞书 bot 已启动，PID=$!"
