#!/bin/bash
# 企业微信入口启动脚本（v12）
# 注意：与飞书入口二选一跑——两者都会各自加载一份 bge-m3/reranker，同跑会吃双份内存。
cd ~/doc_parser_v12
pkill -f wecom_server.py 2>/dev/null
sleep 1
nohup venv/bin/python wecom_server.py > logs/wecom_run.log 2>&1 &
echo "企业微信入口已启动，日志: logs/wecom_run.log  (端口见 .env WECOM_PORT，默认 8090)"
