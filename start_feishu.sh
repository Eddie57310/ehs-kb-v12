#!/bin/bash
cd ~/doc_parser_v12

# 先杀掉所有旧实例，避免新旧代码并行抢同一个飞书 bot 的消息（应答会随机命中旧码），
# 也避免 BM25 pkl / chroma 被两个进程同时占用。pkill 模式只匹配 python 服务进程，
# 不会误杀本脚本（本脚本命令行是 start_feishu.sh，不含 feishu_ws_server_v12.py）。
pkill -f feishu_ws_server_v12.py && echo "已停止旧实例，等待退出释放 BM25 pkl / chroma..." && sleep 3

# 防御：仍有残留就报错退出，绝不在旧实例还活着时再起一个
if pgrep -f feishu_ws_server_v12.py > /dev/null; then
    echo "❌ 仍有 feishu_ws_server_v12.py 进程未退出，请手动 kill 后再启动："
    pgrep -af feishu_ws_server_v12.py
    exit 1
fi

nohup venv/bin/python feishu_ws_server_v12.py > logs/feishu_ws_server_v12_$(date +%Y-%m-%d).log 2>&1 &
echo "飞书 bot 已启动（单实例），PID=$!"
