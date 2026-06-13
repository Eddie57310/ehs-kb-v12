#!/bin/bash
cd /home/rr/doc_parser_v12
LOG=logs/pipeline_$(date +%Y%m%d_%H%M%S).log

echo "[$(date)] === 等待当前 rebuild_md 进程结束 ===" | tee -a $LOG
while pgrep -f "rebuild_md.py" > /dev/null; do sleep 5; done

echo "[$(date)] === 清空 reviewed_md ===" | tee -a $LOG
rm -rf reviewed_md/*

echo "[$(date)] === 开始 rebuild_md.py ===" | tee -a $LOG
venv/bin/python rebuild_md.py 2>&1 | tee -a $LOG

echo "[$(date)] === rebuild_md 完成，统计结果 ===" | tee -a $LOG
grep "done:" $LOG | tail -1 | tee -a $LOG

echo "" | tee -a $LOG
echo "==============================================" | tee -a $LOG
echo "  rebuild_md.py 执行完毕。                  " | tee -a $LOG
echo "  请在 md_reviewer (端口8084) 中完成人工审核   " | tee -a $LOG
echo "  审核完成后再手动执行:                       " | tee -a $LOG
echo "  venv/bin/python index_reviewed_md.py       " | tee -a $LOG
echo "==============================================" | tee -a $LOG

echo "[$(date)] === 全部完成（index_reviewed_md.py 需手动执行） ===" | tee -a $LOG
