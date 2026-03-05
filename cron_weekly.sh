#!/bin/bash
# 前沿洞察 cron 入口 — 每周自动执行
# SG cron: 0 10 * * 2 /root/ai-dataset-radar/cron_weekly.sh
source /root/sg_env.sh
cd "$RADAR_DIR"
exec bash run_weekly.sh >> /var/log/insights_weekly.log 2>&1
