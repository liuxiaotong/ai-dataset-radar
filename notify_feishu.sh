#!/bin/bash
# ═══════════════════════════════════════════════════════
# 飞书群通知 - 前沿洞察发布通知
#
# 用法: bash notify_feishu.sh W14 "标题A｜标题B" 2026-03-04
# 依赖: .env 中的 FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_CHAT_ID
# ═══════════════════════════════════════════════════════
set -euo pipefail

WEEK="${1:?用法: notify_feishu.sh WEEK TITLE DATE}"
TITLE="${2:?缺少标题参数}"
DATE="${3:?缺少日期参数}"

# 加载 .env
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a; source "$SCRIPT_DIR/.env"; set +a
fi

APP_ID="${FEISHU_APP_ID:?需要 FEISHU_APP_ID}"
APP_SECRET="${FEISHU_APP_SECRET:?需要 FEISHU_APP_SECRET}"
CHAT_ID="${FEISHU_CHAT_ID:?需要 FEISHU_CHAT_ID}"
PAGE_URL="https://knowlyr.com/insights/${WEEK}.html"

# ── 1. 获取 tenant_access_token ──
TOKEN_RESP=$(curl -s -X POST \
  "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal" \
  -H "Content-Type: application/json" \
  -d "{\"app_id\":\"$APP_ID\",\"app_secret\":\"$APP_SECRET\"}")

TOKEN=$(echo "$TOKEN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tenant_access_token',''))" 2>/dev/null)

if [ -z "$TOKEN" ]; then
  echo "✗ 获取飞书 token 失败: $TOKEN_RESP"
  exit 1
fi

# ── 2. 构建消息 ──
# 标题分割（用 ｜ 全角分隔符）
TITLE_PART1=$(echo "$TITLE" | sed 's/[｜|].*//' | xargs)
TITLE_PART2=$(echo "$TITLE" | sed 's/.*[｜|]//' | xargs)

MSG_CONTENT=$(WEEK="$WEEK" PART1="$TITLE_PART1" PART2="$TITLE_PART2" PAGE_URL="$PAGE_URL" python3 -c "
import json, os
content = {
    'zh_cn': {
        'title': f'📡 前沿洞察 {os.environ[\"WEEK\"]} 已发布',
        'content': [
            [{'tag': 'text', 'text': os.environ['PART1']}],
            [{'tag': 'text', 'text': os.environ['PART2']}],
            [{'tag': 'text', 'text': ''}],
            [{'tag': 'a', 'text': '👉 点击阅读完整报告', 'href': os.environ['PAGE_URL']}],
        ]
    }
}
print(json.dumps(content, ensure_ascii=False))
")

# ── 3. 发送消息 ──
SEND_RESP=$(curl -s -X POST \
  "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"receive_id\":\"$CHAT_ID\",\"msg_type\":\"post\",\"content\":$(echo "$MSG_CONTENT" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")}")

# 检查结果
CODE=$(echo "$SEND_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('code',999))" 2>/dev/null)

if [ "$CODE" = "0" ]; then
  echo "✓ 飞书通知已发送到 Knowlyr 群"
else
  echo "✗ 飞书发送失败 (code=$CODE): $SEND_RESP"
  exit 1
fi
