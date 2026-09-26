#!/bin/bash
# ═══════════════════════════════════════════════════════
# 飞书私聊卡片发送 - 通用 interactive 卡片发送脚本
#
# 用法:
#   echo '{"config":{...},"header":{...},"elements":[...]}' | bash notify_feishu_card.sh
#   ... | bash notify_feishu_card.sh --dry-run   # 只打印，不发送
#
# 只发私聊（receive_id_type=open_id/user_id），不支持 chat_id/群聊——
# 这个脚本专用于「只发给 Kai 本人」的场景。
#
# 依赖（.env 或环境变量）:
#   FEISHU_APP_ID
#   FEISHU_APP_SECRET
#   FEISHU_KAI_RECEIVE_ID       接收方 ID（open_id/user_id，按应用区分，不可跨应用复用）
#   FEISHU_KAI_RECEIVE_ID_TYPE  可选，默认 open_id
#
# 取 tenant_access_token 的写法沿用自 notify_feishu.sh。
# ═══════════════════════════════════════════════════════
set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

CARD_JSON="$(cat)"
if [ -z "$CARD_JSON" ]; then
  echo "✗ 未从 stdin 收到卡片 JSON" >&2
  exit 1
fi

if [ "$DRY_RUN" = true ]; then
  echo "$CARD_JSON" | python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
  echo "（--dry-run，未发送）"
  exit 0
fi

# 加载 .env（仅用于本地/自托管场景；GitHub Actions 走 secrets 注入的环境变量）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi

APP_ID="${FEISHU_APP_ID:?需要 FEISHU_APP_ID}"
APP_SECRET="${FEISHU_APP_SECRET:?需要 FEISHU_APP_SECRET}"
RECEIVE_ID="${FEISHU_KAI_RECEIVE_ID:?需要 FEISHU_KAI_RECEIVE_ID}"
RECEIVE_ID_TYPE="${FEISHU_KAI_RECEIVE_ID_TYPE:-open_id}"

if [ "$RECEIVE_ID_TYPE" != "open_id" ] && [ "$RECEIVE_ID_TYPE" != "user_id" ]; then
  echo "✗ RECEIVE_ID_TYPE 必须是 open_id 或 user_id（私聊），拒绝发送到: $RECEIVE_ID_TYPE" >&2
  exit 1
fi

# ── 1. 获取 tenant_access_token ──
TOKEN_RESP=$(curl -s -X POST \
  "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal" \
  -H "Content-Type: application/json" \
  -d "{\"app_id\":\"$APP_ID\",\"app_secret\":\"$APP_SECRET\"}")

TOKEN=$(echo "$TOKEN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tenant_access_token',''))" 2>/dev/null)

if [ -z "$TOKEN" ]; then
  echo "✗ 获取飞书 token 失败" >&2
  exit 1
fi

# ── 2. 发送 interactive 卡片（私聊）──
PAYLOAD=$(CARD_JSON="$CARD_JSON" RECEIVE_ID="$RECEIVE_ID" python3 -c "
import json, os
card = json.loads(os.environ['CARD_JSON'])
payload = {
    'receive_id': os.environ['RECEIVE_ID'],
    'msg_type': 'interactive',
    'content': json.dumps(card, ensure_ascii=False),
}
print(json.dumps(payload, ensure_ascii=False))
")

SEND_RESP=$(curl -s -X POST \
  "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=${RECEIVE_ID_TYPE}" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

CODE=$(echo "$SEND_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('code',999))" 2>/dev/null)

if [ "$CODE" = "0" ]; then
  echo "✓ 飞书卡片已私聊发送给 Kai"
else
  # 不打印 SEND_RESP 全文，避免意外把 receive_id 之外的敏感字段带进日志；
  # 只保留 code，便于排查。
  echo "✗ 飞书卡片发送失败 (code=$CODE)" >&2
  exit 1
fi
