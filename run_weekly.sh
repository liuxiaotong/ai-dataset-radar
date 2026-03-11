#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 前沿洞察周刊 - 一键生成流水线
#
# 用法:
#   bash run_weekly.sh                     # 全自动（日期=今天，周号自增，标题自动）
#   bash run_weekly.sh 2026-03-04          # 指定日期
#   bash run_weekly.sh 2026-03-04 W15      # 指定日期+周号
#   bash run_weekly.sh 2026-03-04 W15 "标题A｜标题B"  # 全手动
#   bash run_weekly.sh --skip-scan         # 跳过扫描（已有数据时）
#   bash run_weekly.sh --skip-deploy       # 跳过部署（仅生成）
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# ── 颜色 ──
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${GREEN}▶ Step $1: $2${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }

# ── 解析参数 ──
SKIP_SCAN=false
SKIP_DEPLOY=false
POSITIONAL=()

for arg in "$@"; do
  case $arg in
    --skip-scan)   SKIP_SCAN=true ;;
    --skip-deploy) SKIP_DEPLOY=true ;;
    *)             POSITIONAL+=("$arg") ;;
  esac
done

DATE="${POSITIONAL[0]:-$(date +%Y-%m-%d)}"
WEEK="${POSITIONAL[1]:-auto}"
TITLE="${POSITIONAL[2]:-}"

# ── 路径（本地 macOS 默认，SG 通过环境变量覆盖）──
RADAR_DIR="${RADAR_DIR:-$(cd "$(dirname "$0")" && pwd)}"
WEBSITE_DIR="${WEBSITE_DIR:-$HOME/knowlyr-website}"
VENV="${VENV:-$RADAR_DIR/.venv/bin}"
PYTHON_BRIEF="${PYTHON_BRIEF:-/opt/homebrew/bin/python3}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"

REPORTS="$RADAR_DIR/data/reports/$DATE"

# ── 加载 .env ──
if [ -f "$RADAR_DIR/.env" ]; then
  set -a; source "$RADAR_DIR/.env"; set +a
fi

echo "═══════════════════════════════════════════════════"
echo "  前沿洞察周刊生成 — $DATE"
echo "═══════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════
# Step 1: Radar 扫描 + DataRecipe
# ══════════════════════════════════════════════════════
if [ "$SKIP_SCAN" = true ]; then
  warn "跳过扫描（--skip-scan）"
  [ -d "$REPORTS" ] || fail "扫描数据目录不存在: $REPORTS"
else
  step 1 "Radar 扫描 + DataRecipe 分析"
  cd "$RADAR_DIR"
  "$VENV/python" src/main_intel.py \
    --days 7 \
    --recipe --recipe-limit 5 \
    --json \
    2>&1 | tee "$REPORTS/radar_scan.log"

  [ -d "$REPORTS" ] || fail "扫描完成但数据目录不存在: $REPORTS"
  echo -e "${GREEN}✓ 扫描完成${NC}"
fi

# ══════════════════════════════════════════════════════
# Step 2: Claude Code 分析洞察
# ══════════════════════════════════════════════════════
INSIGHTS_FILE="$REPORTS/intel_report_${DATE}_insights.md"

if [ -f "$INSIGHTS_FILE" ] && [ "$SKIP_SCAN" = true ]; then
  warn "洞察文件已存在，跳过分析"
else
  step 2 "API 分析洞察（不走 Claude Code CLI，避免 OAuth 冲突）"

  PROMPT_FILE="$REPORTS/intel_report_${DATE}_insights_prompt.md"
  [ -f "$PROMPT_FILE" ] || fail "缺少 insights_prompt.md"

  "$VENV/python" "$RADAR_DIR/analyze_insights.py" \
    --date "$DATE" \
    --radar-dir "$RADAR_DIR" \
    2>&1 | tee "$REPORTS/insights_analysis.log"

  [ -f "$INSIGHTS_FILE" ] || fail "分析脚本未生成 insights 文件"
  echo -e "${GREEN}✓ 洞察分析完成（$(wc -c < "$INSIGHTS_FILE") 字节）${NC}"
fi

# ══════════════════════════════════════════════════════
# Step 3: 计算周号 + 提取标题
# ══════════════════════════════════════════════════════
step 3 "计算周号 + 提取标题"

if [ "$WEEK" = "auto" ]; then
  WEEK_NUM=$(WEBSITE_DIR="$WEBSITE_DIR" DATE="$DATE" "$PYTHON_BRIEF" -c "
import json, pathlib, os
issues_path = pathlib.Path(os.environ['WEBSITE_DIR']) / 'insights/.issues.json'
if issues_path.exists():
    issues = json.loads(issues_path.read_text())
    max_w = max(int(i['week'][1:]) for i in issues) if issues else 0
    print(max_w + 1)
else:
    from datetime import date
    d = date.fromisoformat(os.environ['DATE'])
    print(d.isocalendar()[1])
" 2>/dev/null) || {
    warn "周号计算失败，使用 ISO 周号"
    WEEK_NUM=$(date +%V)
  }
  WEEK="W${WEEK_NUM}"
  echo "  周号: ${WEEK}（自动计算）"
else
  echo "  周号: ${WEEK}（手动指定）"
fi

if [ -z "$TITLE" ]; then
  # 从 insights.md 提取候选标题（取第一个）
  TITLE=$(grep -A1 '## 候选标题' "$INSIGHTS_FILE" 2>/dev/null \
    | grep '^[0-9]' | head -1 | sed 's/^[0-9]*[.、) ]*//' | xargs || true)

  if [ -z "$TITLE" ]; then
    # 回退：从第一条发现的标题生成
    TITLE=$(INSIGHTS_FILE="$INSIGHTS_FILE" WEEK="$WEEK" "$PYTHON_BRIEF" -c "
import re, pathlib, os
text = pathlib.Path(os.environ['INSIGHTS_FILE']).read_text()
m = re.search(r'####\s+1\.1\s+(.+?)(?:\s*\[P)', text)
if m:
    t = m.group(1).strip()
    print(t[:20] + '｜AI 数据行业周度洞察')
else:
    print('AI 数据行业前沿洞察｜' + os.environ['WEEK'] + ' 周报')
")
  fi
  echo "  标题: ${TITLE}（自动提取）"
else
  echo "  标题: ${TITLE}（手动指定）"
fi

# ══════════════════════════════════════════════════════
# Step 4: 生成页面
# ══════════════════════════════════════════════════════
step 4 "生成周刊页面"

cd "$WEBSITE_DIR"
git pull --rebase --quiet 2>&1 || warn "git pull 失败，使用本地版本继续"

# 统一标题分隔符：全角｜→ 半角 |（generate_brief.py 按半角解析）
TITLE_NORM=$(echo "$TITLE" | sed 's/｜/ | /g')

"$PYTHON_BRIEF" scripts/generate_brief.py \
  --date "$DATE" \
  --week "$WEEK" \
  --title "$TITLE_NORM" \
  --radar-data "$RADAR_DIR/data/reports"

echo -e "${GREEN}✓ 页面生成完成${NC}"

# ══════════════════════════════════════════════════════
# Step 4.5: 翻译英文版
# ══════════════════════════════════════════════════════
step 4.5 "翻译英文版"

cd "$WEBSITE_DIR"
YEAR=$(echo "$DATE" | cut -d- -f1)
"$PYTHON_BRIEF" scripts/translate.py --file "${YEAR}-${WEEK}.json" 2>&1 || {
  warn "英文 context 翻译失败"
}
"$PYTHON_BRIEF" scripts/translate.py --file .issues.json 2>&1 || {
  warn "英文 issues 翻译失败"
}

# ══════════════════════════════════════════════════════
# Step 5: 部署
# ══════════════════════════════════════════════════════
if [ "$SKIP_DEPLOY" = true ]; then
  warn "跳过部署（--skip-deploy）"
else
  step 5 "Git 提交 + 部署"

  cd "$WEBSITE_DIR"
  git add \
    insights/*.html \
    insights/feed.xml \
    insights/.contexts/ \
    insights/.issues.json \
    assets/imgs/og/ \
    assets/imgs/qr/ \
    sitemap.xml \
    en/insights/ \
    data/i18n/en/contexts/ \
    2>/dev/null || true

  # 检查是否有变更
  if git diff --cached --quiet; then
    warn "无新变更，跳过提交"
  else
    git commit -m "前沿洞察 $WEEK：$TITLE"
    git push origin main
    echo -e "${GREEN}✓ 已推送，等待 GitHub Actions 部署...${NC}"

    # 等待部署完成（最多 3 分钟）
    DEPLOY_OK=false
    echo -n "  等待部署"
    for i in $(seq 1 18); do
      sleep 10
      echo -n "."
      STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://knowlyr.com/insights/${WEEK}.html" 2>/dev/null || echo "000")
      if [ "$STATUS" = "200" ]; then
        echo ""
        echo -e "${GREEN}✓ 部署完成！https://knowlyr.com/insights/${WEEK}.html${NC}"
        DEPLOY_OK=true
        break
      fi
    done

    if [ "$DEPLOY_OK" = false ]; then
      echo ""
      warn "部署可能未完成（最后状态: ${STATUS}），请手动检查 GitHub Actions"
    fi
  fi

  # Step 6: 飞书通知
  step 6 "飞书群通知"
  if [ -f "$RADAR_DIR/notify_feishu.sh" ]; then
    bash "$RADAR_DIR/notify_feishu.sh" "$WEEK" "$TITLE" "$DATE"
  else
    warn "notify_feishu.sh 不存在，跳过通知"
  fi
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo -e "  ${GREEN}✅ 全流程完成！${NC}"
echo "  日期: $DATE  周号: $WEEK"
echo "  标题: $TITLE"
echo "  链接: https://knowlyr.com/insights/${WEEK}.html"
echo "═══════════════════════════════════════════════════"
