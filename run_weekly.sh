#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 前沿洞察周刊 - 一键生成流水线
#
# 用法:
#   bash run_weekly.sh                     # 全自动（日期=今天，周号自增，标题自动）
#   bash run_weekly.sh 2026-03-04          # 指定日期
#   bash run_weekly.sh 2026-03-04 W15      # 指定日期+周号
#   bash run_weekly.sh 2026-03-04 W15 "标题A｜标题B"  # 全手动
#   bash run_weekly.sh --with-recipe       # 额外跑 DataRecipe（更慢，非默认）
#   bash run_weekly.sh --skip-scan         # 跳过扫描（已有数据时）
#   bash run_weekly.sh --skip-deploy       # 跳过部署（仅生成）
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# ── 颜色 ──
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${GREEN}▶ Step $1: $2${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }
PUBLISH_CHANGED=false

# ── 解析参数 ──
SKIP_SCAN=false
SKIP_DEPLOY=false
WITH_RECIPE=false
POSITIONAL=()

for arg in "$@"; do
  case $arg in
    --skip-scan)   SKIP_SCAN=true ;;
    --skip-deploy) SKIP_DEPLOY=true ;;
    --with-recipe) WITH_RECIPE=true ;;
    *)             POSITIONAL+=("$arg") ;;
  esac
done

DATE="${POSITIONAL[0]:-$(date +%Y-%m-%d)}"
WEEK="${POSITIONAL[1]:-auto}"
TITLE="${POSITIONAL[2]:-}"
YEAR="$(echo "$DATE" | cut -d- -f1)"

# ── 路径（本地 macOS 默认，SG 通过环境变量覆盖）──
RADAR_DIR="${RADAR_DIR:-$(cd "$(dirname "$0")" && pwd)}"
WEBSITE_DIR="${WEBSITE_DIR:-$HOME/knowlyr-website}"
VENV="${VENV:-$RADAR_DIR/.venv/bin}"
PYTHON_BRIEF="${PYTHON_BRIEF:-/opt/homebrew/bin/python3}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"

REPORTS="$RADAR_DIR/data/reports/$DATE"
TODAY="$(date +%Y-%m-%d)"

# ── 加载 .env ──
if [ -f "$RADAR_DIR/.env" ]; then
  set -a; source "$RADAR_DIR/.env"; set +a
fi

mkdir -p "$REPORTS"

if [ "$SKIP_SCAN" = false ] && [ "$DATE" != "$TODAY" ]; then
  fail "当前扫描仅支持生成今天（$TODAY）的报告；历史日期请先准备好 reports/$DATE 后再配合 --skip-scan 使用"
fi

publish_via_pr() {
  local year="$1"
  local week="$2"
  local title="$3"
  local year_slug week_slug
  year_slug="$(printf '%s' "$year" | tr '[:upper:]' '[:lower:]')"
  week_slug="$(printf '%s' "$week" | tr '[:upper:]' '[:lower:]')"
  local publish_branch="publish-${year_slug}-${week_slug}-$(date +%Y%m%d%H%M%S)"
  local publish_dir
  publish_dir="$(mktemp -d "${TMPDIR:-/tmp}/knowlyr-weekly.${year}-${week}.XXXXXX")"

  local files=(
    "insights/.contexts/${year}-${week}.json"
    "insights/.issues.json"
    "assets/imgs/og/${year}-${week}.png"
    "assets/imgs/qr/${year}-${week}.png"
    "data/i18n/en/contexts/${year}-${week}.json"
    "data/i18n/en/issues.json"
  )

  cleanup_publish_dir() {
    git -C "$WEBSITE_DIR" worktree remove --force "$publish_dir" >/dev/null 2>&1 || rm -rf "$publish_dir"
    git -C "$WEBSITE_DIR" worktree prune >/dev/null 2>&1 || true
  }
  trap cleanup_publish_dir RETURN

  git -C "$WEBSITE_DIR" fetch origin main --quiet || fail "拉取官网远端 main 失败"
  git -C "$WEBSITE_DIR" worktree add -b "$publish_branch" "$publish_dir" origin/main >/dev/null \
    || fail "创建发布 worktree 失败"

  local rel src
  for rel in "${files[@]}"; do
    src="$WEBSITE_DIR/$rel"
    [ -f "$src" ] || fail "缺少待发布文件: $src"
    mkdir -p "$publish_dir/$(dirname "$rel")"
    cp "$src" "$publish_dir/$rel"
  done

  git -C "$publish_dir" add "${files[@]}" || fail "暂存发布文件失败"
  if git -C "$publish_dir" diff --cached --quiet; then
    warn "发布 worktree 中无可提交变更，跳过部署"
    return 0
  fi

  PUBLISH_CHANGED=true
  git -C "$publish_dir" commit -m "前沿洞察 ${week}：${title}" >/dev/null \
    || fail "创建发布提交失败"
  git -C "$publish_dir" push -u origin "$publish_branch" >/dev/null \
    || fail "推送发布分支失败"

  local pr_body
  pr_body=$(cat <<EOF
## Summary
- publish Frontier Insights ${week} through ${DATE} in zh/en
- add source contexts, issue registries, and OG/QR assets

## Verification
- bash run_weekly.sh ${DATE} ${week} --skip-deploy
EOF
)

  local pr_url
  pr_url="$(
    cd "$publish_dir" && \
    gh pr create \
      --base main \
      --head "$publish_branch" \
      --title "前沿洞察 ${week}：${title}" \
      --body "$pr_body"
  )" || fail "创建 PR 失败"
  echo "  PR: $pr_url"

  (
    cd "$publish_dir" && \
    gh pr merge "$pr_url" --squash --admin --delete-branch >/dev/null
  ) || fail "PR 已创建但自动合并失败: $pr_url"

  echo "  ✓ PR 已合并，等待 Deploy workflow..."

  local run_id=""
  local i
  for i in $(seq 1 20); do
    run_id="$(
      cd "$publish_dir" && \
      gh run list \
        --workflow Deploy \
        --branch main \
        --event push \
        --limit 5 \
        --json databaseId \
        --jq '.[0].databaseId'
    )" || true
    if [ -n "$run_id" ] && [ "$run_id" != "null" ]; then
      break
    fi
    sleep 3
  done
  [ -n "$run_id" ] && [ "$run_id" != "null" ] || fail "未找到 Deploy workflow run"

  (
    cd "$publish_dir" && \
    gh run watch "$run_id" --exit-status
  ) || fail "Deploy workflow 失败: $run_id"
}

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
  if [ "$WITH_RECIPE" = true ]; then
    step 1 "Radar 扫描 + DataRecipe 分析"
  else
    step 1 "Radar 扫描（默认跳过 DataRecipe，优先保证发刊链路）"
  fi
  cd "$RADAR_DIR"
  SCAN_ARGS=(--days 7 --json)
  if [ "$WITH_RECIPE" = true ]; then
    SCAN_ARGS+=(--recipe --recipe-limit 5)
  fi

  set +e
  "$VENV/python" src/main_intel.py "${SCAN_ARGS[@]}" 2>&1 | tee "$REPORTS/radar_scan.log"
  SCAN_STATUS=${PIPESTATUS[0]}
  TEE_STATUS=${PIPESTATUS[1]}
  set -e

  if [ "$TEE_STATUS" -ne 0 ]; then
    warn "扫描日志写入异常（tee exit=${TEE_STATUS}），继续检查核心产物"
  fi

  PROMPT_FILE="$REPORTS/intel_report_${DATE}_insights_prompt.md"
  REPORT_JSON="$REPORTS/intel_report_${DATE}.json"
  if [ "$SCAN_STATUS" -ne 0 ]; then
    if [ -f "$PROMPT_FILE" ] && [ -f "$REPORT_JSON" ]; then
      warn "扫描命令异常退出（exit=${SCAN_STATUS}），但核心报告已生成，继续后续步骤"
    else
      fail "扫描失败（exit=${SCAN_STATUS}），且核心报告未生成"
    fi
  fi

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
  TITLE=$(INSIGHTS_FILE="$INSIGHTS_FILE" "$PYTHON_BRIEF" -c "
import os, pathlib, re
text = pathlib.Path(os.environ['INSIGHTS_FILE']).read_text()
m = re.search(r'^## 候选标题\\s*\\n([\\s\\S]*?)(?=^##\\s|\\Z)', text, re.MULTILINE)
if m:
    for line in m.group(1).splitlines():
        line = line.strip()
        if re.match(r'^\\d+[.、) ]+', line):
            print(re.sub(r'^\\d+[.、) ]+', '', line).strip())
            break
" 2>/dev/null | xargs || true)

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
if [ -n "$(git status --porcelain --untracked-files=normal 2>/dev/null)" ]; then
  warn "官网仓库存在未提交改动，跳过 git pull，直接使用本地脚本与数据生成"
else
  git pull --rebase --quiet 2>&1 || warn "git pull 失败，使用本地版本继续"
fi

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
"$PYTHON_BRIEF" scripts/translate.py --file "${YEAR}-${WEEK}.json" 2>&1 || {
  warn "英文 context 翻译失败"
}
"$PYTHON_BRIEF" scripts/translate.py --file .issues.json 2>&1 || {
  warn "英文 issues 翻译失败"
}

# 翻译发生在 generate_brief 之后，需再刷新一遍英文详情页与首页入口
step 4.6 "刷新英文详情页与英文首页入口"
cd "$WEBSITE_DIR"
"$PYTHON_BRIEF" scripts/build.py --no-fetch --lang en insights 2>&1 || {
  warn "英文洞察页重建失败"
}
"$PYTHON_BRIEF" scripts/build.py --no-fetch --lang en pages 2>&1 || {
  warn "英文主站页面重建失败"
}

# ══════════════════════════════════════════════════════
# Step 5: 部署
# ══════════════════════════════════════════════════════
if [ "$SKIP_DEPLOY" = true ]; then
  warn "跳过部署（--skip-deploy）"
else
  step 5 "PR 发布 + 部署"

  cd "$WEBSITE_DIR"
  PUBLISH_CHANGED=false
  publish_via_pr "$YEAR" "$WEEK" "$TITLE"

  if [ "$PUBLISH_CHANGED" = true ]; then
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://knowlyr.com/insights/${YEAR}-${WEEK}.html" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
      echo -e "${GREEN}✓ 部署完成！https://knowlyr.com/insights/${YEAR}-${WEEK}.html${NC}"
    else
      warn "Deploy workflow 已完成，但线上检查返回 ${STATUS}"
    fi
  else
    warn "本次无源数据变更，跳过线上校验与飞书通知"
  fi

  # Step 6: 飞书通知
  if [ "$PUBLISH_CHANGED" = true ]; then
    step 6 "飞书群通知"
    if [ -f "$RADAR_DIR/notify_feishu.sh" ]; then
      bash "$RADAR_DIR/notify_feishu.sh" "$WEEK" "$TITLE" "$DATE"
    else
      warn "notify_feishu.sh 不存在，跳过通知"
    fi
  fi
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo -e "  ${GREEN}✅ 全流程完成！${NC}"
echo "  日期: $DATE  周号: $WEEK"
echo "  标题: $TITLE"
echo "  链接: https://knowlyr.com/insights/${YEAR}-${WEEK}.html"
echo "═══════════════════════════════════════════════════"
