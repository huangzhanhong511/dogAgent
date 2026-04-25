#!/bin/bash
#
# dogAgent 知识库定期维护脚本
#
# 用法:
#   ./scripts/weekly_lint.sh              # 手动执行
#   crontab: 0 3 * * 1 /path/to/weekly_lint.sh  # 每周一凌晨 3 点
#
# 操作：
#   1. 审计报告（输出到日志）
#   2. 清理无关文章
#   3. LLM 交叉引用更新
#   4. 重建索引
#   5. 快速准确度测试
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/dogAgent/data/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/lint_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR/dogAgent"
source .venv/bin/activate

echo "========================================" | tee -a "$LOG_FILE"
echo "dogAgent 知识库维护 — $TIMESTAMP" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 1. 审计报告
echo "" | tee -a "$LOG_FILE"
echo ">>> Step 1: 审计报告" | tee -a "$LOG_FILE"
python agent/wiki_lint.py --report 2>&1 | tee -a "$LOG_FILE"

# 2. 清理无关文章
echo "" | tee -a "$LOG_FILE"
echo ">>> Step 2: 清理无关文章" | tee -a "$LOG_FILE"
python agent/wiki_lint.py --prune 2>&1 | tee -a "$LOG_FILE"

# 3. 交叉引用更新
echo "" | tee -a "$LOG_FILE"
echo ">>> Step 3: LLM 交叉引用" | tee -a "$LOG_FILE"
python agent/wiki_lint.py --crossref 2>&1 | tee -a "$LOG_FILE"

# 4. 重建索引
echo "" | tee -a "$LOG_FILE"
echo ">>> Step 4: 重建索引" | tee -a "$LOG_FILE"
python agent/build_index.py 2>&1 | tee -a "$LOG_FILE"

# 5. 快速准确度测试
echo "" | tee -a "$LOG_FILE"
echo ">>> Step 5: 准确度测试" | tee -a "$LOG_FILE"
python tests/test_knowledge_accuracy.py --quick 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "维护完成。日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
