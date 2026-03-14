#!/usr/bin/env bash
# run-tasks.sh — Reads task files from tasks/ and processes them one by one with Claude Code.
# Usage: ./run-tasks.sh [--dry-run] [--from N] [--only N]
#
# Options:
#   --dry-run   Show which tasks would be processed without running them
#   --from N    Start from task N (e.g., --from 3 skips tasks 01, 02)
#   --only N    Run only task N

set -euo pipefail

TASKS_DIR="$(cd "$(dirname "$0")" && pwd)/tasks"
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/tasks/logs"

DRY_RUN=false
FROM_TASK=0
ONLY_TASK=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY_RUN=true; shift ;;
    --from) FROM_TASK=$2; shift 2 ;;
    --only) ONLY_TASK=$2; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

# Find all pending task files, sorted by name
get_pending_tasks() {
  for f in "$TASKS_DIR"/*.md; do
    [ -f "$f" ] || continue
    if grep -q "^## Status: pending" "$f"; then
      echo "$f"
    fi
  done | sort
}

# Extract task number from filename (e.g., 01 from 01-verify-compression-quality.md)
task_number() {
  basename "$1" | grep -o '^[0-9]*' | sed 's/^0*//'
}

# Extract task title from first heading
task_title() {
  head -1 "$1" | sed 's/^# //'
}

mark_status() {
  local file="$1" old_status="$2" new_status="$3"
  sed -i '' "s/^## Status: ${old_status}/## Status: ${new_status}/" "$file"
}

echo "=== SWG3 Task Runner ==="
echo ""

PENDING_TASKS=$(get_pending_tasks)

if [ -z "$PENDING_TASKS" ]; then
  echo "No pending tasks found."
  exit 0
fi

echo "Pending tasks:"
for f in $PENDING_TASKS; do
  num=$(task_number "$f")
  title=$(task_title "$f")
  echo "  [$num] $title"
done
echo ""

for TASK_FILE in $PENDING_TASKS; do
  NUM=$(task_number "$TASK_FILE")
  TITLE=$(task_title "$TASK_FILE")

  # Skip tasks before --from
  if [ "$NUM" -lt "$FROM_TASK" ]; then
    continue
  fi

  # Skip tasks not matching --only
  if [ -n "$ONLY_TASK" ] && [ "$NUM" != "$ONLY_TASK" ]; then
    continue
  fi

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Task $NUM: $TITLE"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if $DRY_RUN; then
    echo "  [DRY RUN] Would process: $TASK_FILE"
    echo ""
    continue
  fi

  TASK_CONTENT=$(cat "$TASK_FILE")
  LOG_FILE="$LOG_DIR/$(basename "$TASK_FILE" .md).log"

  # Mark as in_progress
  mark_status "$TASK_FILE" "pending" "in_progress"

  PROMPT="You are working on the SWG3 image compression project at /Users/jarkko/_dev/picture.
This is a WebGPU browser app (in docs/) implementing the Switching Games compression algorithm.
Python reference: switching_compress.py. Browser app: docs/app.js, docs/gpu-compress.js, docs/gpu-fft.js.

Here is your current task:

${TASK_CONTENT}

Instructions:
- Read all relevant files before making changes.
- Implement the task following the steps listed.
- Test your changes where possible (run the local server, check for syntax errors, etc.).
- When done, update the task file status from 'in_progress' to 'completed'.
- If the task cannot be completed (blocked, needs clarification), set status to 'blocked' and add a note.
- Commit your changes with a descriptive message.
- Be concise. Do not over-engineer."

  echo "  Starting at $(date '+%H:%M:%S')..."
  echo ""

  # Run Claude Code with the task prompt
  if claude --print --dangerously-skip-permissions -p "$PROMPT" 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo "  Task $NUM finished at $(date '+%H:%M:%S')"

    # Check if task was marked completed
    if grep -q "^## Status: completed" "$TASK_FILE"; then
      echo "  Status: COMPLETED"
    elif grep -q "^## Status: blocked" "$TASK_FILE"; then
      echo "  Status: BLOCKED"
      echo "  Stopping task runner (blocked task)."
      break
    else
      echo "  Status: UNKNOWN (Claude may not have updated status)"
      mark_status "$TASK_FILE" "in_progress" "needs_review"
    fi
  else
    echo ""
    echo "  Task $NUM FAILED at $(date '+%H:%M:%S')"
    mark_status "$TASK_FILE" "in_progress" "failed"
    echo "  Stopping task runner (failed task)."
    break
  fi

  echo ""
done

echo ""
echo "=== Task Runner Complete ==="
echo ""
echo "Summary:"
for f in "$TASKS_DIR"/*.md; do
  [ -f "$f" ] || continue
  num=$(task_number "$f")
  title=$(task_title "$f")
  status=$(grep "^## Status:" "$f" | head -1 | sed 's/^## Status: //')
  printf "  [%s] %s — %s\n" "$num" "$title" "$status"
done
