#!/usr/bin/env bash
set -euo pipefail

WORKERS="${WORKERS:-12}"
N="${N:-167}"
CHUNK_HOURS="${CHUNK_HOURS:-8}"
POOL_CAP="${POOL_CAP:-4000}"
MAX_PAIRS="${MAX_PAIRS:-3000000}"
CHECK_SECONDS="${CHECK_SECONDS:-30}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ROOT_DIR="$(pwd)"
SCRIPT_NAME="gs_pruned.py"
GS_SCRIPT="$ROOT_DIR/$SCRIPT_NAME"
STOP_FILE="$ROOT_DIR/STOP_GSP_SEARCH"
SUCCESS_FILE="$ROOT_DIR/SUCCESS_GSP_SEARCH.txt"

if [ ! -f "$GS_SCRIPT" ]; then
  echo "Missing $SCRIPT_NAME in $ROOT_DIR"
  exit 1
fi

launch_workers() {
  rm -f "$STOP_FILE" "$SUCCESS_FILE"

  echo "Launching $WORKERS nonstop workers"
  echo "n=$N, order=$((4 * N))"
  echo "Each worker runs in $CHUNK_HOURS hour chunks"
  echo

  for i in $(seq -w 1 "$WORKERS")
  do
    mkdir -p "worker_$i"
    cp "$GS_SCRIPT" "worker_$i/$SCRIPT_NAME"

    cat > "worker_$i/run_forever.sh" <<EOF
#!/usr/bin/env bash
set -u

while true
do
  if [ -f "../STOP_GSP_SEARCH" ]; then
    echo "[forever] stop file found, exiting"
    exit 0
  fi

  if [ -f "hadamard_$((4 * N)).csv" ]; then
    echo "[forever] success CSV found, exiting"
    exit 0
  fi

  echo
  echo "[forever] starting \$(date)"

  caffeinate -dimsu "$PYTHON_BIN" "$SCRIPT_NAME" \\
    --n "$N" \\
    --hours "$CHUNK_HOURS" \\
    --pool-cap "$POOL_CAP" \\
    --max-pairs "$MAX_PAIRS"

  code=\$?
  echo "[forever] stopped with code \$code at \$(date)"

  if [ -f "hadamard_$((4 * N)).csv" ]; then
    echo "[forever] success CSV found, exiting"
    exit 0
  fi

  if [ -f "../STOP_GSP_SEARCH" ]; then
    echo "[forever] stop file found, exiting"
    exit 0
  fi

  echo "[forever] restarting in 10 seconds"
  sleep 10
done
EOF

    chmod +x "worker_$i/run_forever.sh"

    (
      cd "worker_$i"
      nohup ./run_forever.sh > run.log 2>&1 &
      echo $! > pid.txt
    )

    echo "Started worker_$i"
  done
}

stop_workers() {
  touch "$STOP_FILE"

  echo "Stopping all workers"

  for dir in worker_*
  do
    [ -d "$dir" ] || continue

    if [ -f "$dir/pid.txt" ]; then
      pid="$(cat "$dir/pid.txt")"

      if kill -0 "$pid" 2>/dev/null; then
        kill -INT "$pid" 2>/dev/null || true
      fi
    fi
  done

  pkill -f "gs_pruned.py --n $N" 2>/dev/null || true
  pkill -f "caffeinate -dimsu" 2>/dev/null || true
}

watch_success() {
  echo
  echo "Watching for success every $CHECK_SECONDS seconds"
  echo "Leave this terminal open if you want the success message printed here."
  echo

  while true
  do
    for csv in worker_*/hadamard_$((4 * N)).csv
    do
      if [ -f "$csv" ]; then
        echo
        echo "SUCCESS FOUND"
        echo "Verified CSV:"
        echo "$ROOT_DIR/$csv"
        echo

        {
          echo "SUCCESS FOUND"
          echo "Date: $(date)"
          echo "CSV: $ROOT_DIR/$csv"
          echo
          echo "Matching log lines:"
          grep -E "\[SOLVED\]|\[VERIFIED\]" "$(dirname "$csv")/run.log" || true
        } > "$SUCCESS_FILE"

        cat "$SUCCESS_FILE"

        stop_workers

        echo
        echo "All workers stopped."
        echo "Success report saved to:"
        echo "$SUCCESS_FILE"
        exit 0
      fi
    done

    clear
    echo "Still searching. $(date)"
    echo "n=$N, order=$((4 * N)), workers=$WORKERS"
    echo

    for dir in worker_*
    do
      [ -d "$dir" ] || continue

      echo "$dir"

      if [ -f "$dir/run.log" ]; then
        grep "pools" "$dir/run.log" | tail -n 1 | sed 's/^/  /' || true
        tail -n 2 "$dir/run.log" | sed 's/^/  /'
      else
        echo "  no log yet"
      fi

      echo
    done

    sleep "$CHECK_SECONDS"
  done
}

monitor_workers() {
  echo "Worker status"
  echo

  for dir in worker_*
  do
    [ -d "$dir" ] || continue

    echo "$dir"

    if [ -f "$dir/pid.txt" ]; then
      pid="$(cat "$dir/pid.txt")"

      if kill -0 "$pid" 2>/dev/null; then
        echo "  status: running"
      else
        echo "  status: not running"
      fi
    else
      echo "  status: no pid file"
    fi

    if [ -f "$dir/gsp_pool_${N}.npz" ]; then
      echo "  pool size: $(du -h "$dir/gsp_pool_${N}.npz" | awk '{print $1}')"
    else
      echo "  pool size: none yet"
    fi

    if [ -f "$dir/hadamard_$((4 * N)).csv" ]; then
      echo "  CSV FOUND: $dir/hadamard_$((4 * N)).csv"
    fi

    if [ -f "$dir/run.log" ]; then
      echo "  latest:"
      tail -n 3 "$dir/run.log" | sed 's/^/    /'
    fi

    echo
  done
}

run_supervisor() {
  launch_workers
  watch_success
}

cmd="${1:-help}"

case "$cmd" in
  run)
    run_supervisor
    ;;
  launch)
    launch_workers
    ;;
  watch)
    watch_success
    ;;
  monitor)
    monitor_workers
    ;;
  stop)
    stop_workers
    echo "Stopped."
    ;;
  *)
    echo "Commands:"
    echo "  ./gsp_forever_runner.sh run"
    echo "  ./gsp_forever_runner.sh launch"
    echo "  ./gsp_forever_runner.sh watch"
    echo "  ./gsp_forever_runner.sh monitor"
    echo "  ./gsp_forever_runner.sh stop"
    echo
    echo "Best command:"
    echo "  ./gsp_forever_runner.sh run"
    ;;
esac
