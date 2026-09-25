#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python_path=${PLATEAI_SERVICE_PYTHON:-"$root/.venv-service/bin/python"}
config_path="$root/runs/plate-service/config.json"
device=auto
bind_address=127.0.0.1
port=1788

usage() {
    cat <<'EOF'
Usage: ./run_api_1788.sh [options]

Start the local 3waPlateAI microservice. It binds to 127.0.0.1:1788 by default.

Options:
  --python PATH   Service Python interpreter (default: PLATEAI_SERVICE_PYTHON or .venv-service/bin/python)
  --config PATH   Explicit trusted model manifest (default: runs/plate-service/config.json)
  --device MODE   auto or cpu (default: auto)
  --host ADDRESS  Bind address (default: 127.0.0.1)
  --port PORT     Bind port (default: 1788)
  -h, --help      Show this help
EOF
}

while (($#)); do
    case "$1" in
        --python)
            python_path=${2:?--python requires a path}
            shift 2
            ;;
        --config)
            config_path=${2:?--config requires a path}
            shift 2
            ;;
        --device)
            device=${2:?--device requires auto or cpu}
            shift 2
            ;;
        --host)
            bind_address=${2:?--host requires an address}
            shift 2
            ;;
        --port)
            port=${2:?--port requires a port number}
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$device" in
    auto|cpu) ;;
    *)
        printf 'Unsupported device: %s\n' "$device" >&2
        exit 2
        ;;
esac

if [[ ! -x "$python_path" ]]; then
    printf 'Service Python missing. Run setup_api_1788.sh or set PLATEAI_SERVICE_PYTHON.\n' >&2
    exit 1
fi

if [[ ! -f "$config_path" ]]; then
    printf 'Explicit model manifest missing. See docs/plate-service-api.md.\n' >&2
    exit 1
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_path" -m plateai_service.cli --config "$config_path" --device "$device" --host "$bind_address" --port "$port"
