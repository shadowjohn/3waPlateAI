#!/usr/bin/env bash
set -euo pipefail

profile=cu118

usage() {
    cat <<'EOF'
Usage: ./setup_api_1788.sh [--profile cu118|cu128|cpu]

Create the isolated Python 3.11 environment for the local 1788 API.
The script never installs service dependencies into .venv or downloads models.
EOF
}

while (($#)); do
    case "$1" in
        --profile)
            profile=${2:?--profile requires cu118, cu128, or cpu}
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

case "$profile" in
    cu118|cu128|cpu) ;;
    *)
        printf 'Unsupported profile: %s\n' "$profile" >&2
        exit 2
        ;;
esac

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
service_env="$root/.venv-service"
service_python="$service_env/bin/python"

if [[ -e "$service_env" ]]; then
    printf 'Existing .venv-service is preserved. Use it, or create a separately named environment manually.\n' >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    printf 'uv is required. Install uv first, then run this script again.\n' >&2
    exit 1
fi

uv venv --python 3.11 --seed "$service_env"

torch_index="https://download.pytorch.org/whl/$profile"
uv pip install --python "$service_python" torch==2.7.1 torchvision==0.22.1 --index-url "$torch_index"

if [[ "$profile" == cu118 ]]; then
    uv pip install --python "$service_python" paddlepaddle-gpu==3.2.2 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/
    uv pip install --python "$service_python" -r "$root/requirements/service-common.txt" -c "$root/requirements/service-cu118-observed.txt"
else
    uv pip install --python "$service_python" paddlepaddle==3.2.2
    uv pip install --python "$service_python" -r "$root/requirements/service-common.txt"
fi

uv pip check --python "$service_python"
printf 'Environment ready: %s\n' "$service_python"
printf 'Provide an explicit, trusted model manifest before running run_api_1788.sh.\n'
