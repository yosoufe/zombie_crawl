# Source this before running any zombie_crawl / Isaac Lab script:
#
#   source scripts/env.sh
#   python scripts/train.py --task Zombie-Crawl-SO100-v0 --num_envs 64 --max_iterations 100
#
# WHAT this sets up (and why):
#
#   1. Python environment -> the project env (Python 3.12 with Isaac Lab +
#      Isaac Sim + torch, per README 'Setup'). Resolution order:
#        a. $ZOMBIE_CRAWL_ENV if set,
#        b. <repo>/.venv, then <repo>/../.venv (created by the README setup).
#      The project env WINS over an already-activated env (conda base is active
#      in every conda shell and does NOT have this project's packages, so it
#      must not be treated as "the user's env"). An already-activated env is
#      only used when no project env exists; to use your own env instead of the
#      project one, export ZOMBIE_CRAWL_ENV pointing at it.
#      A plain venv has bin/activate; a conda *prefix* env does not, so for it
#      we export CONDA_PREFIX and PATH manually. Both are supported below.
#
#   2. OMNI_KIT_ACCEPT_EULA=YES -> non-interactive (background/headless) Kit
#      runs must not hit the EULA prompt.
#
#   3. LD_LIBRARY_PATH -> clean libxml2 2.12.10 (C-only build), if present at
#      $ZOMBIE_CRAWL_LIBXML2/lib or <repo>/.libxml2/lib. Ubuntu 26.04 ships
#      only libxml2.so.16, but the Isaac Sim URDF asset importer links
#      against libxml2.so.2. Only needed for URDF conversion
#      (scripts/asset_pipeline/convert_asset.py); skipped when absent.
#
# GUI vs headless is a per-run CLI choice, not env: pass --headless or --viz kit
# to the scripts (see README 'Run'). This script never touches HEADLESS.

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _zc_env_self="${BASH_SOURCE[0]}"
else
    _zc_env_self="$0"
fi
ZOMBIE_CRAWL_REPO="$(cd "$(dirname "${_zc_env_self}")/.." && pwd)"
unset _zc_env_self

# 1. Python environment -------------------------------------------------------
if [ -n "${ZOMBIE_CRAWL_ENV:-}" ]; then
    _zc_env_candidates="${ZOMBIE_CRAWL_ENV}"
else
    _zc_env_candidates="${ZOMBIE_CRAWL_REPO}/.venv ${ZOMBIE_CRAWL_REPO}/../.venv"
fi
_zc_env=""
for _zc_try in ${_zc_env_candidates}; do
    if [ -x "${_zc_try}/bin/python" ]; then
        _zc_env="${_zc_try}"
        break
    fi
done
if [ -n "${_zc_env}" ]; then
    if [ -f "${_zc_env}/bin/activate" ]; then
        # shellcheck disable=SC1091
        . "${_zc_env}/bin/activate"
        echo "[env.sh] activated venv: ${_zc_env}"
    else
        # conda prefix env: no bin/activate; shadow whatever conda/venv was
        # active (e.g. conda base) by prepending on PATH
        export CONDA_PREFIX="${_zc_env}"
        export PATH="${CONDA_PREFIX}/bin:${PATH}"
        unset VIRTUAL_ENV
        echo "[env.sh] activated conda prefix env: ${_zc_env}"
    fi
else
    if [ -n "${CONDA_PREFIX:-}" ] || [ -n "${VIRTUAL_ENV:-}" ]; then
        echo "[env.sh] WARNING: no project env in: ${_zc_env_candidates}" >&2
        echo "[env.sh]   using already-active python: $(command -v python)" >&2
        echo "[env.sh]   (it must have this project's deps; see README 'Setup')" >&2
    else
        echo "[env.sh] WARNING: no project env in: ${_zc_env_candidates}" >&2
        echo "[env.sh]   Create it (see README 'Setup'), or point" >&2
        echo "[env.sh]   ZOMBIE_CRAWL_ENV at an existing conda-prefix/venv." >&2
        echo "[env.sh]   Falling back to whatever python is on PATH: $(command -v python || echo none)" >&2
    fi
fi
unset _zc_env _zc_env_candidates _zc_try

# 2. EULA ---------------------------------------------------------------------
export OMNI_KIT_ACCEPT_EULA=YES

# 3. libxml2 (URDF conversion only) --------------------------------------------
_zc_libxml2="${ZOMBIE_CRAWL_LIBXML2:-${ZOMBIE_CRAWL_REPO}/.libxml2}"
if [ -d "${_zc_libxml2}/lib" ]; then
    export LD_LIBRARY_PATH="${_zc_libxml2}/lib:${LD_LIBRARY_PATH:-}"
    echo "[env.sh] libxml2 from ${_zc_libxml2}/lib on LD_LIBRARY_PATH"
else
    echo "[env.sh] libxml2 not found (only needed for URDF conversion; see README)"
fi
unset _zc_libxml2

echo "[env.sh] ready."
