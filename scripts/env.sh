# Source this before running any zombie_crawl / Isaac Lab script on this host.
#
#   source /home/yousof/robotics/experimental/zombie_crawl/scripts/env.sh
#   python scripts/train.py --task Zombie-Crawl-SO100-v0 --num_envs 64 --max_iterations 100 --headless
#
# WHY these things are required on THIS machine:
#
#   1. CONDA_PREFIX/PATH -> the project-local conda prefix env at .venv
#      (Python 3.12, Isaac Sim 6.1.0.0, Isaac Lab 3.0, torch 2.11.0+cu130).
#      It is a conda *prefix* env: there is no bin/activate, so we export
#      CONDA_PREFIX and PATH manually.
#
#   2. OMNI_KIT_ACCEPT_EULA=YES -> non-interactive (background/headless) Kit
#      runs must not hit the EULA prompt.
#
#   3. LD_LIBRARY_PATH -> project-local clean libxml2 2.12.10 (C-only build).
#      Ubuntu 26.04 ships only libxml2.so.16, but the Isaac Sim URDF asset
#      importer links against libxml2.so.2. Only needed for URDF conversion
#      (scripts/convert_asset.py); harmless otherwise.
#
# GUI vs headless:
#   headless : add --headless (or export HEADLESS=1)
#   GUI      : add --viz kit   (in Isaac Lab 3.0 the launcher forces headless
#                unless --viz kit is requested; DISPLAY must point at an X server)

export CONDA_PREFIX=/home/yousof/robotics/experimental/.venv
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export LD_LIBRARY_PATH="/home/yousof/robotics/experimental/.libxml2/lib:${LD_LIBRARY_PATH}"

echo "[env.sh] .venv + EULA + libxml2 ready."
echo "  headless: ... --headless      GUI: ... --viz kit"
