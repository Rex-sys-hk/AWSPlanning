# conda env list && 
# conda init &&

source ~/miniconda3/bin/activate &&
# conda activate AIRP &&
# python plan_utils/routing.py &
conda activate isaac-sim &&
source $ISAACSIM_SA &&
export https_proxy=http://127.0.0.1:7890; \
export http_proxy=http://127.0.0.1:7890; \
export all_proxy=socks5://127.0.0.1:7890 &&
python AGV_env_sync.py
# python plan_utils/routing.py 
