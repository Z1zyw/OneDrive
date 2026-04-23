CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29566}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

cd /nfs/dataset-ofs-voyager-research/ywzhang/code/Solve
source /nfs/dataset-ofs-voyager-research/ywzhang/anaconda3/etc/profile.d/conda.sh
conda activate qwen25-univla

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --use_env \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --eval bbox
