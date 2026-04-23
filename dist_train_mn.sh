CONFIG=$1
GPUS=$2
DISTRIBUTED_NODE_COUNT=$3

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

########################## Luban 变量 ########################
GPU_PER_NODE=${RESOURCE_NUM_GPU:?unset}
NODE_COUNT=${DISTRIBUTED_NODE_COUNT:?unset}
NODE_RANK=${DISTRIBUTED_NODE_RANK:-0}          # rank0 默认为 0
MASTER_FQDN=${DISTRIBUTED_MASTER_HOSTS:?unset} # 由 Luban 注入
MASTER_PORT=${LUBAN_AVAILBLE_PORT_0:-29500}

########################## 解析主节点 IPv4 ###################
# 取逗号前第一段 FQDN -> 转 IP；失败直接退出
MASTER_ADDR=$(getent hosts "${MASTER_FQDN%%,*}" | awk '{print $1}' | head -n1)
[[ -z "$MASTER_ADDR" ]] && { echo "[ERROR] cannot resolve $MASTER_FQDN"; exit 1; }

########################## 网络接口选择 ######################
NET_IF=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)

########################## NCCL/GLOO 环境 ####################
export NCCL_SOCKET_IFNAME=$NET_IF
export GLOO_SOCKET_IFNAME=$NET_IF

# 从xuesong那抄的
NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
export NCCL_IB_HCA
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
# 


export NCCL_IB_DISABLE=0 # TODO
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=0 # TODO
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export GLOO_DISABLE_IPV6=1     # 彻底不碰 IPv6
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($GPU_PER_NODE - 1)))

# sudo bash /mnt/common/jianshu/liquidio/common-dataset-mount/common_dateset_mount.sh mount | grep s3_common_dataset 
cd /nfs/dataset-ofs-voyager-research/ywzhang/code/Solve
# source /nfs/dataset-ofs-voyager-research/xschen/anaconda3/etc/profile.d/conda.sh
# conda activate solve

source /nfs/dataset-ofs-voyager-research/ywzhang/anaconda3/etc/profile.d/conda.sh
conda activate qwen25-univla

########################## 打印参数 ##########################
echo "========== Distributed Launch =========="
echo "CFG            : $CFG"
# echo "WORKDIR        : $WORKDIR"
echo "GPUS / node    : $GPU_PER_NODE"
echo "NNODES         : $NODE_COUNT"
echo "NODE_RANK      : $NODE_RANK"
echo "MASTER_ADDR    : $MASTER_ADDR"
echo "MASTER_PORT    : $MASTER_PORT"
echo "NET_IF         : $NET_IF"
echo "========================================"

########################## Torchrun 启动 #####################
export DISTRIBUTED_NODE_RANK=$NODE_RANK
export DISTRIBUTED_NODE_COUNT=$NODE_COUNT

# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


# # export GLOO_SOCKET_FAMILY=ipv4
# export GLOO_SOCKET_IFNAME=eth0   # 或者你的网卡名称
# export NCCL_SOCKET_IFNAME=eth0   # 如果使用 NCCL
# export NO_IPV6=1                 # 禁用 IPv6
# export NCCL_IB_DISABLE=1   # 禁用 InfiniBand，如果不需要
# export GLOO_SOCKET_IFNAME=eth0  # 或者你的网卡名称，如ens33, enp0s3等
# export NCCL_SOCKET_IFNAME=eth0
# # export MASTER_ADDR=127.0.0.1    # 使用IPv4回环地址
# # export MASTER_PORT=29500

# export GLOO_SOCKET_FAMILY=AF_INET
# export NCCL_SOCKET_FAMILY=AF_INET
# export NCCL_DEBUG=INFO

if [ -z $LUBAN_AVAILBLE_PORT_0 ]; then
    export LUBAN_AVAILBLE_PORT_0=${PORT}
fi
echo "LUBAN_AVAILBLE_PORT_0: ${LUBAN_AVAILBLE_PORT_0}"

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --use_env \
#     --nproc_per_node=$GPUS \
#     --master_port=${LUBAN_AVAILBLE_PORT_0} \
#     $(dirname "$0")/tools/train.py \
#     $CONFIG \
#     --seed 0 \
#     --launcher pytorch ${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nnodes=$NODE_COUNT \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=${LUBAN_AVAILBLE_PORT_0} \
    $(dirname "$0")/tools/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch

    
    # --launcher pytorch ${@:3}