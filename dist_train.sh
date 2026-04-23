CONFIG=$1
GPUS=$2
NNODES=${NNOES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# 单机 127.0.0.1
# 多机 需要设置 MASTER_ADDR 为主节点 IP 地址

# sudo bash /mnt/common/jianshu/liquidio/common-dataset-mount/common_dateset_mount.sh mount | grep s3_common_dataset 
cd /nfs/dataset-ofs-voyager-research/ywzhang/code/OneDrive

# 通过雪松的 conda 激活
# source /nfs/dataset-ofs-voyager-research/xschen/anaconda3/etc/profile.d/conda.sh
# 在 /home/.conda 下
# conda activate solve


# source /home/luban/anaconda3/bin/activate
# source /home/luban/anaconda3/etc/profile.d/conda.sh
# source /nfs/dataset-ofs-voyager-research/ywzhang/anaconda3/etc/profile.d/conda.sh
# conda activate /nfs/dataset-ofs-voyager-research/ywzhang/envs/qwen25_univla
# conda activate qwen25_univla

source /nfs/dataset-ofs-voyager-research/ywzhang/anaconda3/etc/profile.d/conda.sh
conda activate qwen25-univla

# conda info --envs


# export GLOO_SOCKET_IFNAME=eth0   # 或者你的网卡名称
# export NCCL_SOCKET_IFNAME=eth0   # 如果使用 NCCL
# export NO_IPV6=1                 # 禁用 IPv6
# export NCCL_IB_DISABLE=1   # 禁用 InfiniBand，如果不需要
# export GLOO_SOCKET_IFNAME=eth0  # 或者你的网卡名称，如ens33, enp0s3等
# export NCCL_SOCKET_IFNAME=eth0
# # export MASTER_ADDR=127.0.0.1    # 使用IPv4回环地址
# export MASTER_PORT=29500

# export GLOO_SOCKET_FAMILY=AF_INET
# export NCCL_SOCKET_FAMILY=AF_INET
# export NCCL_DEBUG=INFO

# python your_training_script.py

if [ -z $LUBAN_AVAILBLE_PORT_0 ]; then
    export LUBAN_AVAILBLE_PORT_0=${PORT}
fi
echo "LUBAN_AVAILBLE_PORT_0: ${LUBAN_AVAILBLE_PORT_0}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --use_env \
    --nproc_per_node=$GPUS \
    --master_port=${LUBAN_AVAILBLE_PORT_0} \
    $(dirname "$0")/tools/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}
