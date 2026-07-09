#!/bin/bash
set -e

# Activate environment
eval "$(/miniconda3/bin/conda shell.bash hook)" && conda activate openpi 2>/dev/null || \
    source /home/fanyiming/.conda/envs/openpi/bin/activate

cd /home/fanyiming/openpi

# Install openpi + boto3
pip install -e . --no-deps 2>/dev/null || true
pip install boto3 -q 2>/dev/null || true

# OSS credentials (read from environment)
export AWS_ACCESS_KEY_ID="${ALIBABA_ACCESS_KEY_ID:?Set ALIBABA_ACCESS_KEY_ID env var}"
export AWS_SECRET_ACCESS_KEY="${ALIBABA_ACCESS_KEY_SECRET:?Set ALIBABA_ACCESS_KEY_SECRET env var}"

S3_ENDPOINT="https://oss-cn-wulanchabu-internal.aliyuncs.com"

# S3 prefixes for all date directories
S3_PREFIXES="s3://shengshu-base2-test/ario/xingchen/xingchen3-Pretrain_XC03_叠短袖_260623_01/,\
s3://shengshu-base2-test/ario/xingchen/xingchen3-Pretrain_XC03_叠短袖_260624_01/,\
s3://shengshu-base2-test/ario/xingchen/xingchen3-Pretrain_XC03_叠短袖_260625_01/,\
s3://shengshu-base2-test/ario/xingchen/xingchen3-Pretrain_XC03_叠短袖_260626_01/,\
s3://shengshu-base2-test/ario/xingchen/xingchen3-Pretrain_XC03_叠短袖_260629_01/,\
s3://shengshu-base2-test/ario/xingchen/xingchen3-Pretrain_XC03_叠短袖_260630_01/,\
s3://shengshu-base2-test/ario/xingchen/xingchen3-Pretrain_XC03_叠短袖_260703_01/"

echo "=== Running data conversion (streaming from OSS) ==="
python examples/xingchen/convert_ario_to_lerobot.py \
    --s3_prefixes "$S3_PREFIXES" \
    --s3_endpoint "$S3_ENDPOINT" \
    --video_downsample_rate 6 \
    --min_frames 1885

echo "=== Computing normalization statistics ==="
python scripts/compute_norm_stats.py pi05_xingchen_fold

echo "=== All done ==="
