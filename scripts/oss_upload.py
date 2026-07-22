"""
export AWS_ACCESS_KEY_ID="ALIBABA_CLOUD_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="ALIBABA_CLOUD_ACCESS_KEY_SECRET"

# 上传整个目录
aws s3 cp /home/fanyiming/openpi/checkpoints/pi05_base_pytorch \
    s3://shengshu-base2-test/ali-checkpoint/fanyiming/pi05_base_pytorch \
    --recursive \
    --endpoint-url https://oss-cn-wulanchabu.aliyuncs.com

# 上传单个文件
aws s3 cp /path/to/file.pt \
    s3://shengshu-base2-test/ali-checkpoint/fanyiming/file.pt \
    --endpoint-url https://oss-cn-wulanchabu.aliyuncs.com
"""
import boto3
import os
from botocore.config import Config

# 创建客户端（和项目中 ario_dataset.py 一致）
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    endpoint_url="https://oss-cn-wulanchabu.aliyuncs.com",  # 外网用这个
    # endpoint_url="https://oss-cn-wulanchabu-internal.aliyuncs.com",  # 内网用这个
    region_name="cn-wulanchabu",
    config=Config(s3={"addressing_style": "virtual"}, signature_version="s3v4"),
)

BUCKET = "shengshu-base2-test"

# 上传本地文件到 OSS
# s3.upload_file(
#     Filename="/path/to/local/file.pt",
#     Bucket=BUCKET,
#     Key="ario/xingchen/your_folder/file.pt",  # OSS 上的路径
# )

# 上传整个目录
local_dir = "/home/fanyiming/openpi/checkpoints/pi05_base_pytorch"
oss_prefix = "ali-checkpoint/fanyiming/"
for root, dirs, files in os.walk(local_dir):
    for f in files:
        local_path = os.path.join(root, f)
        relative = os.path.relpath(local_path, local_dir)
        s3.upload_file(local_path, BUCKET, oss_prefix + relative)
