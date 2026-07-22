"""
export AWS_ACCESS_KEY_ID="ALIBABA_CLOUD_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="ALIBABA_CLOUD_ACCESS_KEY_SECRET"

# 下载整个目录
aws s3 cp s3://shengshu-base2-test/ali-checkpoint/fanyiming/pi05_base_pytorch \
    /home/fanyiming/openpi_Ario/checkpoints/pi05_base_pytorch \
    --recursive \
    --endpoint-url https://oss-cn-wulanchabu.aliyuncs.com
    > download.log 2>&1

# 下载单个文件
aws s3 cp s3://shengshu-base2-test/ali-checkpoint/fanyiming/file.pt \
    /path/to/local/file.pt \
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

# 下载单个文件
s3.download_file(
    Bucket=BUCKET,
    Key="ario/xingchen/your_folder/file.pt",
    Filename="/path/to/save/file.pt",
)

# 列出并下载目录下所有文件
prefix = "ario/xingchen/your_folder/"
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        local_path = "/path/to/save/" + key[len(prefix):]
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(BUCKET, key, local_path)
