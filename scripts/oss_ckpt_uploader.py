"""Upload committed Orbax checkpoints to Aliyun OSS.

This sidecar watches a local training run, uploads each fully committed step
through the S3-compatible API, verifies every remote file by size, and only
then removes the local step. Repeated runs are idempotent: files already
present with the expected size are skipped.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import os
import shutil
import sys
import time
from typing import Any

for proxy_var in (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
):
    os.environ.pop(proxy_var, None)

import boto3  # noqa: E402
from botocore.config import Config  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402

CHECKPOINT_METADATA = "_CHECKPOINT_METADATA"


def log(message: str) -> None:
    print(f"[uploader] {time.strftime('%H:%M:%S')} {message}", flush=True)


def committed_steps(run_dir: str) -> list[int]:
    """Return step directories containing Orbax's commit marker."""
    if not os.path.isdir(run_dir):
        return []

    steps = []
    for name in os.listdir(run_dir):
        step_dir = os.path.join(run_dir, name)
        if (
            name.isdigit()
            and os.path.isdir(step_dir)
            and os.path.exists(os.path.join(step_dir, CHECKPOINT_METADATA))
        ):
            steps.append(int(name))
    return sorted(steps)


def local_files(step_dir: str) -> list[str]:
    files = []
    for root, _, names in os.walk(step_dir):
        files.extend(os.path.join(root, name) for name in names)
    return files


def parse_oss_uri(uri: str) -> tuple[str, str]:
    for scheme in ("s3+ali://", "oss://", "s3://"):
        if uri.startswith(scheme):
            path = uri[len(scheme) :]
            break
    else:
        raise ValueError(f"Unsupported OSS URI: {uri}")
    bucket, separator, prefix = path.partition("/")
    if not bucket or not separator:
        raise ValueError(f"OSS URI must include bucket and prefix: {uri}")
    return bucket, prefix.rstrip("/")


def create_oss_client() -> Any:
    access_key = os.environ.get("ALI__AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("ALI__AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("ALI__OSS_ENDPOINT", "https://oss-cn-wulanchabu.aliyuncs.com")
    addressing_style = os.environ.get("ALI__AWS_S3_ADDRESSING_STYLE", "virtual")
    if not access_key or not secret_key:
        raise RuntimeError("ALI__AWS_ACCESS_KEY_ID and ALI__AWS_SECRET_ACCESS_KEY are required")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="cn-wulanchabu",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": addressing_style},
            # Newer botocore versions otherwise use aws-chunked request
            # trailers, which Aliyun OSS's S3-compatible API rejects.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def remote_size(client: Any, bucket: str, key: str) -> int | None:
    try:
        return int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def upload_step(
    client: Any,
    bucket: str,
    remote_prefix: str,
    step_dir: str,
    step: int,
    workers: int,
) -> None:
    """Upload and verify every file in one checkpoint step."""
    files = local_files(step_dir)
    if not files:
        raise RuntimeError(f"Checkpoint contains no files: {step_dir}")
    step_prefix = f"{remote_prefix}/{step}"
    log(f"uploading {step_dir} ({len(files)} files) -> s3://{bucket}/{step_prefix}")

    def upload_one(local_path: str) -> None:
        relative_path = os.path.relpath(local_path, step_dir).replace(os.sep, "/")
        destination = f"{step_prefix}/{relative_path}"
        expected_size = os.path.getsize(local_path)
        if remote_size(client, bucket, destination) == expected_size:
            return
        client.upload_file(local_path, bucket, destination)

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(upload_one, path): path for path in files}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append((futures[future], repr(error)))

    if errors:
        for path, error in errors[:20]:
            log(f"ERROR {path}: {error}")
        raise RuntimeError(f"{len(errors)} file(s) failed to upload")

    invalid = []
    for local_path in files:
        relative_path = os.path.relpath(local_path, step_dir).replace(os.sep, "/")
        destination = f"{step_prefix}/{relative_path}"
        if remote_size(client, bucket, destination) != os.path.getsize(local_path):
            invalid.append(relative_path)
    if invalid:
        raise RuntimeError(f"Remote verification failed for {len(invalid)} file(s): {invalid[:10]}")

    log(f"verified {len(files)} files for {step_dir}")


def sync_run_file(
    client: Any,
    bucket: str,
    remote_prefix: str,
    local_run_dir: str,
    filename: str,
) -> None:
    source = os.path.join(local_run_dir, filename)
    if not os.path.isfile(source):
        return
    destination = f"{remote_prefix}/{filename}"
    try:
        client.upload_file(source, bucket, destination)
    except Exception as error:
        log(f"{filename} sync failed; will retry: {error!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-run-dir", required=True)
    parser.add_argument("--oss-run-dir", required=True)
    parser.add_argument("--poll-secs", type=int, default=60)
    parser.add_argument("--settle-secs", type=int, default=120)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-last-local", action="store_true")
    parser.add_argument("--max-idle-polls", type=int, default=0)
    args = parser.parse_args()

    bucket, remote_prefix = parse_oss_uri(args.oss_run_dir)
    client = create_oss_client()
    uploaded: set[int] = set()
    idle_polls = 0
    log(f"watching {args.local_run_dir} -> {args.oss_run_dir}")

    while True:
        now = time.time()
        pending = []
        for step in committed_steps(args.local_run_dir):
            if step in uploaded:
                continue
            metadata = os.path.join(args.local_run_dir, str(step), CHECKPOINT_METADATA)
            if now - os.path.getmtime(metadata) >= args.settle_secs:
                pending.append(step)

        for step in pending:
            step_dir = os.path.join(args.local_run_dir, str(step))
            try:
                upload_step(client, bucket, remote_prefix, step_dir, step, args.workers)
                uploaded.add(step)
            except Exception as error:
                log(f"step {step} failed; will retry: {error!r}")

        uploaded_on_disk = [
            step for step in sorted(uploaded) if os.path.isdir(os.path.join(args.local_run_dir, str(step)))
        ]
        to_delete = uploaded_on_disk[:-1] if args.keep_last_local else uploaded_on_disk
        for step in to_delete:
            step_dir = os.path.join(args.local_run_dir, str(step))
            shutil.rmtree(step_dir)
            log(f"deleted local {step_dir}")

        remaining = committed_steps(args.local_run_dir)
        not_uploaded = [step for step in remaining if step not in uploaded]
        if args.max_idle_polls and not not_uploaded and not pending:
            idle_polls += 1
            if idle_polls >= args.max_idle_polls:
                log(f"no pending checkpoints for {idle_polls} polls; exiting")
                return 0
        else:
            idle_polls = 0

        sync_run_file(client, bucket, remote_prefix, args.local_run_dir, "loss.txt")
        time.sleep(args.poll_secs)


if __name__ == "__main__":
    sys.exit(main())
