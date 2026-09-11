import numpy as np

from scripts import find_songling_outlier_frames as scanner


def test_scan_episode_reports_raw_jump_and_training_delta():
    qpos = np.zeros((6, 14), dtype=np.float32)
    qpos[3, 2] = 100.0

    findings = scanner.scan_episode(
        qpos,
        episode_uri="s3://bucket/episode/",
        dimensions=(2,),
        action_horizon=2,
        action_start_offset=1,
        top_k=1,
    )

    by_kind = {finding.kind: finding for finding in findings}
    assert by_kind["absolute_value"].source_frame == 3
    assert by_kind["absolute_value"].source_value == 100.0
    assert abs(by_kind["adjacent_jump"].delta) == 100.0
    assert abs(by_kind["training_delta"].delta) == 100.0
    assert by_kind["training_delta"].target_frame - by_kind["training_delta"].source_frame in (1, 2)


def test_scan_episode_reproduces_end_of_episode_clamping():
    qpos = np.zeros((3, 14), dtype=np.float32)
    qpos[:, 7] = [0.0, 1.0, 3.0]

    findings = scanner.scan_episode(
        qpos,
        episode_uri="s3://bucket/episode/",
        dimensions=(7,),
        action_horizon=3,
        action_start_offset=1,
        top_k=9,
    )

    training_findings = [finding for finding in findings if finding.kind == "training_delta"]
    terminal = [finding for finding in training_findings if finding.source_frame == 1 and finding.horizon == 3]
    assert len(terminal) == 1
    assert terminal[0].target_frame == 2
    assert terminal[0].delta == 2.0


def test_top_findings_keeps_largest_per_kind_and_dimension():
    top = scanner.TopFindings(top_k=1)
    for value in (1.0, -3.0, 2.0):
        top.add(
            scanner.Finding(
                kind="adjacent_jump",
                episode_uri="s3://bucket/episode/",
                dimension=2,
                source_frame=0,
                target_frame=1,
                horizon=1,
                source_value=0.0,
                target_value=value,
                delta=value,
            )
        )

    findings = top.sorted_findings()
    assert len(findings) == 1
    assert findings[0].delta == -3.0
