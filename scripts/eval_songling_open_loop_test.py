import numpy as np

from scripts import eval_songling_open_loop as evaluator


def test_rmse_accumulator_tracks_mean_and_max_by_step_and_dimension():
    accumulator = evaluator.RmseAccumulator(horizon=3)
    zeros = np.zeros((3, evaluator.ACTION_DIM))
    accumulator.update(np.ones_like(zeros), zeros)
    accumulator.update(np.full((1, evaluator.ACTION_DIM), 2.0), np.zeros((1, evaluator.ACTION_DIM)))

    mean_rmse, max_rmse = accumulator.results()

    np.testing.assert_allclose(mean_rmse[:, 0], [np.sqrt(2.5), 1.0, 1.0])
    np.testing.assert_allclose(max_rmse[:, 0], [2.0, 1.0, 1.0])
    np.testing.assert_array_equal(accumulator.counts, [2, 1, 1])


def test_candidate_indices_exclude_each_episode_terminal_frame():
    indices = evaluator.candidate_indices([3, 2], stride=1)

    np.testing.assert_array_equal(indices, [0, 1, 3])


def test_plot_contains_all_chunk_steps(tmp_path):
    mean_rmse = np.ones((5, evaluator.ACTION_DIM))
    max_rmse = np.full((5, evaluator.ACTION_DIM), 2.0)
    output = tmp_path / "rmse.png"

    evaluator.plot_rmse(output, mean_rmse, max_rmse)

    assert output.is_file()
    assert output.stat().st_size > 0
