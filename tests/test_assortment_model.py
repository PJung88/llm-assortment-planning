"""Stage 10: the customer-choice evaluator and the exact MILP."""
import numpy as np


def test_hand_calculated_cases(stage10):
    stage10.hand_tests()        # raises CheckFailed on any mismatch


def test_milp_matches_brute_force_on_random_instances(stage10):
    df = stage10.validate(n_instances=8, seed=123)
    assert df["match"].all()


def test_evaluator_accounts_for_every_customer(stage10):
    rng = np.random.default_rng(1)
    n = 12
    W = rng.uniform(0, 1, (n, n)) * (rng.uniform(size=(n, n)) < 0.4)
    np.fill_diagonal(W, 0)
    d = rng.uniform(0.5, 2, n)
    for alpha in (0.0, 0.45, 0.75):
        x = (rng.uniform(size=n) < 0.5).astype(int)
        r = stage10.evaluate(x, d, W, alpha)
        assert np.isclose(r["direct"] + r["switched"] + r["walked"], d.sum())


def test_switching_rate_equals_alpha_when_all_candidates_stocked(stage10):
    W = np.zeros((3, 3)); W[0, 1], W[0, 2] = 0.7, 0.3   # rescaled weights sum to 1
    x = np.array([0, 1, 1])
    for alpha in (0.45, 0.60, 0.75):
        q0, Q = stage10.choice_shares(x, W, alpha)
        assert np.isclose(Q[0].sum(), alpha) and np.isclose(q0[0], 1 - alpha)
