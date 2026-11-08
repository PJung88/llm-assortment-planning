"""Stage 08: the Plackett-Luce fit behind the substitution shares."""
import numpy as np


def sample_rankings(rng, w, n):
    """Draw n full rankings from a Plackett-Luce model with strengths w."""
    out = []
    for _ in range(n):
        left, r = list(range(len(w))), []
        while left:
            p = w[left] / w[left].sum()
            pick = rng.choice(len(left), p=p)
            r.append(left.pop(pick))
        out.append(r)
    return np.array(out)


def test_neutral_pseudo_rankings_put_every_item_in_every_position(stage08):
    R = stage08.neutral_rankings(7)
    for pos in range(7):
        assert sorted(R[:, pos]) == list(range(7))


def test_neutral_rankings_alone_give_equal_strengths(stage08):
    R = np.empty((0, 7), dtype=int)
    w, _, ok = stage08.fit_pl(R, np.empty(0), smoothing=2.0)
    assert ok and np.allclose(w, 1 / 7)


def test_fit_recovers_known_strengths(stage08):
    rng = np.random.default_rng(0)
    true = np.array([0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03])
    R = sample_rankings(rng, true, 4000)
    w, _, ok = stage08.fit_pl(R, np.ones(len(R)), smoothing=1e-6)
    assert ok
    assert np.allclose(w, true, atol=0.01)
    assert list(np.argsort(-w)) == list(range(7))


def test_smoothing_prevents_zero_strength_for_never_chosen_item(stage08):
    R = np.array([[0, 1, 2, 3, 4, 5, 6]] * 16)          # 16 identical rankings
    w, _, ok = stage08.fit_pl(R, np.ones(16), smoothing=2.0)
    assert ok and (w > 0).all() and w[0] < 0.99
    assert np.all(np.diff(w) < 0)                          # order preserved
