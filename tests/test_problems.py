import numpy as np
import pytest

from slud import problems

POWELL_OPT = [1.098159329699759e-5, 9.106146739867318]

OPTIMA = [
    (problems.rosenbrock, [1.0, 1.0]),
    (problems.powell_badly_scaled, POWELL_OPT),
    (problems.powell_flipped, [POWELL_OPT[0], -POWELL_OPT[1]]),
    (problems.brown_badly_scaled, [1e6, 2e-6]),
    (problems.brown_flipped, [1e6, -2e-6]),
    (problems.cp_fit_c3h8, problems.C3H8_A),
]


@pytest.mark.parametrize("f, x_opt", OPTIMA)
def test_optimum_is_zero(f, x_opt):
    assert f(np.array(x_opt)) < 1e-12


@pytest.mark.parametrize("f, x_opt", OPTIMA)
def test_population_call_matches_row_by_row(f, x_opt):
    X = np.array(x_opt) * np.random.default_rng(0).uniform(0.5, 1.5, size=(50, len(x_opt)))
    np.testing.assert_allclose(f(X), [f(x) for x in X])
    assert np.all(f(X) > 0)


def test_cp_reference_is_physical():
    # propane cp/R between 200 K and 1000 K is positive and rising
    assert np.all(problems.CP_REF > 0)
    assert np.all(np.diff(problems.CP_REF) > 0)


def test_cp_tradeoff_front_is_exact_and_conflicting():
    pareto_set = problems.cp_tradeoff_pareto_set(200)
    front = problems.cp_tradeoff_c3h8(pareto_set)
    # sorted by f_low, the front must be strictly decreasing in f_high (a true trade-off)
    order = np.argsort(front[:, 0])
    assert np.all(np.diff(front[order, 1]) < 0)
    # no random fit beats the front: each sample is dominated by some front point
    X = pareto_set[100] * np.random.default_rng(0).uniform(0.9, 1.1, size=(500, 4))
    F = problems.cp_tradeoff_c3h8(X)
    dominated = np.all(front[None, :, :] <= F[:, None, :] + 1e-12, axis=2).any(axis=1)
    assert dominated.all()
    # the Pareto set crosses zero in a1 and a4: the case where the sign handling matters
    assert np.ptp(np.sign(pareto_set[:, 0])) == 2 and np.ptp(np.sign(pareto_set[:, 3])) == 2
