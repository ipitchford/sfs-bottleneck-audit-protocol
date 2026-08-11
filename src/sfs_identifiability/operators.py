"""Linear operators linking population size in coalescent time to the expected SFS.

The transformed history is h(tau) = dt/dtau, where
    tau(t) = integral_0^t 1 / N_e(u) du.
For a sample of n haplotypes, the first-coalescence moments are
    c_m = integral_0^infinity h(tau) exp(-choose(m, 2) tau) dtau.
The expected unfolded SFS is an invertible linear transform of these moments.

This module also constructs the expected branch-length spectrum directly from
the pure-death coalescent generator. With mutation rate set to one per unit
branch length, a constant-size population has expectation 2 / i in frequency
class i.
"""

from __future__ import annotations

from math import comb
from typing import Iterable, Tuple

import numpy as np
from scipy.linalg import expm


def coalescent_rates(n: int) -> np.ndarray:
    """Return choose(m, 2) for m=2,...,n."""
    if n < 2:
        raise ValueError("n must be at least 2")
    m = np.arange(2, n + 1, dtype=float)
    return m * (m - 1.0) / 2.0


def coalescent_generator(n: int) -> np.ndarray:
    """Transient column generator for lineage counts 2,...,n.

    If p(tau) is a column vector, p'(tau) = Q p(tau). State k loses mass at
    rate choose(k, 2) and moves to k-1 when k>2. State 1 is absorbing and is
    excluded from this transient generator.
    """
    rates = coalescent_rates(n)
    q = np.zeros((n - 1, n - 1), dtype=float)
    for k in range(2, n + 1):
        idx = k - 2
        rate = rates[idx]
        q[idx, idx] = -rate
        if k > 2:
            q[idx - 1, idx] = rate
    return q


def branch_count_matrix(n: int) -> np.ndarray:
    """Expected number of b-descendant branches conditional on k lineages.

    Rows correspond to derived count b=1,...,n-1 and columns to lineage count
    k=2,...,n. The formula is

        B[b,k] = k * C(n-b-1, k-2) / C(n-1, k-1),

    with impossible combinations set to zero.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    bmat = np.zeros((n - 1, n - 1), dtype=float)
    for b in range(1, n):
        for k in range(2, n + 1):
            if n - b - 1 >= k - 2 and n - b - 1 >= 0:
                bmat[b - 1, k - 2] = (
                    k * comb(n - b - 1, k - 2) / comb(n - 1, k - 1)
                )
    return bmat


def moment_grid_operator(
    n: int, edges: Iterable[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map piecewise-constant h(tau) to first-coalescence moments.

    Returns rates, A, tail, where A[m,j] is the integral of exp(-rate_m tau)
    over grid cell j. ``tail`` is the contribution of a unit history after the
    final edge. A user who fixes the tail at one obtains c = A @ h + tail.
    """
    edges_arr = np.asarray(list(edges), dtype=float)
    if edges_arr.ndim != 1 or len(edges_arr) < 2:
        raise ValueError("edges must be a one-dimensional array of length >= 2")
    if not np.all(np.diff(edges_arr) > 0) or edges_arr[0] != 0.0:
        raise ValueError("edges must be strictly increasing and start at zero")
    rates = coalescent_rates(n)
    left = edges_arr[:-1][None, :]
    right = edges_arr[1:][None, :]
    rr = rates[:, None]
    operator = (np.exp(-rr * left) - np.exp(-rr * right)) / rr
    tail = np.exp(-rates * edges_arr[-1]) / rates
    return rates, operator, tail


def direct_sfs_grid_operator(
    n: int, edges: Iterable[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Map piecewise-constant h(tau) directly to expected branch SFS.

    Returns ``G, tail``. For grid values h and a unit tail after the final edge,
    the expected branch-length spectrum is ``G @ h + tail``. Multiplication by
    a mutation-rate constant converts branch lengths to expected mutation
    counts without altering the normalised SFS.
    """
    edges_arr = np.asarray(list(edges), dtype=float)
    if edges_arr.ndim != 1 or len(edges_arr) < 2:
        raise ValueError("edges must be a one-dimensional array of length >= 2")
    if not np.all(np.diff(edges_arr) > 0) or edges_arr[0] != 0.0:
        raise ValueError("edges must be strictly increasing and start at zero")

    q = coalescent_generator(n)
    bmat = branch_count_matrix(n)
    q_inv = np.linalg.inv(q)
    initial = np.zeros(n - 1, dtype=float)
    initial[-1] = 1.0

    columns = []
    for left, right in zip(edges_arr[:-1], edges_arr[1:]):
        integrated_state = q_inv @ (expm(q * right) - expm(q * left)) @ initial
        columns.append(bmat @ integrated_state)
    operator = np.column_stack(columns)
    tail = bmat @ (-q_inv @ expm(q * edges_arr[-1]) @ initial)
    return operator, tail


def expected_sfs_piecewise(
    n: int,
    edges: Iterable[float],
    history: Iterable[float],
    *,
    tail_value: float = 1.0,
) -> np.ndarray:
    """Expected branch-length SFS for a piecewise-constant transformed history."""
    edges_arr = np.asarray(list(edges), dtype=float)
    history_arr = np.asarray(list(history), dtype=float)
    if len(history_arr) != len(edges_arr) - 1:
        raise ValueError("history must have one value per grid cell")
    if np.any(history_arr <= 0) or tail_value <= 0:
        raise ValueError("history and tail_value must be positive")
    operator, tail = direct_sfs_grid_operator(n, edges_arr)
    return operator @ history_arr + tail_value * tail


def piecewise_linear_sfs_operator(
    n: int, nodes: Iterable[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Map continuous piecewise-linear h(tau) node values to the expected SFS.

    ``h`` is linear on each [nodes[j], nodes[j+1]] with value ``v[j]`` at node
    j, and equal to the tail value after the final node. Returns ``P, tail``
    with expected spectrum ``P @ v + tail_value * tail``. Cell integrals use
    the closed forms

        int_a^b e^{Q tau} d tau = Q^{-1} (e^{Qb} - e^{Qa}),
        int_a^b tau e^{Q tau} d tau
            = Q^{-1} (b e^{Qb} - a e^{Qa}) - Q^{-2} (e^{Qb} - e^{Qa}),

    so the continuous history enters exactly, not through a step approximation.
    """
    nodes_arr = np.asarray(list(nodes), dtype=float)
    if nodes_arr.ndim != 1 or len(nodes_arr) < 2:
        raise ValueError("nodes must be a one-dimensional array of length >= 2")
    if not np.all(np.diff(nodes_arr) > 0) or nodes_arr[0] != 0.0:
        raise ValueError("nodes must be strictly increasing and start at zero")

    q = coalescent_generator(n)
    bmat = branch_count_matrix(n)
    q_inv = np.linalg.inv(q)
    initial = np.zeros(n - 1, dtype=float)
    initial[-1] = 1.0

    node_count = len(nodes_arr)
    operator = np.zeros((n - 1, node_count), dtype=float)
    exp_states = [expm(q * t) @ initial for t in nodes_arr]
    for j, (left, right) in enumerate(zip(nodes_arr[:-1], nodes_arr[1:])):
        width = right - left
        e_left = exp_states[j]
        e_right = exp_states[j + 1]
        moment_zero = q_inv @ (e_right - e_left)
        moment_one = q_inv @ (right * e_right - left * e_left) - q_inv @ (
            q_inv @ (e_right - e_left)
        )
        # h(tau) = v_j (right - tau)/width + v_{j+1} (tau - left)/width
        left_column = bmat @ (right * moment_zero - moment_one) / width
        right_column = bmat @ (moment_one - left * moment_zero) / width
        operator[:, j] += left_column
        operator[:, j + 1] += right_column
    tail = bmat @ (-q_inv @ exp_states[-1])
    return operator, tail


def normalise_sfs(sfs: Iterable[float]) -> np.ndarray:
    """Return an SFS normalised to sum to one."""
    arr = np.asarray(list(sfs), dtype=float)
    total = float(np.sum(arr))
    if not np.isfinite(total) or total <= 0 or np.any(arr < 0):
        raise ValueError("SFS must be finite, non-negative and have positive sum")
    return arr / total
