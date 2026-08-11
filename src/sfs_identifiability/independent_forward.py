"""Independent coalescence-to-SFS bridge sharing no code with operators.py.

The primary implementation (operators.py) builds the transient coalescent
generator and integrates it with a matrix exponential. This module reaches the
same expected branch-length SFS through a mathematically different route:

1. Tavare's (1984) closed-form alternating exponential sum for the ancestral
   lineage-count process, with every coefficient computed in exact rational
   arithmetic;
2. the classical block-size fraction k*C(n-b-1,k-2)/C(n-1,k-1), verified here
   against an exact first-principles enumeration of Kingman merge sequences
   (dynamic programming over integer partitions with Fraction probabilities);
3. exact rational assembly of the (c_2,...,c_n)-to-SFS weight matrix W, with a
   fraction-free determinant supplying an exact invertibility certificate.

No generator matrix, no matrix exponential and no floating-point cancellation
enter the coefficient construction. Floats appear only in the final
mpmath-evaluated matrix-vector product. Two internal exact identities
(partition of unity for the Tavare coefficients, and the classical
constant-history spectrum 2/b recovered as an exact rational identity) are
asserted at construction time, so a wrong formula cannot load silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import comb
from typing import Dict, Iterable, Tuple

import mpmath as mp
import numpy as np


def _rising(a: int, terms: int) -> int:
    out = 1
    for i in range(terms):
        out *= a + i
    return out


def _falling(a: int, terms: int) -> int:
    out = 1
    for i in range(terms):
        out *= a - i
    return out


def tavare_coefficients(n: int) -> Dict[int, Dict[int, Fraction]]:
    """Exact coefficients rho[k][m] with P(N(tau)=k) = sum_m rho[k][m] e^{-C(m,2) tau}.

    Tavare (1984), Theorem 6.1: for 1 <= k <= n and k <= m <= n,

        rho[k][m] = (2m-1) (-1)^{m-k} k_{(m-1)} n_{[m]} / ( k! (m-k)! n_{(m)} ),

    with rising factorial a_{(j)} = a(a+1)...(a+j-1) and falling factorial
    a_{[j]} = a(a-1)...(a-j+1). Two exact identities are asserted before the
    coefficients are returned:

    * initial condition: sum_m rho[k][m] = 1 if k == n else 0;
    * partition of unity: sum_k rho[k][m] = 1 if m == 1 else 0.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    rho: Dict[int, Dict[int, Fraction]] = {}
    for k in range(1, n + 1):
        rho[k] = {}
        for m in range(k, n + 1):
            numerator = (2 * m - 1) * _rising(k, m - 1) * _falling(n, m)
            denominator = 1
            for i in range(1, k + 1):
                denominator *= i
            for i in range(1, m - k + 1):
                denominator *= i
            denominator *= _rising(n, m)
            value = Fraction(numerator, denominator)
            if (m - k) % 2 == 1:
                value = -value
            rho[k][m] = value

    for k in range(1, n + 1):
        total = sum(rho[k].values())
        expected = Fraction(1) if k == n else Fraction(0)
        if total != expected:
            raise AssertionError(
                f"Tavare initial-condition identity failed at k={k}: {total}"
            )
    for m in range(1, n + 1):
        total = sum(rho[k][m] for k in range(1, n + 1) if m in rho[k])
        expected = Fraction(1) if m == 1 else Fraction(0)
        if total != expected:
            raise AssertionError(
                f"Tavare partition-of-unity identity failed at m={m}: {total}"
            )
    return rho


def block_count_fraction(n: int, b: int, k: int) -> Fraction:
    """Expected number of ancestral blocks of size b when k blocks remain.

    The classical formula k*C(n-b-1,k-2)/C(n-1,k-1) (Fu 1995). It is verified
    against exact merge-sequence enumeration by
    ``verify_block_counts_by_enumeration``.
    """
    if not (2 <= k <= n and 1 <= b <= n - 1):
        return Fraction(0)
    if n - b - 1 < k - 2:
        return Fraction(0)
    return Fraction(k * comb(n - b - 1, k - 2), comb(n - 1, k - 1))


def verify_block_counts_by_enumeration(n: int) -> None:
    """Exact DP over Kingman merge sequences confirming block_count_fraction.

    States are integer partitions of n (multisets of block sizes). From a state
    with k blocks, every unordered pair of blocks merges with probability
    1/C(k,2). The expected number of blocks of each size at every level k is
    accumulated exactly and compared with the closed formula. Raises
    AssertionError on any discrepancy. Practical for n <= 9.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    state: Dict[Tuple[int, ...], Fraction] = {tuple([1] * n): Fraction(1)}
    for k in range(n, 1, -1):
        expected_counts: Dict[int, Fraction] = {}
        for partition, probability in state.items():
            for size in partition:
                expected_counts[size] = (
                    expected_counts.get(size, Fraction(0)) + probability
                )
        for b in range(1, n):
            formula = block_count_fraction(n, b, k)
            enumerated = expected_counts.get(b, Fraction(0))
            if formula != enumerated:
                raise AssertionError(
                    f"block-count mismatch at n={n}, k={k}, b={b}: "
                    f"formula {formula}, enumeration {enumerated}"
                )
        next_state: Dict[Tuple[int, ...], Fraction] = {}
        pair_count = Fraction(1, comb(k, 2))
        for partition, probability in state.items():
            blocks = list(partition)
            for i, j in combinations(range(k), 2):
                merged = sorted(
                    [size for idx, size in enumerate(blocks) if idx not in (i, j)]
                    + [blocks[i] + blocks[j]]
                )
                key = tuple(merged)
                next_state[key] = (
                    next_state.get(key, Fraction(0)) + probability * pair_count
                )
        state = next_state


def sfs_weight_matrix_exact(n: int) -> list[list[Fraction]]:
    """Exact W with E[xi_b] = sum_m W[b][m] c_m, c_m = int h e^{-C(m,2) tau} d tau.

    W[b][m] = sum_{k=2}^{m} k*C(n-b-1,k-2)/C(n-1,k-1) * rho[k][m], assembled in
    exact rational arithmetic. Constructing W asserts the classical identity

        sum_m W[b][m] / C(m,2) = 2/b   (exactly, for every b),

    which is the constant-history spectrum and involves no floating point.
    """
    rho = tavare_coefficients(n)
    weights: list[list[Fraction]] = []
    for b in range(1, n):
        row: list[Fraction] = []
        for m in range(2, n + 1):
            total = Fraction(0)
            for k in range(2, m + 1):
                a_bk = block_count_fraction(n, b, k)
                if a_bk:
                    total += a_bk * rho[k][m]
            row.append(total)
        weights.append(row)

    for b in range(1, n):
        acc = Fraction(0)
        for m in range(2, n + 1):
            acc += weights[b - 1][m - 2] / Fraction(m * (m - 1), 2)
        if acc != Fraction(2, b):
            raise AssertionError(
                f"constant-history identity failed at b={b}: {acc} != 2/{b}"
            )
    return weights


def exact_invertibility_certificate(n: int) -> Tuple[bool, str, float]:
    """Exact determinant of W by fraction-free elimination, plus float condition.

    Returns (invertible, determinant_string, float_condition_estimate). The
    determinant is computed in exact rational arithmetic, so a non-zero result
    is a certificate that the (c_2,...,c_n)-to-SFS transform has full rank.
    """
    weights = sfs_weight_matrix_exact(n)
    size = n - 1
    matrix = [[weights[i][j] for j in range(size)] for i in range(size)]
    determinant = Fraction(1)
    sign = 1
    for col in range(size):
        pivot_row = next(
            (r for r in range(col, size) if matrix[r][col] != 0), None
        )
        if pivot_row is None:
            return False, "0", float("inf")
        if pivot_row != col:
            matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
            sign = -sign
        pivot = matrix[col][col]
        determinant *= pivot
        for r in range(col + 1, size):
            factor = matrix[r][col] / pivot
            for c in range(col, size):
                matrix[r][c] -= factor * matrix[col][c]
    determinant *= sign
    float_matrix = np.array(
        [[float(value) for value in row] for row in weights], dtype=float
    )
    condition = float(np.linalg.cond(float_matrix))
    return determinant != 0, str(determinant), condition


@dataclass(frozen=True)
class IndependentForward:
    """Precomputed exact weight matrix with mpmath evaluation at fixed precision."""

    sample_size: int
    precision: int
    weights_exact: tuple

    @classmethod
    def build(cls, n: int, *, precision: int = 60) -> "IndependentForward":
        weights = sfs_weight_matrix_exact(n)
        return cls(
            sample_size=n,
            precision=precision,
            weights_exact=tuple(tuple(row) for row in weights),
        )

    def _rates(self) -> list:
        return [
            mp.mpf(m * (m - 1)) / 2 for m in range(2, self.sample_size + 1)
        ]

    def expected_sfs_piecewise_constant(
        self,
        edges: Iterable[float],
        history: Iterable[float],
        *,
        tail_value: float = 1.0,
    ) -> np.ndarray:
        """Expected branch SFS for piecewise-constant h, all sums in mpmath."""
        edges_list = [mp.mpf(str(float(e))) for e in edges]
        history_list = [mp.mpf(str(float(h))) for h in history]
        if len(history_list) != len(edges_list) - 1:
            raise ValueError("history must have one value per grid cell")
        if edges_list[0] != 0:
            raise ValueError("edges must start at zero")
        with mp.workdps(self.precision):
            rates = self._rates()
            moments = []
            for rate in rates:
                total = mp.mpf(0)
                for left, right, value in zip(
                    edges_list[:-1], edges_list[1:], history_list
                ):
                    total += value * (mp.e**(-rate * left) - mp.e**(-rate * right)) / rate
                total += mp.mpf(str(float(tail_value))) * mp.e**(-rate * edges_list[-1]) / rate
                moments.append(total)
            out = []
            for row in self.weights_exact:
                acc = mp.mpf(0)
                for weight, moment in zip(row, moments):
                    acc += mp.mpf(weight.numerator) / mp.mpf(weight.denominator) * moment
                out.append(float(acc))
        return np.asarray(out, dtype=float)

    def expected_sfs_piecewise_linear(
        self,
        nodes: Iterable[float],
        values: Iterable[float],
        *,
        tail_value: float = 1.0,
    ) -> np.ndarray:
        """Expected branch SFS for continuous piecewise-linear h on the nodes.

        h is linear on each [nodes[j], nodes[j+1]] with h(nodes[j]) = values[j],
        and equal to tail_value after the final node. The segment integrals
        int (alpha + beta tau) e^{-r tau} d tau use their closed form.
        """
        node_list = [mp.mpf(str(float(x))) for x in nodes]
        value_list = [mp.mpf(str(float(v))) for v in values]
        if len(node_list) != len(value_list):
            raise ValueError("values must have one entry per node")
        if node_list[0] != 0:
            raise ValueError("nodes must start at zero")
        with mp.workdps(self.precision):
            rates = self._rates()
            moments = []
            for rate in rates:
                total = mp.mpf(0)
                for left, right, v_left, v_right in zip(
                    node_list[:-1], node_list[1:], value_list[:-1], value_list[1:]
                ):
                    width = right - left
                    beta = (v_right - v_left) / width
                    alpha = v_left - beta * left
                    e_left = mp.e**(-rate * left)
                    e_right = mp.e**(-rate * right)
                    total += (
                        (alpha + beta * left) * e_left / rate
                        - (alpha + beta * right) * e_right / rate
                        + beta * (e_left - e_right) / rate**2
                    )
                total += mp.mpf(str(float(tail_value))) * mp.e**(-rate * node_list[-1]) / rate
                moments.append(total)
            out = []
            for row in self.weights_exact:
                acc = mp.mpf(0)
                for weight, moment in zip(row, moments):
                    acc += mp.mpf(weight.numerator) / mp.mpf(weight.denominator) * moment
                out.append(float(acc))
        return np.asarray(out, dtype=float)
