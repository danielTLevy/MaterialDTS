"""Pareto front quality metrics.

Provides a ParetoMetrics class that computes multi-objective indicators
with respect to a ground-truth Pareto front, a ground-truth nadir, and
an ideal (utopian) reference point.

Implemented metrics:
- Hypervolume (HV)
- GDPlus (Generational Distance Plus)
- IGDPlus (Inverted Generational Distance Plus)
- R2 indicator (using uniform reference points)

Inputs are NumPy arrays and follow a minimization convention. If your
problem is a maximization, negate objective values before calling.
"""
from src.utils.pareto_utils import pareto_front
from typing import Optional
import numpy as np

from pymoo.indicators.hv import HV
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus


class ParetoMetrics:
    """Compute pareto-front metrics relative to a ground-truth front.

    Parameters
    ----------
    gt_pareto : np.ndarray
        Ground-truth Pareto front, shape (n_points, n_objectives). Use the
        same sign convention as pymoo (minimization).
    gt_nadir : Optional[np.ndarray]
        Ground-truth nadir point. If None, computed as gt_pareto.max(axis=0).
    ideal : Optional[np.ndarray]
        Utopian / ideal point. If None, computed as gt_pareto.min(axis=0).
    """

    def __init__(self, gt_rewards: np.ndarray, gt_nadir: Optional[np.ndarray] = None, ideal: Optional[np.ndarray] = None):
        gt_rewards_filtered = gt_rewards[~np.isnan(gt_rewards).any(axis=1)]
        gt_pareto_indices = pareto_front(gt_rewards_filtered, minimization=True)
        self.gt_pareto = gt_rewards_filtered[gt_pareto_indices]
        self.ideal = np.asarray(ideal, dtype=float) if ideal is not None else self.gt_pareto.min(axis=0)
        self.nadir = np.asarray(gt_nadir, dtype=float) if gt_nadir is not None else self.gt_pareto.max(axis=0)
        

        if self.ideal.shape != self.nadir.shape:
            raise ValueError("ideal and nadir must have the same shape (n_objectives,) or be omitted")


    # -------------------- normalization helpers --------------------
    def normalize(self, arr: np.ndarray) -> np.ndarray:
        """Zero-to-one normalize arr using stored ideal and nadir points.

        Clips to [0,1] to avoid division-by-zero or out-of-range values.
        """
        arr = np.asarray(arr, dtype=float)
        denom = self.nadir - self.ideal
        # avoid divide by zero
        denom = np.where(denom == 0.0, 1.0, denom)
        normed = (arr - self.ideal) / denom
        return np.clip(normed, 0.0, 1.0)

    # -------------------- pymoo-based indicators --------------------
    def hv(self, front: np.ndarray, ref_point: Optional[np.ndarray] = None, normalize: bool = True) -> float:
        """Compute Hypervolume of `front` with respect to `ref_point`.

        Parameters
        ----------
        front : np.ndarray
            Candidate Pareto front, shape (m, n_obj).
        ref_point : Optional[np.ndarray]
            Reference point for hypervolume. If None, defaults to stored nadir.
        normalize : bool
            If True, normalize front and ref_point to [0,1] using ideal/nadir.

        Returns
        -------
        float
            Hypervolume value.
        """
        front = np.asarray(front, dtype=float)
        if ref_point is None:
            ref_point = np.asarray(self.nadir, dtype=float)
        else:
            ref_point = np.asarray(ref_point, dtype=float)
        if normalize:
            ind = HV(ref_point=self.normalize(ref_point), pf=self.normalize(self.gt_pareto), norm_ref_point=False, zero_to_one=False)
            return float(ind(self.normalize(front)))
        else:
            ind = HV(ref_point=ref_point, pf=self.gt_pareto, norm_ref_point=False, zero_to_one=False)
            return float(ind(front))

    def gd_plus(self, front: np.ndarray, normalize: bool = True) -> float:
        """Compute GD+ (Generational Distance Plus) between candidate front and ground-truth PF.

        Returns a scalar where lower is better.
        """
        return self._compute_indicator(GDPlus, front, normalize)

    def igd_plus(self, front: np.ndarray, normalize: bool = True) -> float:
        """Compute IGD+ (Inverted Generational Distance Plus).

        Lower is better.
        """
        return self._compute_indicator(IGDPlus, front, normalize)

    def _compute_indicator(self, indicator_cls, front: np.ndarray, normalize: bool = True) -> float:
        """Generic helper to compute a pymoo indicator (GDPlus / IGDPlus).

        indicator_cls is expected to be callable/class accepting the pareto-front
        (pf) in its constructor and also callable to evaluate candidate fronts.
        """
        front = np.asarray(front, dtype=float)
        if normalize:
            pf = self.normalize(self.gt_pareto)
            arr = self.normalize(front)
        else:
            pf = self.gt_pareto
            arr = front
        ind = indicator_cls(pf)
        return float(ind(arr))

    # -------------------- R2 / reference-point based indicator --------------------
    @staticmethod
    def uniform_reference_points(nobj: int, p: int = 4, scaling: Optional[float] = None) -> np.ndarray:
        """Generate uniformly distributed reference points on the unit simplex.
        Taken from multi-Objective GFlowNets: https://github.com/GFNOrg/multi-objective-gfn
        """
        def gen_refs_recursive(ref, nobj, left, total, depth):
            points = []
            if depth == nobj - 1:
                ref[depth] = left / total
                points.append(ref)
            else:
                for i in range(left + 1):
                    ref[depth] = i / total
                    points.extend(gen_refs_recursive(ref.copy(), nobj, left - i, total, depth + 1))
            return points

        ref_points = np.array(gen_refs_recursive(np.zeros(nobj), nobj, p, p, 0))
        if scaling is not None:
            ref_points *= scaling
            ref_points += (1 - scaling) / nobj
        return ref_points

    @staticmethod
    def r2_indicator_set(reference_points: np.ndarray, solutions: np.ndarray, utopian_point: np.ndarray) -> float:
        """Compute the R2 indicator for a set of solutions and reference points.
        """
        min_list = []
        for v in reference_points:
            max_list = []
            for a in solutions:
                max_list.append(np.max(v * np.abs(utopian_point - a)))
            min_list.append(np.min(max_list))

        r2 = np.sum(min_list) / len(reference_points)
        return float(r2)

    def r2(self, front: np.ndarray, p: int = 4, normalize: bool = True, reference_points: Optional[np.ndarray] = None) -> float:
        """Compute R2 indicator of the candidate front.

        Parameters
        ----------
        front : np.ndarray
            Candidate front (m, n_obj).
        p : int
            Parameter controlling reference point granularity (see uniform_reference_points).
        normalize : bool
            If True, normalize front and gt_pareto by ideal/nadir before computing R2.
        reference_points : Optional[np.ndarray]
            Precomputed reference points. If None, generated with uniform_reference_points.
        """
        front = np.asarray(front, dtype=float)
        nobj = front.shape[1]
        if reference_points is None:
            ref_points = self.uniform_reference_points(nobj, p=p)
        else:
            ref_points = np.asarray(reference_points, dtype=float)

        if normalize:
            sol = -self.normalize(front)
            utopian = self.normalize(self.ideal)
            return float(self.r2_indicator_set(ref_points, sol, utopian))
        else:
            sol = -front
            utopian = self.ideal
            return float(self.r2_indicator_set(ref_points, sol, utopian))


__all__ = ["ParetoMetrics"]
