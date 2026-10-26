r"""
losses.py - THE NOVELTY: HALO, a Heat-Adaptive, Load-aware LINEX Objective
for LightGBM.

1. The problem with symmetric losses (research gap G1)
------------------------------------------------------
MSE treats "forecast 240 GW, actual 245 GW" (5 GW SHORT) exactly like
"forecast 245 GW, actual 240 GW" (5 GW SURPLUS). For the grid they are
very different:
   * 5 GW surplus  -> some thermal units were scheduled and are backed down;
                      DISCOMs pay capacity charges (~Rs 2.5/kWh).
   * 5 GW short    -> buy in the real-time market at the Rs 10/kWh cap; if
                      the market is dry -> load-shedding (rolling blackouts
                      in 45 C heat - a public-health problem, not just money).

2. Why not the usual fixes?
---------------------------
 a) Quantile / pinball loss (gap G4): asymmetric but LINEAR on both sides,
    so a 1 GW and a 10 GW shortfall differ only 10x in cost. Real shortfall
    cost is CONVEX: ~3-5 % of demand is covered by reserves (cheap), beyond
    that comes load-shedding (very expensive).
 b) Piecewise-weighted MSE with one fixed multiplier for "heatwave days"
    (the prototype notebook): a hard 0/1 regime switch with a constant
    asymmetry (gap G5) - a marginal 40.1 C day and a 47 C severe heatwave
    get the same treatment, and the constant-Hessian trick makes the
    Newton step ignore the curvature of the loss.

3. HALO: the loss
-----------------
Let e = y - y_hat (in % of the weekly baseline; e > 0 = UNDER-forecast).
LINEX (Varian 1975; Zellner 1986), scaled so that it equals e^2/2 for small e:

        L_a(e) = ( exp(a e) - a e - 1 ) / a^2              (a > 0)

  * small |e|      : L ~ e^2/2 -> behaves like ordinary MSE (accuracy kept)
  * large under-fc : L grows EXPONENTIALLY (convex shortfall cost)
  * large over-fc  : L grows only LINEARLY (surplus is cheap)
Numerically, with a = 0.6 per %:  under by 4 % -> L = 21.2 ;  over by 4 % ->
L = 4.1  (5.1x).  Under by 1 % -> 0.62 ; over by 1 % -> 0.41 (1.5x).
So the asymmetry automatically grows with the size of the miss - just like
the real cost of a shortfall once reserves are exhausted.

The asymmetry a is NOT a constant. It adapts per sample (the "conditioning"):

        a_i = a0  +  a_heat * s_i  +  a_scarcity * g_i

  s_i in [0,1]  heat severity of day i: demand-weighted IMD/EHF heat
                severity index, normalised by its training 99th percentile.
                -> a 47 C severe heatwave over UP+Rajasthan gets s ~ 1,
                   a mild 40 C day s ~ 0.1, a winter day s = 0.
  g_i in [0,1]  grid scarcity: how close the TRUE peak is to the record of
                the previous 365 days,  g = clip((peak/record - 0.95)/0.05, 0, 1).
                Near the all-time record there is no spare capacity left, so
                a shortfall means load-shedding instead of an exchange
                purchase. (The loss may use labels - it is only evaluated
                during training; at prediction time no label is needed.)
  a0            small base asymmetry so ordinary days stay ~MSE.

4. Why LINEX has the "right" optimal forecast
--------------------------------------------
The forecast that minimises expected LINEX is
        y* = (1/a) * log E[exp(a Y)]  ~  mu + (a/2) * sigma^2      (Gaussian Y)
i.e. the mean PLUS a safety margin proportional to the UNCERTAINTY.
Heatwave days are both higher-cost (bigger a) AND more uncertain (bigger
sigma) -> the margin grows on exactly those days, while calm winter days
(low sigma, small a) get almost none. This is precisely how a sensible
operator sets reserves: more head-room when the day is both risky and hard
to predict. A quantile loss or a constant multiplier cannot do this.

5. Making it work inside gradient boosting
-----------------------------------------
LightGBM needs gradient g and Hessian h per sample (w.r.t. y_hat):
        u = a e
        g = -(exp(u) - 1) / a
        h =  exp(u)
  * Huberised tail: for u > u_max the exponential is continued linearly
    (exp(u_max) * (1 + u - u_max)), so a single glitchy target (e.g. a
    data spike or an unforeseen grid event) cannot produce an exploding
    gradient that dominates a whole tree.
  * Hessian floor: on the over-forecast side exp(u) -> 0; a near-zero
    Hessian would make the leaf Newton step (-sum g / sum h) blow up, so h
    is floored at h_min.
  * MW consistency: g and h are multiplied by base_i / mean(base) so that a
    1 % miss in 2025 (240 GW base) counts more than 1 % in 2019 (160 GW):
    the loss is then proportional to megawatts, i.e. to real cost.
Unlike the prototype's constant-Hessian trick, the true curvature is kept:
the leaf values are exact Newton steps on the actual loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HALOObjective:
    """
    Heat-Adaptive, Load-aware LINEX Objective for lightgbm.train().

    Parameters
    ----------
    a0 : base asymmetry (per % of error). 0.1 -> near-MSE on ordinary days.
    a_heat : extra asymmetry at full heat severity (s = 1).
    a_scarcity : extra asymmetry when the true peak reaches the annual record.
    u_max : Huber cap on a*e (exp(4) = 55x the MSE gradient at most).
    h_min : Hessian floor for the over-forecast side.
    """
    a0: float = 0.10
    a_heat: float = 0.60
    a_scarcity: float = 0.30
    u_max: float = 4.0
    h_min: float = 0.05
    # per-dataset side information, registered with .register()
    _side: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- setup
    def register(self, dataset, severity, scarcity, scale, weight=None):
        """Attach per-row side information to a lightgbm.Dataset.

        severity, scarcity : arrays in [0, 1] (s_i and g_i above)
        scale              : base_i / mean(base) (MW consistency)
        weight             : optional sample weights (0 = ignore row)
        """
        n = len(severity)
        w = np.ones(n) if weight is None else np.asarray(weight, float)
        self._side[id(dataset)] = dict(
            a=self.asymmetry(np.asarray(severity, float), np.asarray(scarcity, float)),
            scale=np.asarray(scale, float) * w,
        )
        return dataset

    def asymmetry(self, severity: np.ndarray, scarcity: np.ndarray) -> np.ndarray:
        return self.a0 + self.a_heat * np.clip(severity, 0, 1) + self.a_scarcity * np.clip(scarcity, 0, 1)

    # ------------------------------------------------------------ the maths
    def _exp_huber(self, u):
        """exp(u) for u <= u_max, first-order continuation above it."""
        cap = np.exp(self.u_max)
        return np.where(u <= self.u_max, np.exp(np.minimum(u, self.u_max)),
                        cap * (1.0 + u - self.u_max))

    def loss(self, y, y_hat, a, scale):
        e = y - y_hat
        u = a * e
        # integral of the Huberised gradient (continuous and C1 at u_max)
        cap = np.exp(self.u_max)
        core = np.exp(np.minimum(u, self.u_max)) - np.minimum(u, self.u_max) - 1.0
        extra = np.where(u > self.u_max,
                         (cap - 1.0) * (u - self.u_max) + 0.5 * cap * (u - self.u_max) ** 2, 0.0)
        return scale * (core + extra) / a ** 2

    def grad_hess(self, y, y_hat, a, scale):
        u = a * (y - y_hat)
        ex = self._exp_huber(u)
        grad = -(ex - 1.0) / a
        hess = np.where(u <= self.u_max, np.exp(np.minimum(u, self.u_max)), np.exp(self.u_max))
        hess = np.maximum(hess, self.h_min)
        return grad * scale, hess * scale

    # ---------------------------------------------------- lightgbm adapters
    def objective(self, preds, train_data):
        side = self._side[id(train_data)]
        return self.grad_hess(train_data.get_label(), preds, side["a"], side["scale"])

    def feval(self, preds, eval_data):
        """Mean HALO loss - used for early stopping on the validation set."""
        side = self._side[id(eval_data)]
        y = eval_data.get_label()
        lo = self.loss(y, preds, side["a"], side["scale"])
        return "halo", float(lo.sum() / max(side["scale"].sum(), 1e-9)), False


@dataclass
class PrototypeAsymmetricL2:
    """
    The objective from the team's earlier synthetic-data prototype, kept as
    an ABLATION baseline: residual gradient multiplied by alpha_heatwave on
    under-forecast heatwave days and by alpha_ordinary on other
    under-forecasts; Hessian fixed at 1. (Binary regime, constant
    asymmetry, curvature discarded - the three limitations HALO removes.)
    """
    alpha_heatwave: float = 21.0
    alpha_ordinary: float = 4.0
    _side: dict = field(default_factory=dict, repr=False)

    def register(self, dataset, heatwave_flag, weight=None):
        n = len(heatwave_flag)
        w = np.ones(n) if weight is None else np.asarray(weight, float)
        self._side[id(dataset)] = dict(hw=np.asarray(heatwave_flag) == 1, w=w)
        return dataset

    def objective(self, preds, train_data):
        side = self._side[id(train_data)]
        residual = preds - train_data.get_label()
        grad = residual.copy()
        under = residual < 0
        grad[side["hw"] & under] *= self.alpha_heatwave
        grad[~side["hw"] & under] *= self.alpha_ordinary
        hess = np.ones_like(residual)
        return grad * side["w"], hess * side["w"]


def numerical_gradient_check(obj: HALOObjective, n: int = 200, seed: int = 0) -> float:
    """Max |analytic - finite-difference| gradient error (should be ~1e-5)."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 4, n)
    yh = y + rng.normal(0, 5, n)
    a = obj.asymmetry(rng.uniform(0, 1, n), rng.uniform(0, 1, n))
    scale = rng.uniform(0.5, 1.5, n)
    g, _ = obj.grad_hess(y, yh, a, scale)
    eps = 1e-5
    num = (obj.loss(y, yh + eps, a, scale) - obj.loss(y, yh - eps, a, scale)) / (2 * eps)
    return float(np.max(np.abs(num - g)))
