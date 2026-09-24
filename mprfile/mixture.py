"""
mprfile.mixture — Gaussian-mixture fits in user-defined regions (ROIs), with model
comparison and overfitting diagnostics.

Each ROI [lo, hi] is fitted by maximum likelihood on the individual event masses
(no binning) with a mixture of k Gaussians, truncated to the ROI. For the same ROI,
every k from 1 to k_max is fitted too, so the chosen k can be compared with the
alternatives by BIC and AIC.

    from mprfile.mixture import fit_roi
    res = fit_roi(masses, 400, 620, k=2, resolution=15)
    res.components      # mass, ± error, sigma, weight, events per component
    res.models          # k, log-likelihood, AIC, BIC, ΔBIC, evidence
    res.warnings        # overfitting / reliability warnings

Interpretation of ΔBIC (BIC(k) − BIC(best)), after Kass & Raftery (1995):
0–2 not worth more than a mention, 2–6 positive, 6–10 strong, >10 very strong
evidence against the model with the higher BIC.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

__all__ = ["MixtureFit", "ROIResult", "fit_mixture", "compare_models", "fit_roi", "fit_rois",
           "bootstrap_roi", "evidence_label", "ashman_d"]


def evidence_label(dbic: float) -> str:
    d = abs(dbic)
    if d < 2:
        return "weak"
    if d < 6:
        return "positive"
    if d < 10:
        return "strong"
    return "very strong"


def ashman_d(mu1, s1, mu2, s2) -> float:
    """Ashman's D: separation of two Gaussians in units of their widths.
    D > 2 is needed for two components to be cleanly separable."""
    return float(np.sqrt(2) * abs(mu1 - mu2) / np.sqrt(s1 ** 2 + s2 ** 2))


@dataclass
class MixtureFit:
    lo: float
    hi: float
    k: int
    n: int                       # events inside the ROI
    weights: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    loglik: float
    converged: bool
    mu_err: np.ndarray | None = None
    sigma_err: np.ndarray | None = None
    weight_err: np.ndarray | None = None
    sigma_at_bound: np.ndarray | None = None
    background_weight: float = 0.0     # fitted share of the flat background (0 = none)
    sigma_at_max: np.ndarray | None = None
    has_background: bool = False

    def __post_init__(self):
        self.has_background = self.has_background or self.background_weight > 0

    @property
    def n_params(self) -> int:
        """Number of free parameters: 3k − 1 (+1 with a flat background)."""
        return 3 * self.k - 1 + int(self.has_background)

    @property
    def aic(self) -> float:
        """Akaike information criterion, 2p − 2 ln L."""
        return 2 * self.n_params - 2 * self.loglik

    @property
    def bic(self) -> float:
        """Bayesian information criterion, p ln n − 2 ln L (lower is better)."""
        return self.n_params * np.log(max(self.n, 1)) - 2 * self.loglik

    @property
    def counts(self) -> np.ndarray:
        """Expected number of events per component (inside the ROI)."""
        return self.weights_in_roi * self.n

    def _inside(self):
        return self.weights * (norm.cdf(self.hi, self.mu, self.sigma) - norm.cdf(self.lo, self.mu, self.sigma))

    @property
    def weights_in_roi(self) -> np.ndarray:
        """Share of the events *inside the ROI* belonging to each Gaussian (the fitted
        weights refer to the untruncated Gaussians; the background takes the rest)."""
        g = self._inside()
        return g / (g.sum() + self.background_weight)

    @property
    def background_share(self) -> float:
        """Share of the ROI's events assigned to the flat background."""
        if not self.has_background:
            return 0.0
        return float(self.background_weight / (self._inside().sum() + self.background_weight))

    def density(self, x, bin_width: float = 1.0, per_component: bool = False):
        """Fitted curve in *events per bin* for a histogram with this bin width."""
        x = np.asarray(x, float)
        z = self._inside().sum() + self.background_weight
        comps = [self.n * bin_width * w * norm.pdf(x, m, s) / z for w, m, s in zip(self.weights, self.mu, self.sigma)]
        bg = np.full_like(x, self.n * bin_width * self.background_weight / (self.hi - self.lo) / z)
        inside = (x >= self.lo) & (x <= self.hi)
        comps = [np.where(inside, c, np.nan) for c in comps]
        bg = np.where(inside, bg, np.nan)
        if per_component:
            return comps, (bg if self.has_background else None)
        return np.sum(comps, axis=0) + (bg if self.has_background else 0)

    def table(self) -> pd.DataFrame:
        """Fitted components: mass, ± error, σ, share of the ROI's events, events."""
        df = pd.DataFrame({"mass_kDa": self.mu, "sigma_kDa": self.sigma, "weight": self.weights_in_roi,
                           "counts": self.counts})
        if self.mu_err is not None:
            df.insert(1, "mass_err_kDa", self.mu_err)
        return df.sort_values("mass_kDa", ignore_index=True)


# ---------------------------------------------------------------------------- fitting
# Parameters θ: [logits of components 1..K-1 (component 0 fixed at 0), μ_1..μ_k, log σ_1..log σ_k]
# where K = k Gaussians (+1 flat background if used; the background is the last weight).

def _unpack(theta, k, bg=False):
    K = k + int(bg)
    logits = np.concatenate([[0.0], theta[:K - 1]])
    w = np.exp(logits - logsumexp(logits))
    mu = theta[K - 1:K - 1 + k]
    sigma = np.exp(theta[K - 1 + k:])
    return w, mu, sigma


def _nll_grad(theta, x, lo, hi, k, bg=False):
    """Negative log-likelihood of the truncated mixture and its exact gradient."""
    K = k + int(bg)
    w, mu, sigma = _unpack(theta, k, bg)
    n = len(x)
    z = (x[:, None] - mu[None, :]) / sigma[None, :]
    log_comp = np.log(w[:k])[None, :] - 0.5 * z ** 2 - np.log(sigma)[None, :] - 0.5 * np.log(2 * np.pi)
    if bg:
        log_comp = np.concatenate([log_comp, np.full((n, 1), np.log(w[k]) - np.log(hi - lo))], axis=1)
    lse = logsumexp(log_comp, axis=1)
    r = np.exp(log_comp - lse[:, None])                     # responsibilities (n × K)
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    P = norm.cdf(b) - norm.cdf(a)
    Pall = np.concatenate([P, [1.0]]) if bg else P
    Z = float(np.sum(w * Pall))
    nll = -(lse.sum() - n * np.log(max(Z, 1e-300)))

    rg = r[:, :k]
    g_mu = -np.sum(rg * z, axis=0) / sigma + n / Z * w[:k] * (norm.pdf(a) - norm.pdf(b)) / sigma
    g_ls = -np.sum(rg * (z ** 2 - 1), axis=0) + n / Z * w[:k] * (a * norm.pdf(a) - b * norm.pdf(b))
    R = r.sum(axis=0)
    g_logit = -(R - n * w) + n / Z * w * (Pall - Z)
    grad = np.concatenate([g_logit[1:], g_mu, g_ls])
    return nll, grad


def _starts(x, lo, hi, k, n_starts, rng, sigma_min):
    """Starting points: evenly spaced quantiles, histogram maxima, and random subsets."""
    starts = []
    q = np.quantile(x, (np.arange(k) + 0.5) / k)
    s0 = max(np.std(x) / (2 * k), sigma_min * 1.5)
    starts.append((q, np.full(k, s0)))
    h, e = np.histogram(x, bins=max(10, min(80, len(x) // 15)))
    c = 0.5 * (e[1:] + e[:-1])
    picks = []
    for t in c[np.argsort(h)[::-1]]:
        if all(abs(t - p) > (hi - lo) / (4 * k) for p in picks):
            picks.append(t)
        if len(picks) == k:
            break
    if len(picks) == k:
        starts.append((np.sort(picks), np.full(k, s0)))
    while len(starts) < n_starts:
        starts.append((np.sort(rng.choice(x, k, replace=False)), np.full(k, s0 * rng.uniform(0.5, 2.0))))
    return starts


def fit_mixture(values, lo: float, hi: float, k: int, *, background: bool = False,
                sigma_min: float = 2.0, n_starts: int = 6, seed: int = 0,
                errors: bool = True) -> MixtureFit:
    """Maximum-likelihood fit of k truncated Gaussians (+ optional flat background) to the
    values inside [lo, hi]. sigma_min (kDa) prevents the degenerate zero-width solutions
    every mixture likelihood has."""
    x = np.asarray(values, float)
    x = x[(x >= lo) & (x <= hi) & np.isfinite(x)]
    n = len(x)
    K = k + int(background)
    if n < 3 * K + 5:
        raise ValueError(f"only {n} events in {lo:g}–{hi:g} kDa: too few for {k} Gaussian(s)")
    rng = np.random.default_rng(seed)
    # with a flat background, a Gaussian wider than half the ROI is indistinguishable from it
    sigma_max = (hi - lo) / 2 if background else (hi - lo)
    bounds = [(-15, 15)] * (K - 1) + [(lo, hi)] * k + [(np.log(sigma_min), np.log(sigma_max))] * k
    starts = _starts(x, lo, hi, k, n_starts, rng, sigma_min)
    if background:
        # also start from the best fit without background (avoids the flat-Gaussian ridge)
        try:
            nb = fit_mixture(x, lo, hi, k, background=False, sigma_min=sigma_min, n_starts=max(3, n_starts // 2),
                             seed=seed, errors=False)
            starts.insert(0, (nb.mu, np.minimum(nb.sigma, sigma_max * 0.8)))
        except ValueError:
            pass
    best = None
    for mu0, s0 in starts:
        w0 = np.full(K, 1.0 / K)
        if background:
            w0 = np.concatenate([np.full(k, 0.85 / k), [0.15]])
        th0 = np.concatenate([np.log(w0[1:] / w0[0]), np.clip(mu0, lo, hi),
                              np.log(np.clip(s0, sigma_min * 1.01, sigma_max * 0.99))])
        r = minimize(_nll_grad, th0, args=(x, lo, hi, k, background), jac=True, method="L-BFGS-B",
                     bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    w, mu, sigma = _unpack(best.x, k, background)
    order = np.argsort(mu)
    fit = MixtureFit(lo, hi, k, n, w[:k][order], mu[order], sigma[order], -float(best.fun), bool(best.success),
                     background_weight=float(w[k]) if background else 0.0)
    fit.sigma_at_bound = (sigma[order] <= sigma_min * 1.02)
    fit.sigma_at_max = (sigma[order] >= sigma_max * 0.98)
    if errors:
        try:
            g = lambda t: _nll_grad(t, x, lo, hi, k, background)[1]
            th = best.x
            step = 1e-4 * np.maximum(np.abs(th), 1.0)
            H = np.array([(g(th + np.eye(len(th))[i] * step[i]) - g(th - np.eye(len(th))[i] * step[i]))
                          / (2 * step[i]) for i in range(len(th))])
            cov = np.linalg.pinv(0.5 * (H + H.T))
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
            fit.mu_err = se[K - 1:K - 1 + k][order]
            fit.sigma_err = (se[K - 1 + k:] * sigma)[order]
            # at a parameter bound the curvature-based error is meaningless
            bad = fit.sigma_at_bound | fit.sigma_at_max | ~(fit.mu_err > 0)
            fit.mu_err = np.where(bad, np.nan, fit.mu_err)
            fit.sigma_err = np.where(bad, np.nan, fit.sigma_err)
        except (np.linalg.LinAlgError, ValueError):
            pass
    return fit


def compare_models(values, lo, hi, k_max: int = 5, *, background: bool = False,
                   sigma_min: float = 2.0, seed: int = 0):
    """Fit k = 1..k_max in the same ROI. Returns (table, fits)."""
    fits = {}
    for k in range(1, k_max + 1):
        try:
            fits[k] = fit_mixture(values, lo, hi, k, background=background, sigma_min=sigma_min,
                                  seed=seed, errors=False)
        except ValueError:
            break
    rows = [{"k": k, "log_likelihood": f.loglik, "n_params": f.n_params, "AIC": f.aic, "BIC": f.bic}
            for k, f in fits.items()]
    t = pd.DataFrame(rows)
    if len(t):
        t["dBIC"] = t.BIC - t.BIC.min()
        t["dAIC"] = t.AIC - t.AIC.min()
        t["evidence_against"] = [("—" if d == 0 else evidence_label(d)) for d in t.dBIC]
    return t, fits


# ---------------------------------------------------------------------------- ROI result
@dataclass
class ROIResult:
    lo: float
    hi: float
    k: int
    fit: MixtureFit
    models: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    bootstrap: pd.DataFrame | None = None
    bic_best_k: int | None = None
    background: bool = True

    @property
    def label(self) -> str:
        """'lo–hi kDa'."""
        return f"{self.lo:.0f}–{self.hi:.0f} kDa"

    @property
    def risk(self) -> str:
        """'low', 'moderate' or 'high' risk that the chosen k over- (or under-)fits."""
        text = " ".join(self.warnings)
        if "OVERFITTING" in text or "UNDERFITTING" in text:
            return "high"
        return "moderate" if self.warnings else "low"

    def components(self, n_total: int | None = None) -> pd.DataFrame:
        """Component table with the ROI label, and percentages relative to n_total events if given."""
        df = self.fit.table()
        df.insert(0, "roi", self.label)
        if n_total:
            df["percent"] = 100 * df.counts / n_total
        if self.bootstrap is not None:
            df["mass_sd_bootstrap"] = self.bootstrap["mass_sd"].values
            df["weight_sd_bootstrap"] = self.bootstrap["weight_sd"].values
        return df

    def spec(self) -> tuple:
        """(lo, hi, k): the ROI definition to reproduce this fit via AnalysisSettings(rois=[...])."""
        return (round(self.lo, 1), round(self.hi, 1), self.k)

    def model_summary(self) -> str:
        """One-line verdict: chosen k, BIC-preferred k and the evidence (ΔBIC)."""
        t = self.models.set_index("k")
        parts = [f"ROI {self.label}: {self.k} Gaussian(s)" + (" + flat background" if self.background else "")]
        if self.bic_best_k == self.k:
            others = t.drop(index=self.k)
            nxt = others.dBIC.min() if len(others) else float("nan")
            parts.append(f"BIC-preferred (next best ΔBIC +{nxt:.1f}, {evidence_label(nxt)})")
        else:
            parts.append(f"BIC prefers {self.bic_best_k} (ΔBIC of your choice +{t.loc[self.k, 'dBIC']:.1f}, "
                         f"{evidence_label(t.loc[self.k, 'dBIC'])})")
        return "; ".join(parts)


def fit_roi(values, lo: float, hi: float, k: int, *, resolution: float | None = None,
            k_max: int = 4, background: bool = True, min_events: int = 20, min_weight: float = 0.05,
            sigma_min: float | None = None, seed: int = 0, models: pd.DataFrame | None = None) -> ROIResult:
    """Fit k Gaussians in [lo, hi], compare with k = 1..k_max and diagnose overfitting.

    resolution: typical width (σ, kDa) of a single species on this instrument, e.g. the
                calibrant peak width (Calibration.resolution_kDa). Enables the width checks.
    background: include a flat background in the ROI. Recommended: without it, the
                continuum of events between peaks tends to be 'explained' by extra broad
                Gaussians, which BIC then rewards.
    """
    if hi <= lo:
        raise ValueError("ROI upper limit must be above the lower limit")
    smin = sigma_min if sigma_min else (max(2.0, 0.3 * resolution) if resolution else 2.0)
    k_max = max(k_max, k + 1)
    if models is None or k not in set(models.k):
        models, fits = compare_models(values, lo, hi, k_max, background=background, sigma_min=smin, seed=seed)
    else:
        fits = {int(kk): None for kk in models.k}         # reuse the model comparison
    if k not in fits:
        raise ValueError(f"too few events in {lo:g}–{hi:g} kDa for {k} Gaussian(s)")
    fit = fit_mixture(values, lo, hi, k, background=background, sigma_min=smin, seed=seed)
    res = ROIResult(lo, hi, k, fit, models, background=background)
    res.bic_best_k = int(models.loc[models.BIC.idxmin(), "k"])
    res.warnings = diagnose(res, resolution=resolution, min_events=min_events, min_weight=min_weight)
    return res


def diagnose(res: ROIResult, *, resolution=None, min_events=20, min_weight=0.05) -> list[str]:
    """Overfitting / reliability checks for the chosen k in one ROI."""
    w: list[str] = []
    f, t, k = res.fit, res.models, res.k
    bic = dict(zip(t.k, t.BIC))
    best = res.bic_best_k
    if best is not None and k in bic:
        d = bic[k] - bic[best]
        if best < k and d > 2:
            w.append(f"OVERFITTING risk: BIC prefers {best} Gaussian(s) over {k} "
                     f"(ΔBIC = +{d:.1f}, {evidence_label(d)} evidence against {k})")
        elif best > k and d > 6:
            w.append(f"UNDERFITTING: BIC prefers {best} Gaussians over {k} "
                     f"(ΔBIC = +{d:.1f}, {evidence_label(d)} evidence against {k})")
        elif best != k and d <= 2:
            w.append(f"{k} and {best} Gaussian(s) are statistically indistinguishable (ΔBIC = {d:.1f})")
    if k > 1 and (k - 1) in bic and bic[k] - bic[k - 1] > -2 and not (best is not None and best < k):
        w.append(f"Adding Gaussian #{k} barely improves the fit (ΔBIC vs {k - 1}: {bic[k] - bic[k - 1]:+.1f})")

    if f.n / f.n_params < 20:
        w.append(f"Few events per fitted parameter ({f.n} events / {f.n_params} parameters)")
    wr = f.weights_in_roi
    for i in range(k):
        tag = f"component at {f.mu[i]:.0f} kDa"
        if f.counts[i] < min_events:
            w.append(f"{tag}: only {f.counts[i]:.0f} events")
        elif wr[i] < min_weight:
            w.append(f"{tag}: only {100 * wr[i]:.1f}% of the ROI's events")
        if f.sigma_at_max is not None and f.sigma_at_max[i]:
            w.append(f"{tag}: width reached its upper limit — not a peak, indistinguishable from background")
        elif f.sigma_at_bound is not None and f.sigma_at_bound[i]:
            w.append(f"{tag}: width collapsed to the lower limit — fitting noise")
        elif resolution and f.sigma[i] > 3 * resolution:
            w.append(f"{tag}: much broader (σ {f.sigma[i]:.0f}) than one species (σ ≈ {resolution:.0f} kDa) "
                     "— probably a heterogeneous population or background, not a single peak")
        elif resolution and f.sigma[i] < 0.5 * resolution:
            w.append(f"{tag}: narrower (σ {f.sigma[i]:.0f}) than the instrument resolution "
                     f"(σ ≈ {resolution:.0f} kDa) — likely fitting noise")
        if min(f.mu[i] - res.lo, res.hi - f.mu[i]) < f.sigma[i]:
            w.append(f"{tag}: lies within 1σ of the ROI edge — widen the ROI")
        if f.mu_err is not None and not np.isfinite(f.mu_err[i]):
            w.append(f"{tag}: position uncertainty cannot be determined (parameter at a limit)")
        elif f.mu_err is not None and f.mu_err[i] > 0.5 * f.sigma[i]:
            w.append(f"{tag}: position poorly determined (± {f.mu_err[i]:.0f} kDa)")
    for i in range(k - 1):
        d = ashman_d(f.mu[i], f.sigma[i], f.mu[i + 1], f.sigma[i + 1])
        if d < 2:
            w.append(f"components at {f.mu[i]:.0f} and {f.mu[i + 1]:.0f} kDa are not resolved "
                     f"(Ashman's D = {d:.2f} < 2)")
    return w


def bootstrap_roi(values, res: ROIResult, n_boot: int = 100, seed: int = 1,
                  resolution: float | None = None) -> pd.DataFrame:
    """Resample the events with replacement and refit; unstable components are flagged in
    res.warnings. Returns per-component SDs of mass and weight."""
    x = np.asarray(values, float)
    x_in = x[(x >= res.lo) & (x <= res.hi)]
    rng = np.random.default_rng(seed)
    smin = max(2.0, 0.3 * resolution) if resolution else 2.0
    mus, ws = [], []
    for b in range(n_boot):
        xb = rng.choice(x_in, len(x_in), replace=True)
        try:
            fb = fit_mixture(xb, res.lo, res.hi, res.k, background=res.background, sigma_min=smin,
                             seed=b, n_starts=3, errors=False)
        except ValueError:
            continue
        mus.append(fb.mu); ws.append(fb.weights_in_roi)
    mus, ws = np.array(mus), np.array(ws)
    out = pd.DataFrame({"mass_kDa": res.fit.mu, "mass_sd": mus.std(0), "weight": res.fit.weights_in_roi,
                        "weight_sd": ws.std(0), "n_boot": len(mus)})
    res.bootstrap = out
    res.warnings = [w for w in res.warnings if "bootstrap" not in w]
    for i, r in out.iterrows():
        if r.mass_sd > 0.5 * res.fit.sigma[i]:
            res.warnings.append(f"component at {r.mass_kDa:.0f} kDa: unstable under bootstrap "
                                f"(mass SD {r.mass_sd:.0f} kDa)")
        if r.weight > 0 and r.weight_sd / r.weight > 0.5:
            res.warnings.append(f"component at {r.mass_kDa:.0f} kDa: share unstable under bootstrap "
                                f"({100 * r.weight:.0f} ± {100 * r.weight_sd:.0f}%)")
    return out


def fit_rois(values, rois, *, resolution=None, k_max: int = 4, bootstrap: int = 0,
             **kw) -> list[ROIResult]:
    """Fit several ROIs given as (lo, hi, k) tuples."""
    out = []
    for lo, hi, k in rois:
        r = fit_roi(values, float(lo), float(hi), int(k), resolution=resolution, k_max=k_max, **kw)
        if bootstrap:
            bootstrap_roi(values, r, bootstrap, resolution=resolution)
        out.append(r)
    return out
