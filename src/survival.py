"""Kaplan-Meier with right-censoring, log-rank test, and bootstrap — numpy only."""
import numpy as np


def kaplan_meier(times, events):
    """times: attempts-until-event-or-censor. events: 1=broke defense, 0=censored.
    Returns (t_grid, survival). Survival = P(defense still unbroken after t attempts)."""
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    order = np.argsort(times, kind="mergesort")
    times, events = times[order], events[order]

    uniq = np.unique(times)
    surv, s = [], 1.0
    for t in uniq:
        at_risk = np.sum(times >= t)
        d = np.sum((times == t) & (events == 1))
        if at_risk > 0 and d > 0:
            s *= (1.0 - d / at_risk)
        surv.append(s)
    return uniq, np.array(surv)


def km_median(times, events):
    """Attempts at which 50% of teams have broken the defense.
    np.inf if the curve never reaches 0.5 (majority of teams never succeeded)."""
    t, s = kaplan_meier(times, events)
    below = np.where(s <= 0.5)[0]
    return float(t[below[0]]) if len(below) else np.inf


def km_at(times, events, horizon):
    """P(defense survives past `horizon` attempts). Finite, so always comparable
    even when median is undefined — this is the metric to rank on."""
    t, s = kaplan_meier(times, events)
    prior = np.where(t <= horizon)[0]
    return float(s[prior[-1]]) if len(prior) else 1.0


def logrank(t1, e1, t2, e2):
    """Two-sample log-rank chi-square (1 df). Returns (chi2, approx p)."""
    t1, e1, t2, e2 = map(np.asarray, (t1, e1, t2, e2))
    allt = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))
    O1 = E1 = V = 0.0
    for t in allt:
        n1, n2 = np.sum(t1 >= t), np.sum(t2 >= t)
        d1, d2 = np.sum((t1 == t) & (e1 == 1)), np.sum((t2 == t) & (e2 == 1))
        n, d = n1 + n2, d1 + d2
        if n < 2 or d == 0:
            continue
        O1 += d1
        E1 += d * n1 / n
        V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if V <= 0:
        return 0.0, 1.0
    chi2 = (O1 - E1) ** 2 / V
    # survival function of chi2 with 1 df = erfc(sqrt(chi2/2))
    from math import erfc, sqrt
    return float(chi2), float(erfc(sqrt(chi2 / 2.0)))


def bootstrap_teams(team_records, stat_fn, n_boot=2000, seed=0):
    """Resample TEAMS with replacement (the unit of independence), recompute stat."""
    rng = np.random.default_rng(seed)
    m = len(team_records)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        val = stat_fn([team_records[i] for i in idx])
        if val is not None and np.isfinite(val):
            out.append(val)
    return np.array(out)


def ci(vals, lo=2.5, hi=97.5):
    if len(vals) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, lo)), float(np.percentile(vals, hi)))
