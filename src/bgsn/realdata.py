"""Real hourly streams: road traffic and urban air quality.

Two public UCI series are used, both hourly and both strongly autocorrelated after
any reasonable regression adjustment:

* ``metro``  -- Metro Interstate Traffic Volume (I-94 westbound, Minneapolis-St Paul,
  2012-10 to 2018-09; Hogue, UCI ML Repository).  Target: log traffic volume.
* ``airq``   -- Beijing Multi-Site Air-Quality Data (12 monitoring sites, 2013-03 to
  2017-02; Zhang et al., UCI ML Repository).  Target: log(PM2.5 + 1) at one site.

Both are read into a *time-ordered* design matrix with an intercept, standardised
continuous predictors and calendar harmonics.  Timestamps are sorted and de-duplicated;
missing hours are simply absent, so "lag k" means "k observations back" rather than
"k hours back".  This is the usual convention for HAC/blocking on irregular streams and
does not affect validity, only the interpretation of the block length.

Nothing here is fitted or tuned on the response: the feature construction is fixed in
advance for both datasets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


@dataclass
class RealStream:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list
    note: str


def _harmonics(hour: np.ndarray, dow: np.ndarray) -> tuple:
    cols, names = [], []
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * hour / 24.0), np.cos(2 * np.pi * k * hour / 24.0)]
        names += [f"sin{k}_day", f"cos{k}_day"]
    cols += [np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0)]
    names += ["sin_week", "cos_week"]
    return cols, names


def _standardise(cols: list, names: list) -> tuple:
    out = []
    for c in cols:
        c = np.asarray(c, float)
        s = c.std()
        out.append((c - c.mean()) / (s if s > 0 else 1.0))
    return out, names


def load_metro(data_dir: str = DATA_DIR) -> RealStream:
    path = os.path.join(data_dir, "Metro_Interstate_Traffic_Volume.csv.gz")
    # keep_default_na=False: the literal string "None" in `holiday` must not become NaN
    df = pd.read_csv(path, compression="gzip", keep_default_na=False)
    for c in ("temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["temp", "rain_1h", "clouds_all", "traffic_volume"])
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = (df.sort_values("date_time")
            .drop_duplicates(subset="date_time", keep="first")
            .reset_index(drop=True))
    df = df[df["traffic_volume"] > 0].reset_index(drop=True)
    # a handful of sentinel temperatures (0 K) and rainfall (9831 mm) are dropped
    df = df[(df["temp"] > 200) & (df["rain_1h"] < 100)].reset_index(drop=True)

    hour = df["date_time"].dt.hour.to_numpy()
    dow = df["date_time"].dt.dayofweek.to_numpy()
    # the raw `holiday` label is stamped only on the 00:00 row of a holiday; propagate it
    # to every hour of that calendar date, which is what the feature is meant to encode
    hol_dates = set(df.loc[df["holiday"] != "None", "date_time"].dt.date)
    holiday = df["date_time"].dt.date.isin(hol_dates).to_numpy(float)

    # NB: rainfall is deliberately excluded.  `rain_1h` is identically zero over stretches
    # of several thousand consecutive hours, so log1p(rain) is a constant column on some
    # contiguous segments, which makes the design exactly rank deficient there and the
    # per-segment population targets undefined.  Temperature and cloud cover carry the
    # weather signal.
    hcols, hnames = _harmonics(hour, dow)
    raw = [df["temp"].to_numpy(), df["clouds_all"].to_numpy()] + hcols
    names = ["temp", "clouds"] + hnames
    cols, names = _standardise(raw, names)
    cols = [np.ones(len(df))] + cols + [holiday, (dow >= 5).astype(float)]
    names = ["intercept"] + names + ["holiday", "weekend"]
    X = np.column_stack(cols)
    # The response is standardised for the same reason the covariates are.  Without it the
    # mean of log(volume) -- about 7.9 -- sits entirely in the intercept, which is then some
    # 400 standard errors away from the theta_0 = 0 that any streaming method starts from; a
    # 1/t schedule needs of order that many steps just to travel the distance, so a short
    # segment would be measuring the distance to the initialisation rather than anything about
    # inference.  Standardising is an affine reparametrisation of a linear model, so it changes
    # no coverage statement.
    y = _standardise([np.log(df["traffic_volume"].to_numpy(float))], ["y"])[0][0]
    return RealStream("metro", np.ascontiguousarray(X), np.ascontiguousarray(y), names,
                      note="hourly I-94 westbound volume; response = log(volume)")


def load_airq(site: str = "Aotizhongxin", data_dir: str = DATA_DIR) -> RealStream:
    base = os.path.join(data_dir, "airq", "PRSA_Data_20130301-20170228")
    path = os.path.join(base, f"PRSA_Data_{site}_20130301-20170228.csv")
    df = pd.read_csv(path)
    keep = ["PM2.5", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "hour", "year", "month", "day"]
    df = df[keep].dropna().reset_index(drop=True)
    df = df[df["PM2.5"] >= 0].reset_index(drop=True)
    ts = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"],
                             hour=df["hour"]))
    order = np.argsort(ts.to_numpy())
    df = df.iloc[order].reset_index(drop=True)
    ts = ts.iloc[order].reset_index(drop=True)

    hour = df["hour"].to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    hcols, hnames = _harmonics(hour, dow)
    raw = [df["TEMP"].to_numpy(), df["PRES"].to_numpy(), df["DEWP"].to_numpy(),
           np.log1p(df["RAIN"].to_numpy()), df["WSPM"].to_numpy()] + hcols
    names = ["TEMP", "PRES", "DEWP", "log_RAIN", "WSPM"] + hnames
    cols, names = _standardise(raw, names)
    cols = [np.ones(len(df))] + cols
    names = ["intercept"] + names
    X = np.column_stack(cols)
    y = _standardise([np.log1p(df["PM2.5"].to_numpy(float))], ["y"])[0][0]
    return RealStream(f"airq[{site}]", np.ascontiguousarray(X),
                      np.ascontiguousarray(y), names,
                      note="hourly PM2.5 at one Beijing site; response = log(1 + PM2.5)")


AIRQ_SITES = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng",
              "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"]


# --------------------------------------------------------------------------------------
# semi-synthetic protocol: real covariates, AR(1) errors, exact conditional oracle
# --------------------------------------------------------------------------------------
def conditional_oracle(X: np.ndarray, psi: float, sigma_u: float, theta_star: np.ndarray,
                       max_lag: int = 400):
    r"""Exact oracle for ``y_t = x_t'theta* + u_t`` with a *given* covariate path.

    With ``u`` an AR(1) of lag-k autocovariance ``sigma_u^2 psi^{|k|}`` independent of X,
    the score is ``s_t = -u_t x_t`` and, conditionally on the covariate path,

        H      = n^{-1} sum_t x_t x_t'
        Gamma0 = sigma_u^2 H
        S      = sigma_u^2 [ C_0 + sum_{k>=1} psi^k (C_k + C_k') ],
        C_k    = n^{-1} sum_t x_t x_{t+k}'.

    The sum is truncated where ``psi^k`` underflows the requested tolerance, so the
    oracle is exact to machine precision for the realised covariate path.  This gives a
    ground-truth ``theta*`` **and** a ground-truth long-run covariance on genuinely real
    covariate dependence.
    """
    from .streams import Oracle
    n = X.shape[0]
    C0 = X.T @ X / n
    S = C0.copy()
    k = 1
    while k <= max_lag and psi ** k > 1e-14:
        Ck = X[:-k].T @ X[k:] / n
        S += psi ** k * (Ck + Ck.T)
        k += 1
    return Oracle(theta_star=theta_star, H=C0, Gamma0=sigma_u ** 2 * C0,
                  S=sigma_u ** 2 * S, exact=True,
                  meta=dict(kind="conditional", psi=psi, lags_used=k - 1))


def ar1_errors(n: int, psi: float, sigma_u: float, seed: int) -> np.ndarray:
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    u0 = sigma_u * rng.standard_normal()
    e = rng.standard_normal(n) * sigma_u * np.sqrt(1 - psi ** 2)
    return lfilter([1.0], [1.0, -psi], e, zi=[psi * u0])[0]
