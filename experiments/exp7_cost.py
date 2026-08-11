"""Cost: wall-clock scaling in d, and the price of each inference device.

Claims tested.
(1) BGSN with b = m = d is O(dN) in time: doubling d should roughly double the time at
    fixed N (the per-observation cost is O(d), not O(d^2)).
(2) The i.i.d. plug-in score covariance the base method's theory prescribes costs O(d^2)
    per *observation*, so the dependence-blind interval is strictly MORE expensive than
    our long-run interval, which costs O(d^2) per *block*.
(3) The offline HAC reference costs O(L N d^2) time and O(Nd) memory and is not streaming.

Times are single-threaded (see _env.py) and exclude data generation and JIT warm-up.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import numpy as np
import common
from bgsn import dgp as DG, estimators as E

N = int(os.environ.get("N", 200_000))
DS = [5, 10, 20, 40, 80, 160]
REP = int(os.environ.get("REP", 7))


def work_counts(N, d, b, m, p, plugin, warm_mult=200.0):
    """Exact, machine-independent operation counts for one pass.

    Deterministic accounting, so the cost claim does not depend on machine load:
      gradient           : 2 N d               flops (one dot product and one axpy per obs)
      curvature          : 3 d^2 per rank-one slot (matvec + outer for the data term,
                           outer for the basis-direction ridge term)
      preconditioned step: d^2 per block
      covariance         : 2 d^2 per block (batch-means and lag-one accumulators)
      i.i.d. plug-in     : d^2 per observation   <-- the term that is O(d^2 N)
    """
    T = N // b
    n_warm = int(np.ceil(warm_mult * d / b))
    n_curv = n_warm * b + max(T - n_warm, 0) * (b // m) * p
    return dict(
        n_blocks=T, n_curv=float(n_curv),
        flops_grad=2.0 * N * d,
        flops_curv=3.0 * n_curv * d ** 2,
        flops_step=1.0 * T * d ** 2,
        flops_cov=2.0 * T * d ** 2,
        flops_plugin=(1.0 * N * d ** 2 if plugin else 0.0),
        flops_total=(2.0 * N * d + 3.0 * n_curv * d ** 2 + 3.0 * T * d ** 2
                     + (1.0 * N * d ** 2 if plugin else 0.0)))


def time_interleaved(fns, rep=REP):
    """Time several callables with the repetitions interleaved, reporting the minimum.

    Interleaving matters on a shared machine: if all repetitions of method A ran before all
    repetitions of method B, a change in external load between the two would be
    indistinguishable from a difference between the methods.  Round-robin ordering exposes
    every method to the same load pattern, and the minimum over repetitions is the standard
    robust summary, since external load can only ever make a measurement slower.
    """
    for f in fns.values():                 # JIT / cache warm-up, not timed
        f()
    best = {k: np.inf for k in fns}
    for _ in range(rep):
        for k, f in fns.items():
            t0 = time.perf_counter()
            f()
            best[k] = min(best[k], time.perf_counter() - t0)
    return {k: float(v) for k, v in best.items()}


if __name__ == "__main__":
    rows = []
    for d in DS:
        g = DG.make_lin_hom(d=d, phi=0.7, psi=0.7)
        X, y = g.sample(N, 1)
        X = np.ascontiguousarray(X); y = np.ascontiguousarray(y)
        b = d
        r = dict(d=d, N=N, b=b)
        fns = {
            "BGSN": lambda: E.streaming_newton(X, y, "linear", b=b, m=b, p=1.0,
                                               variance="twoscale", seed=0),
            "BGSN_plugin": lambda: E.streaming_newton(X, y, "linear", b=b, m=b, p=1.0,
                                                      variance="plugin", seed=0),
            "SN_iid_p1": lambda: E.streaming_newton(X, y, "linear", b=b, m=1, p=1.0,
                                                    variance="twoscale", seed=0),
            "SN_iid_pinv_d": lambda: E.streaming_newton(X, y, "linear", b=b, m=1,
                                                        p=1.0 / d, variance="twoscale",
                                                        seed=0),
            "ASGD": lambda: E.first_order(X, y, "linear", b=b, c_step=0.1,
                                          variance="randomscaling", seed=0),
        }
        r.update(time_interleaved(fns, REP))
        if d <= 40:
            r.update(time_interleaved(
                {"OfflineHAC_L100": lambda: E.offline_hac(X, y, "linear", lag=100)}, 2))
        r["work"] = dict(
            BGSN=work_counts(N, d, b, b, 1.0, False),
            BGSN_plugin=work_counts(N, d, b, b, 1.0, True),
            SN_iid_p1=work_counts(N, d, b, 1, 1.0, False),
            SN_iid_pinv_d=work_counts(N, d, b, 1, 1.0 / d, False))
        rows.append(r)
        print(f"d={d:4d}  BGSN {r['BGSN']*1e3:8.1f}ms  "
              f"BGSN+plugin {r['BGSN_plugin']*1e3:9.1f}ms  "
              f"SN-iid(p=1) {r['SN_iid_p1']*1e3:9.1f}ms  "
              f"SN-iid(p=1/d) {r['SN_iid_pinv_d']*1e3:8.1f}ms  "
              f"ASGD {r['ASGD']*1e3:7.1f}ms" +
              (f"  offlineHAC {r['OfflineHAC_L100']*1e3:9.0f}ms"
               if "OfflineHAC_L100" in r else ""))
    print("\n-- deterministic operation counts (flops, leading order) --")
    for r in rows:
        wk = r["work"]
        print(f"  d={r['d']:4d}  BGSN {wk['BGSN']['flops_total']:.3e}  "
              f"+plugin {wk['BGSN_plugin']['flops_total']:.3e}  "
              f"SN-iid(p=1) {wk['SN_iid_p1']['flops_total']:.3e}  "
              f"SN-iid(p=1/d) {wk['SN_iid_pinv_d']['flops_total']:.3e}")
    ldd = np.log(np.array(DS, float))
    for k in ["BGSN", "BGSN_plugin", "SN_iid_p1", "SN_iid_pinv_d"]:
        fl = np.log([r["work"][k]["flops_total"] for r in rows])
        print(f"  flop exponent in d, {k:15s}: {np.polyfit(ldd[-4:], fl[-4:], 1)[0]:.3f}")

    # empirical exponents  time ~ C d^s  at fixed N
    ld = np.log(np.array(DS, float))
    for k in ["BGSN", "BGSN_plugin", "SN_iid_p1", "SN_iid_pinv_d", "ASGD"]:
        t = np.log(np.array([r[k] for r in rows]))
        s = np.polyfit(ld[-4:], t[-4:], 1)[0]
        print(f"  empirical exponent in d (last four points), {k:15s}: {s:.2f}")
    common.save("exp7_cost.json",
                dict(N=N, ds=DS, rep=REP, rows=rows,
                     flop_exponents={k: float(np.polyfit(ld[-4:],
                                     np.log([r["work"][k]["flops_total"]
                                             for r in rows])[-4:], 1)[0])
                                     for k in ["BGSN", "BGSN_plugin", "SN_iid_p1",
                                               "SN_iid_pinv_d"]},
                     exponents={k: float(np.polyfit(ld[-4:],
                                                    np.log([r[k] for r in rows])[-4:], 1)[0])
                                for k in ["BGSN", "BGSN_plugin", "SN_iid_p1",
                                          "SN_iid_pinv_d", "ASGD"]}))
