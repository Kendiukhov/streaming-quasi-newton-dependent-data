"""Critical values of the random-scaling pivot, simulated rather than copied.

The random-scaling t-statistic of Lee, Liao, Seo and Shin converges to
``W(1) / sqrt(int_0^1 (W(r) - r W(1))^2 dr)``.  We simulate its quantiles so that the
number used in the experiments is verifiable from this repository, and we cross-check
against the tabulation in Kiefer, Vogelsang and Bunzel (2000).

Two-sided 1-alpha intervals need the (1-alpha/2) quantile.  Several papers quote 5.32 as
"the 95% critical value"; that is the ONE-sided 95% quantile and yields badly
under-covering two-sided intervals.  We report both so the distinction is explicit.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import common  # noqa: E402
from bgsn import estimators as E  # noqa: E402

if __name__ == "__main__":
    # We simulate the ABSOLUTE pivot |T|.  The published tables list quantiles of the
    # SIGNED statistic T, so the q-quantile of |T| must be compared with the
    # (1+q)/2-quantile of T.  Getting this wrong by one level is exactly the trap that
    # leads papers to quote 5.32 as a two-sided 95% value.
    levels = (0.80, 0.90, 0.95, 0.98)
    q = E.random_scaling_critical_values(levels=levels, n_paths=4_000_000,
                                        n_terms=400, seed=20260810)
    published_signed = {0.90: 3.890, 0.95: 5.374, 0.975: 6.811, 0.99: 8.544}
    rows = []
    for lv in levels:
        signed_lv = (1 + lv) / 2
        rows.append(dict(abs_quantile=lv, signed_quantile=signed_lv, simulated=q[lv],
                         published_kvb_signed=published_signed.get(round(signed_lv, 3))))
    for r in rows:
        pub = r["published_kvb_signed"]
        print(f"  q_{r['abs_quantile']:.2f}(|T|) = q_{r['signed_quantile']:.3f}(T): "
              f"simulated {r['simulated']:.3f}"
              + (f" | KVB(2000)/LLSS Table 2: {pub:.3f}" if pub else ""))
    print(f"\n  two-sided 95% critical value = q_0.95(|T|) = q_0.975(T): "
          f"simulated {q[0.95]:.3f}, published {published_signed[0.975]:.3f}; "
          f"we use the published value ({common.RS_CRIT95}), which is the more "
          f"conservative of the two.")
    print(f"  (5.374 is q_0.95 of the SIGNED statistic = q_0.90(|T|), simulated "
          f"{q[0.90]:.3f}: using it as a two-sided 95% value gives ~90% intervals.)")
    common.save("exp0_critical_values.json",
                dict(rows=rows, n_paths=4_000_000, n_terms=400,
                     simulated_two_sided_95=q[0.95],
                     used_two_sided_95=common.RS_CRIT95,
                     note="rows compare q_lv(|T|) with the published q_{(1+lv)/2}(T)"))
