"""Build every figure in the paper from the saved JSON results.

Design rules: one message per figure; the message is stated in the title; series are
labelled directly rather than only in a legend where that is clearer; a colourblind-safe
palette (no red/green pairing); nominal levels drawn as reference lines, never implied.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import common

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "lines.linewidth": 1.6, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "legend.frameon": False,
})

# colourblind-safe (Okabe-Ito); no red/green pairs used to encode the same axis
C = dict(ours="#0072B2", bm="#56B4E9", plugin="#D55E00", base="#CC79A7",
         asgd="#E69F00", rs="#009E73", offline="#666666", oracle="#000000")


def _ok(name):
    p = os.path.join(common.RESULTS, name)
    return os.path.exists(p)



def _logticks(ax, values):
    """Label a log x-axis at exactly the sampled values, compactly, with no minor ticks.

    Matplotlib's default log locator adds 2x, 3x, 4x minor labels which, at the width of a
    three-panel figure, overlap into an unreadable smear.
    """
    import matplotlib.ticker as mt
    vals = sorted(set(values))
    ax.set_xticks(vals)
    ax.set_xticklabels([("%g" % (v / 10 ** int(np.floor(np.log10(v)))))
                        + r"$\times10^{%d}$" % int(np.floor(np.log10(v))) for v in vals],
                       fontsize=7)
    ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.set_xlim(min(vals) / 1.35, max(vals) * 1.35)


def savefig(fig, name):
    path = os.path.join(common.FIGURES, name)
    fig.savefig(path)
    plt.close(fig)
    print("[fig]", path)


# --------------------------------------------------------------------------------------
def fig1_headline():
    if not (_ok("exp2_coverage_law.json") and _ok("exp1_main.json")):
        return
    d2 = common.load("exp2_coverage_law.json")
    d1 = common.load("exp1_main.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1),
                           gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.28})

    # (a) the coverage law
    r = d2["rows"]
    x = np.array([q["kappa"] for q in r])
    grid = np.linspace(1, max(x) * 1.02, 200)
    a = ax[0]
    a.plot(grid, 2 * norm.cdf(common.Z95 / np.sqrt(grid)) - 1, color=C["plugin"],
           lw=1.4, zorder=1,
           label=r"theory  $2\Phi(z_{0.025}/\sqrt{\kappa})-1$")
    a.errorbar(x, [q["cov_BGSN-plugin"] for q in r],
               yerr=[1.96 * q["covse_BGSN-plugin"] for q in r], fmt="o", ms=4.5,
               color=C["plugin"], mfc="white", mew=1.3, capsize=2, lw=1,
               label="measured, dependence-blind interval")
    a.errorbar(x, [q["cov_BGSN"] for q in r], yerr=[1.96 * q["covse_BGSN"] for q in r],
               fmt="s", ms=4.5, color=C["ours"], capsize=2, lw=1,
               label="measured, BGSN (this paper)")
    a.axhline(0.95, color="k", ls=":", lw=1)
    a.text(x.max(), 0.955, "nominal 95%", ha="right", va="bottom", fontsize=7.5)
    a.set_xlabel(r"variance inflation $\kappa=(1{+}\phi\psi)/(1{-}\phi\psi)$")
    a.set_ylabel("coverage of a nominal 95% interval")
    a.set_ylim(0.5, 1.0)
    a.set_title("(a) Dependence-blind intervals fail by a\npredictable amount",
                loc="left")
    a.legend(loc="lower left")

    # (b) coverage by method on the AR-hom design.
    # A width-versus-coverage scatter was the original design and it no longer works: with the
    # final estimator, BGSN, plain batch means, the oracle and offline HAC all land within 0.01
    # of one another at coverage ~0.95 and width ~1.0, so markers and labels sat on top of each
    # other.  The clustering is the result, so the fix is an encoding in which coincident values
    # cannot hide -- one row per method, coverage on the axis, width printed alongside.
    a = ax[1]
    rows = {q["method"]: q for q in d1["AR-hom"]["rows"]}
    order = ["Oracle-var", "Offline-HAC", "BGSN", "BGSN-BM", "SN-iid+2scale",
             "BGSN-plugin", "SN-iid", "AdaGrad-2scale", "ASGD-RS", "ASGD-2scale",
             "ASGD-plugin"]
    label = {"BGSN": "BGSN (ours)", "BGSN-BM": "BGSN, plain batch means",
             "BGSN-plugin": "BGSN, i.i.d. plug-in", "SN-iid": "streaming Newton, i.i.d. theory",
             "SN-iid+2scale": "streaming Newton + our variance",
             "ASGD-plugin": "ASGD, i.i.d. plug-in", "ASGD-2scale": "ASGD + our variance",
             "ASGD-RS": "ASGD + random scaling", "AdaGrad-2scale": "AdaGrad + our variance",
             "Offline-HAC": "offline HAC (not streaming)", "Oracle-var": "oracle variance"}
    colour = {"BGSN": C["ours"], "BGSN-BM": C["bm"], "BGSN-plugin": C["plugin"],
              "SN-iid": C["base"], "SN-iid+2scale": C["ours"], "ASGD-plugin": C["asgd"],
              "ASGD-2scale": C["asgd"], "ASGD-RS": C["rs"], "AdaGrad-2scale": C["asgd"],
              "Offline-HAC": C["offline"], "Oracle-var": C["oracle"]}
    present = [m for m in order if m in rows]
    y = list(range(len(present)))[::-1]
    a.axvline(0.95, color="k", ls=":", lw=1, zorder=1)
    for yi, m in zip(y, present):
        q = rows[m]
        a.plot([0, q["coverage"]], [yi, yi], color=colour[m], lw=1.1, alpha=0.45, zorder=2)
        a.plot(q["coverage"], yi, "o", ms=5.5, color=colour[m], zorder=3,
               mec="white", mew=0.6)
        a.text(1.005, yi, r"$\times$%.2f" % q["width_rel"], va="center", ha="left",
               fontsize=6.4, color="0.25")
    # Row labels go INSIDE the panel: as y-tick labels they were wide enough to run into
    # panel (a) and collide with its legend.
    for yi, m in zip(y, present):
        a.text(-0.03, yi, label[m], va="center", ha="right", fontsize=6.6,
               color=colour[m] if m not in ("Oracle-var", "Offline-HAC") else "0.2")
    a.set_yticks([])
    a.set_xlim(-0.78, 1.17)
    a.set_ylim(-0.8, len(present) - 0.2)
    a.set_xticks([0.0, 0.25, 0.5, 0.75, 0.95])
    a.set_xticklabels(["0", "0.25", "0.5", "0.75", "0.95"])
    a.set_xlabel("coverage of a nominal 95% interval")
    a.grid(axis="y", visible=False)
    a.spines["left"].set_visible(False)
    a.text(1.005, len(present) - 0.42, "width", fontsize=6.4, color="0.25", va="center")
    a.set_title("(b) Only the long-run variance reaches\nnominal coverage", loc="left")
    savefig(fig, "fig1_headline.pdf")


# --------------------------------------------------------------------------------------
def fig2_lrv():
    if not _ok("exp3_lrv.json"):
        return
    d = common.load("exp3_lrv.json")
    r = d["rows"]
    b = np.array([q["b"] for q in r], float)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    a = ax[0]
    a.axhline(1.0, color="k", ls=":", lw=1)
    # exact expectations, both closed form for this design and both plotted:
    #   batch means : sum_{|k|<b} (1-|k|/b) r^|k|   / [(1+r)/(1-r)]
    #   two-scale   : weights 1 for |k|<=b, (2b-|k|)/b for b<|k|<2b
    rr = d["phi"] * d["psi"]
    kap = (1 + rr) / (1 - rr)

    def th_bm(bb):
        k = np.arange(1, bb)
        return (1 + 2 * np.sum((1 - k / bb) * rr ** k)) / kap

    def th_ft(bb):
        k1 = np.arange(1, bb + 1)
        k2 = np.arange(bb + 1, 2 * bb)
        return (1 + 2 * np.sum(rr ** k1)
                + 2 * np.sum(((2 * bb - k2) / bb) * rr ** k2)) / kap

    a.plot(b, [th_bm(int(x)) for x in b], color=C["bm"], lw=4, alpha=0.3, zorder=1,
           solid_capstyle="round", label="exact expectation, batch means")
    a.plot(b, [th_ft(int(x)) for x in b], color=C["ours"], lw=4, alpha=0.3, zorder=1,
           solid_capstyle="round", label="exact expectation, two-scale")
    a.errorbar(b, [q["bm_alg"] for q in r], yerr=[1.96 * q["bm_alg_se"] for q in r],
               fmt="o", ms=4.5, color=C["bm"], mfc="white", mew=1.2, capsize=2, lw=0,
               label="measured, plain batch means")
    a.errorbar(b, [q["ft_alg"] for q in r], yerr=[1.96 * q["ft_alg_se"] for q in r],
               fmt="s", ms=4.5, color=C["ours"], capsize=2, lw=0,
               label="measured, two-scale (used by BGSN)")
    a.set_xscale("log"); a.set_xticks([5, 20, 80, 320])
    a.set_xticklabels(["5", "20", "80", "320"])
    a.set_xlabel("block length $b$")
    a.set_ylabel(r"$\mathbb{E}[\widehat{S}]/S$  (diagonal mean)")
    a.set_title("(a) Both estimators match their exact\nexpectations at every block length",
                loc="left")
    a.legend(loc="lower right", fontsize=7.2)

    a = ax[1]
    a.axhline(0.0, color="k", ls=":", lw=1)
    a.plot(b, [abs(q["ft_alg"] - q["ft_score"]) for q in r], "o-", ms=4,
           color=C["ours"], label=r"$|\widehat S$ from iterates $-\;\widehat S$ at $\theta^\star|$")
    a.plot(b, [abs(q["ft_drift"]) for q in r], "s-", ms=4, color=C["plugin"],
           label="drift-only component")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks([5, 20, 80, 320]); a.set_xticklabels(["5", "20", "80", "320"])
    a.set_xlabel("block length $b$")
    a.set_ylabel(r"contribution relative to $S$")
    a.set_title("(b) The moving iterate contributes\n$O((b{+}d)/N)$: essentially nothing",
                loc="left")
    a.legend(loc="upper left")
    savefig(fig, "fig2_lrv.pdf")


# --------------------------------------------------------------------------------------
def fig3_gapping():
    if not _ok("exp4_gapping.json"):
        return
    d = common.load("exp4_gapping.json")
    rows = d["rows"]
    m = np.array([q["m"] for q in rows], float)
    # EXACT independent-data benchmark, not a fitted normalisation: for Gaussian x with
    # covariance Sigma and alpha == 1, n * E||Hhat - H||_F^2 = (tr Sigma)^2 + ||Sigma||_F^2.
    from bgsn import dgp as DG
    Sig = DG.make_lin_hom(d=20, phi=d["phi"], psi=0.7).oracle.H
    base = float(np.trace(Sig) ** 2 + np.sum(Sig ** 2))
    fig, a = plt.subplots(figsize=(3.9, 3.0))
    a.axhline(1.0, color="k", ls=":", lw=1)
    a.text(1.05, 1.03, "i.i.d. benchmark", ha="left", va="bottom", fontsize=7.5)
    # predictions drawn wide and pale behind the measurements, which are markers only
    a.plot(m, 1 + np.array([q["pred_excess_bern"] for q in rows]), lw=5, alpha=0.28,
           color=C["plugin"], solid_capstyle="round",
           label=r"predicted, thinning: $1+(\kappa_H-1)/m$")
    a.plot(m, 1 + np.array([q["pred_excess_gap"] for q in rows]), lw=5, alpha=0.28,
           color=C["ours"], solid_capstyle="round",
           label=r"predicted, gapping: $1+O(\rho^{m})$")
    a.plot(m, np.array([q["nvar_bern"] for q in rows]) / base, "o", ms=5,
           color=C["plugin"], mfc="white", mew=1.5,
           label="measured, Bernoulli thinning")
    a.plot(m, np.array([q["nvar_gap"] for q in rows]) / base, "s", ms=5,
           color=C["ours"], label="measured, deterministic gapping")
    a.set_xscale("log")
    a.set_xticks([1, 2, 5, 10, 20, 40, 80])
    a.set_xticklabels(["1", "2", "5", "10", "20", "40", "80"])
    a.set_xlabel("gap $m$ (mean spacing of curvature updates)")
    a.set_ylabel("curvature-estimator variance,\nrelative to the i.i.d. benchmark")
    a.set_title("At matched cost, gapping decorrelates\nexponentially, thinning only as $1/m$",
                loc="left")
    a.legend(loc="upper right")
    savefig(fig, "fig3_gapping.pdf")


# --------------------------------------------------------------------------------------
def fig4_cost():
    if not _ok("exp7_cost.json"):
        return
    d = common.load("exp7_cost.json")
    r = d["rows"]
    ds = np.array([q["d"] for q in r], float)
    fig, a = plt.subplots(figsize=(3.7, 2.9))
    series = [("BGSN", "BGSN: long-run interval", C["ours"], "s"),
              ("BGSN_plugin", "the same, with the i.i.d.\nplug-in variance", C["plugin"], "o"),
              ("SN_iid_p1", "un-thinned curvature ($p=1$)", C["base"], "D"),
              ("ASGD", "ASGD (first order)", C["asgd"], "^")]
    for k, lab, col, mk in series:
        v = np.array([q[k] for q in r], float) * 1e3
        a.plot(ds, v, marker=mk, ms=4.5, color=col, label=lab)
    if any("OfflineHAC_L100" in q for q in r):
        dd = [q["d"] for q in r if "OfflineHAC_L100" in q]
        vv = [q["OfflineHAC_L100"] * 1e3 for q in r if "OfflineHAC_L100" in q]
        a.plot(dd, vv, marker="P", ms=4.5, color=C["offline"],
               label="offline HAC (not streaming)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("dimension $d$   (block length $b=d$, $N=%s$)" % f"{d['N']:,}")
    a.set_ylabel("wall-clock time (ms, single thread)")
    e = d["exponents"]
    big = d.get("flop_exponents_large_N", {})
    ratio = np.median([q["BGSN_plugin"] / q["BGSN"] for q in
                       [{k: r[k] for k in ("BGSN", "BGSN_plugin")} for r in r[2:]]])
    rr = [q["BGSN_plugin"] / q["BGSN"] for q in r]
    ttl = ("The long-run interval is cheaper at every $d$:\n"
           "$%.1f$--$%.1f\\times$ faster than the i.i.d. plug-in" % (min(rr), max(rr)))
    if big:
        ttl += " (asymptotically $d^{%.2f}$ vs $d^{%.2f}$)" % (big["BGSN"],
                                                              big["BGSN_plugin"])
    a.set_title(ttl, loc="left", fontsize=8.2)
    # Five curves fan across the whole panel, so there is no free corner: put the legend below
    # the axes.  savefig uses bbox_inches="tight", so it is not clipped.
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.3,
             frameon=False, handlelength=1.6, columnspacing=1.1, labelspacing=0.25)
    savefig(fig, "fig4_cost.pdf")


# --------------------------------------------------------------------------------------
def fig5_ablations():
    if not _ok("exp5_ablations.json"):
        return
    d = common.load("exp5_ablations.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"wspace": 0.42})
    a = ax[0]
    r = d["N"]
    n = [q["N"] for q in r]
    for k, lab, col, mk in (("cov_ft", "two-scale (BGSN)", C["ours"], "s"),
                            ("cov_bm", "plain batch means", C["bm"], "v"),
                            ("cov_plugin", "i.i.d. plug-in", C["plugin"], "o")):
        a.errorbar(n, [q[k] for q in r], yerr=[1.96 * q[k + "_se"] for q in r],
                   marker=mk, ms=4, color=col, capsize=2, lw=1.3, label=lab)
    a.axhline(0.95, color="k", ls=":", lw=1)
    a.set_xscale("log"); a.set_xlabel("stream length $N$"); a.set_ylabel("coverage")
    a.set_ylim(0.6, 1.0); a.set_title("(a) sample size", loc="left", fontsize=8.4)
    a.legend(loc="center right", fontsize=6.4, framealpha=0.95, borderpad=0.3,
             labelspacing=0.22)

    a = ax[1]
    r = d["b"]
    bb = [q["b"] for q in r]
    a.errorbar(bb, [q["cov_ft"] for q in r], yerr=[1.96 * q["cov_ft_se"] for q in r],
               marker="s", ms=4, color=C["ours"], capsize=2, lw=1.3, label="two-scale")
    a.errorbar(bb, [q["cov_bm"] for q in r], yerr=[1.96 * q["cov_bm_se"] for q in r],
               marker="v", ms=4, color=C["bm"], capsize=2, lw=1.3, label="batch means")
    a.axhline(0.95, color="k", ls=":", lw=1)
    a.set_xscale("log"); a.set_xlabel("block length $b$")
    a.set_ylim(0.6, 1.0)          # same scale as (a); a second "coverage" label collided with it
    a.set_title("(b) block length ($N=200{,}000$)", loc="left", fontsize=8.4)
    a.legend(loc="lower center", fontsize=6.4, framealpha=0.95, borderpad=0.3,
             labelspacing=0.22)

    a = ax[2]
    r = d["warm"]
    w = [q["warm_mult"] for q in r]
    a.errorbar(w, [q["cov_ft"] for q in r], yerr=[1.96 * q["cov_ft_se"] for q in r],
               marker="s", ms=4, color=C["ours"], capsize=2, lw=1.3)
    a2 = a.twinx()
    a2.plot(w, [q["eff"] for q in r], marker="o", ms=4, color=C["plugin"], lw=1.2)
    a2.set_ylabel("variance / theory", color=C["plugin"])
    a2.tick_params(axis="y", colors=C["plugin"]); a2.grid(False)
    a.axhline(0.95, color="k", ls=":", lw=1)
    a.set_xscale("log")
    a.set_xlabel(r"warm-up $c_{\mathrm{w}}$ (updates $/\,d$)")
    a.set_ylabel("coverage", color=C["ours"])
    a.set_title("(c) curvature warm-up", loc="left", fontsize=8.4)
    savefig(fig, "fig5_ablations.pdf")


# --------------------------------------------------------------------------------------
def fig6_real():
    if not _ok("exp6_real.json"):
        return
    d = common.load("exp6_real.json")
    order = ["BGSN", "BGSN-BM", "SN-iid", "BGSN-plugin", "ASGD-2scale", "ASGD-plugin",
             "ASGD-RS"]
    lab = {"BGSN": "BGSN (ours)", "BGSN-BM": "BGSN, batch means",
           "SN-iid": "streaming Newton,\ni.i.d. theory",
           "BGSN-plugin": "BGSN, i.i.d. plug-in",
           "ASGD-2scale": "ASGD + our variance", "ASGD-plugin": "ASGD, i.i.d. plug-in",
           "ASGD-RS": "ASGD + random scaling"}
    col = {"BGSN": C["ours"], "BGSN-BM": C["bm"], "SN-iid": C["base"],
           "BGSN-plugin": C["plugin"], "ASGD-2scale": C["asgd"],
           "ASGD-plugin": C["plugin"], "ASGD-RS": C["rs"]}
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.0),
                           gridspec_kw=dict(width_ratios=[1.35, 0.95, 1.1], wspace=0.34))
    for k, (key, title) in enumerate([("protoA_metro", "(a) traffic, exact truth"),
                                      ("protoA_airq", "(b) air quality, exact truth")]):
        a = ax[k]
        rows = {}
        for q in d[key]["rows"]:
            rows.setdefault(q["method"], []).append(q)
        ms = [m for m in order if m in rows]
        y = np.arange(len(ms))
        vals = [np.mean([q["coverage"] for q in rows[m]]) for m in ms]
        errs = [1.96 * np.sqrt(np.mean([q["coverage_se"] ** 2 for q in rows[m]]))
                for m in ms]
        a.barh(y, vals, xerr=errs, color=[col[m] for m in ms], height=0.66,
               error_kw=dict(lw=0.9, capsize=2))
        a.set_yticks(y); a.set_yticklabels([lab[m] for m in ms] if k == 0 else [])
        a.invert_yaxis()
        a.axvline(0.95, color="k", ls=":", lw=1)
        a.set_xlim(0, 1.0)
        a.set_xticks([0, 0.25, 0.5, 0.75, 0.95])
        a.set_xticklabels(["0", "", "0.5", "", "0.95"])
        # one label under the pair: two full-length labels collided in the middle
        if k == 0:
            a.set_xlabel("coverage of a nominal 95% interval", x=1.02)
        a.set_title(title, loc="left", fontsize=8.6)
        a.grid(axis="y", alpha=0)

    a = ax[2]
    r = d["protoA_metro_bsweep"]
    bb = [q["b"] for q in r]
    a.plot(bb, [q["coverage"]["BGSN"] for q in r], "s-", ms=4, color=C["ours"],
           label="two-scale (BGSN)")
    a.plot(bb, [q["coverage"]["BGSN-BM"] for q in r], "v-", ms=4, color=C["bm"],
           label="batch means")
    a.plot(bb, [q["coverage"]["BGSN-plugin"] for q in r], "o-", ms=4, color=C["plugin"],
           label="i.i.d. plug-in")
    # The oracle-variance curve is the comparison that matters here: the shortfall from 0.95 on
    # a 8000-hour segment is the point estimate, not the covariance, and our curve tracking the
    # oracle's is what shows that.
    if all("Oracle-var" in q["coverage"] for q in r):
        a.plot(bb, [q["coverage"]["Oracle-var"] for q in r], "--", lw=1.1, color=C["oracle"],
               marker="*", ms=6, label="oracle variance")
    a.axhline(0.95, color="k", ls=":", lw=1)
    a.set_xscale("log"); a.set_xlabel("block length $b$ (traffic)")
    a.set_ylabel("coverage"); a.set_ylim(0.25, 1.02)
    a.set_title("(c) block length, real covariates", loc="left", fontsize=8.6)
    a.legend(loc="lower left", fontsize=5.9, handlelength=1.5, framealpha=0.95,
             frameon=True, borderpad=0.3, labelspacing=0.22)
    # The block-adequacy diagnostic on a twin axis.  The caption claims it is overlaid here, so
    # it has to be drawn: an earlier version described a curve that was not in the figure.
    a2 = a.twinx()
    a2.plot(bb, [q["adequacy"] for q in r], "--", lw=1.2, color="0.35", marker="x", ms=4)
    a2.set_ylabel(r"diagnostic $r_j$", color="0.35", fontsize=8)
    a2.tick_params(axis="y", labelcolor="0.35", labelsize=7)
    a2.set_ylim(0, max(q["adequacy"] for q in r) * 1.35)
    a2.grid(False)
    savefig(fig, "fig6_real.pdf")


def fig7_lrv_rate():
    if not _ok("exp8_lrv_rate.json"):
        return
    d = common.load("exp8_lrv_rate.json")
    res = d["results"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # (a) head-to-head on the SAME object: the sandwich Sigma = H^-1 S H^-1
    a = ax[0]
    if "log" in res and res["log"]["rows"]:
        r = res["log"]["rows"]; n = [q["N"] for q in r]
        e = res["log"].get("exponent_sandwich")
        a.errorbar(n, [q["err_sandwich"] for q in r],
                   yerr=[1.96 * q["err_sandwich_se"] for q in r], marker="s", ms=4.5,
                   color=C["ours"], capsize=2, lw=1.4,
                   label=(r"ours: $\bar H^{-1}\widehat S^{\mathrm{FT}}\bar H^{-1}$, "
                          r"$b\asymp\log N$" + (f" (slope {e:.2f})" if e else "")))
        e2 = res["log"].get("exponent_iterate_obm")
        a.errorbar(n, [q["err_iterate_obm"] for q in r],
                   yerr=[1.96 * q["err_iterate_obm_se"] for q in r], marker="o", ms=4.5,
                   color=C["plugin"], mfc="white", mew=1.3, capsize=2, lw=1.4,
                   label=(r"iterate-path OBM, $b_n\asymp T^{3/4}$"
                          + (f" (slope {e2:.2f})" if e2 else "")))
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("stream length $N$")
    a.set_ylabel(r"$\|\widehat\Sigma-\Sigma\|_2/\|\Sigma\|_2$")
    a.set_title("(a) The same object, two ways:\nthe sandwich an interval actually uses",
                loc="left")
    # both curves are flat-ish, one high one low, so the middle of the panel is the only place a
    # legend does not sit on top of data
    a.legend(loc="center left", fontsize=6.6, framealpha=0.95, borderpad=0.3,
             labelspacing=0.24, handlelength=1.8)

    # (b) the long-run covariance itself, under each estimator's own optimal schedule
    a = ax[1]
    style = {"log": (C["ours"], "s", "ft", r"two-scale, $b\asymp\log N$"),
             "cbrt": (C["bm"], "v", "bm", r"batch means, $b\asymp N^{1/3}$")}
    for name, (col, mk, key, lab) in style.items():
        if name not in res or not res[name]["rows"]:
            continue
        r = res[name]["rows"]; n = [q["N"] for q in r]
        e = res[name].get("exponent_ft" if key == "ft" else "exponent_bm")
        a.errorbar(n, [q[f"err_{key}"] for q in r],
                   yerr=[1.96 * q[f"err_{key}_se"] for q in r], marker=mk, ms=4.5,
                   color=col, capsize=2, lw=1.4,
                   label=lab + (f" (slope {e:.2f})" if e is not None else ""))
    if "log" in res and res["log"]["rows"]:
        ns = [q["N"] for q in res["log"]["rows"]]
        nn = np.array([min(ns), max(ns)], float)
        ref = res["log"]["rows"][0]["err_ft"] * (nn / nn[0]) ** -0.5
        a.plot(nn, ref, ls=":", color="k", lw=1, label=r"slope $-1/2$")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("stream length $N$")
    a.set_ylabel(r"$\|\widehat S-S\|_2/\|S\|_2$")
    a.set_title("(b) The long-run covariance itself,\neach at its own optimal $b$", loc="left")
    a.legend(loc="lower left", fontsize=6.6, framealpha=0.95, borderpad=0.3,
             labelspacing=0.24, handlelength=1.8, bbox_to_anchor=(-0.02, -0.03))
    savefig(fig, "fig7_lrv_rate.pdf")



def fig8_hard_design():
    """Coverage against N on the two designs whose point estimate has not converged.

    The point of the figure is the *gap* between our curve and the oracle-variance curve: where
    it is invisible the undercoverage belongs to the central limit theorem and not to our
    covariance estimator, and where it is not we own it.
    """
    if not _ok("exp10_hard_design.json"):
        return
    d = common.load("exp10_hard_design.json")
    designs = [(k, v) for k, v in d.items() if isinstance(v, dict)]
    fig, ax = plt.subplots(1, len(designs) + 1, figsize=(2.5 * (len(designs) + 1), 2.9))
    for k, (label, dd) in enumerate(designs):
        a = ax[k]
        r = dd["rows"]
        N = [q["N"] for q in r]
        a.axhline(0.95, color="k", lw=0.8, ls=":", label="nominal")
        a.errorbar(N, [q["cov_bgsn"] for q in r],
                   yerr=[1.96 * q["cov_bgsn_se"] for q in r], marker="o", ms=5,
                   color=C["ours"], capsize=2, lw=1.5, label="ours (BGSN)")
        a.plot(N, [q["cov_oracle"] for q in r], marker="s", ms=4, mfc="none",
               color=C["oracle"], lw=1.2, ls="--", label="oracle variance (infeasible)")
        a.plot(N, [q["cov_plugin"] for q in r], marker="v", ms=4, color=C["plugin"], lw=1.5,
               label="dependence-blind")
        a.set_xscale("log"); a.set_xlabel("stream length $N$")
        _logticks(a, N)
        if k == 0:
            a.set_ylabel("coverage")
            a.legend(fontsize=6.2, loc="lower right", framealpha=0.95, frameon=True,
                     borderpad=0.3, labelspacing=0.22, handlelength=1.5)
        a.set_ylim(0.0, 1.06)
        a.set_title(f"({chr(97 + k)}) {label}", fontsize=9, loc="left")

    a = ax[-1]
    a.axhline(1.0, color="k", lw=0.8, ls=":", label="efficient")
    for (label, dd), mk in zip(designs, ("o", "^")):
        r = dd["rows"]
        a.plot([q["N"] for q in r], [q["rmse_rel_median"] for q in r], marker=mk, ms=5,
               lw=1.5, label=label)
    a.set_xscale("log"); a.set_xlabel("stream length $N$")
    _logticks(a, [q["N"] for q in designs[0][1]["rows"]])
    a.set_ylabel("RMSE / efficient s.e.")
    a.set_title(f"({chr(97 + len(designs))}) why: the point estimate", fontsize=9, loc="left")
    a.legend(fontsize=6.2, loc="upper right", framealpha=0.95, borderpad=0.3,
             labelspacing=0.22, handlelength=1.5)

    savefig(fig, "fig8_hard_design.pdf")


if __name__ == "__main__":
    fig1_headline(); fig2_lrv(); fig3_gapping(); fig4_cost(); fig5_ablations(); fig6_real()
    fig7_lrv_rate(); fig8_hard_design()
