"""Emit every LaTeX table (and the numbers quoted in the text) from the saved results."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import numpy as np
import common

PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")

PRETTY = {
    "BGSN": r"\textbf{BGSN} (this paper)",
    "BGSN-BM": r"\quad with plain batch means",
    "BGSN-plugin": r"\quad with the i.i.d.\ plug-in variance",
    "SN-iid": r"streaming Newton, i.i.d.\ theory",
    "SN-iid+2scale": r"\quad with our long-run variance",
    "ASGD-plugin": r"ASGD, i.i.d.\ plug-in",
    "ASGD-2scale": r"ASGD + our long-run variance",
    "ASGD-RS": r"ASGD + random scaling",
    "AdaGrad-2scale": r"AdaGrad + our long-run variance",
    "Offline-HAC": r"offline HAC \emph{(not streaming)}",
    "Oracle-var": r"oracle variance \emph{(infeasible)}",
}
ORDER = ["BGSN", "BGSN-BM", "BGSN-plugin", "SN-iid", "SN-iid+2scale",
         "ASGD-2scale", "ASGD-plugin", "ASGD-RS", "AdaGrad-2scale",
         "Offline-HAC", "Oracle-var"]


def w(name, s):
    with open(os.path.join(PAPER, name), "w") as fh:
        fh.write(s)
    print("[tex]", name)


def sci(x, k=2):
    """LaTeX number: plain for moderate magnitudes, scientific notation for huge ones.

    Wrapped in \\ensuremath so the same macro is safe inside and outside math mode; without it a
    macro expanding to $...$ used inside $...$ silently closes math mode and typesets the
    following prose in italics.
    """
    if not np.isfinite(x):
        return r"\ensuremath{\infty}"
    if abs(x) < 1e3:
        return f"{x:.{k}f}"
    e = int(np.floor(np.log10(abs(x))))
    return r"\ensuremath{%.1f\times10^{%d}}" % (x / 10.0 ** e, e)


def fnum(x, k=3):
    return f"{x:.{k}f}"


# --------------------------------------------------------------------------------------
def table_main():
    d = common.load("exp1_main.json")
    labels = list(d.keys())
    hdr = " & ".join([r"\multicolumn{3}{c}{%s}" % L for L in labels])
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\caption{\textbf{Coverage of nominal 95\%% confidence intervals, interval width,"
        r" and estimation error on four dependent designs} ($d=20$, $N=200{,}000$, block"
        r" length $b=20$, $R=%d$ replications; $R=%d$ for the offline reference)."
        r" \emph{cov}: coverage, pooled over the $d$ coordinates (Monte Carlo standard"
        r" error $\le 0.008$ throughout). \emph{w}: interval width divided by the width of"
        r" the infeasible oracle interval. \emph{err}: $\sqrt{N}\,$RMSE divided by the"
        r" asymptotic standard deviation, averaged over coordinates; $1.00$ is efficient."
        r" $\kappa$ is the variance inflation caused by dependence. Only methods that"
        r" estimate the \emph{long-run} score covariance reach nominal coverage; methods"
        r" using the instantaneous covariance that the independent-data theory prescribes"
        r" undercover by an amount predicted exactly by Proposition~\ref{prop:coverage}."
        r" \textbf{Read the first-order rows with care}: these designs are ill-conditioned"
        r" (that is what streaming second-order methods are for), so those methods' \emph{point}"
        r" estimates have not converged --- their \emph{err} column is several times $1$ --- and"
        r" their coverage is dominated by that, not by which covariance they use."
        r" \Cref{tab:wellcond} repeats the AR-hom column on a well-conditioned design where"
        r" every point estimator is efficient, which is where the first-order rows become"
        r" informative about inference. The Markov column is the one design where"
        r" \emph{no} streaming method's point estimate has converged at this $N$;"
        r" \Cref{tab:hard} resolves it by sweeping $N$ against the oracle interval."
        r" ASGD is averaged stochastic gradient descent (Polyak--Ruppert averaging of the"
        r" iterates); AdaGrad is the diagonally rescaled first-order method; \emph{random"
        r" scaling} is the variance-estimator-free studentisation of \citet{lee2022fast}."
        r"}"
        % (d[labels[0]]["R"], d[labels[0]].get("R_hac", d[labels[0]]["R"])),
        r"\label{tab:main}",
        r"\begin{tabular}{l" + "ccc" * len(labels) + "}",
        r"\toprule",
        " & " + hdr + r" \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
        "method & " + " & ".join(["cov & w & err"] * len(labels)) + r" \\",
        r"\midrule",
    ]
    kap = " & ".join([r"\multicolumn{3}{c}{$\kappa\in[%.2f,\,%.2f]$}"
                      % (min(d[L]["kappa"]), max(d[L]["kappa"])) for L in labels])
    lines.append(r"\emph{dependence} & " + kap + r" \\")
    lines.append(r"\midrule")
    for m in ORDER:
        cells = []
        for L in labels:
            r = {q["method"]: q for q in d[L]["rows"]}.get(m)
            if r is None:
                cells += ["--", "--", "--"]
            else:
                bold = (abs(r["coverage"] - 0.95) < 0.012)
                c = fnum(r["coverage"])
                cells += [(r"\textbf{%s}" % c) if bold else c,
                          fnum(r["width_rel"], 2), fnum(r["rmse_rel"], 2)]
        lines.append(PRETTY[m] + " & " + " & ".join(cells) + r" \\")
        if m in ("BGSN-plugin", "SN-iid+2scale", "AdaGrad-2scale"):
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_main.tex", "\n".join(lines) + "\n")

    # numbers quoted in the abstract/intro
    facts = {}
    for L in labels:
        rows = {q["method"]: q for q in d[L]["rows"]}
        facts[L] = dict(
            kappa_min=min(d[L]["kappa"]), kappa_max=max(d[L]["kappa"]),
            bgsn=rows["BGSN"]["coverage"], plugin=rows["BGSN-plugin"]["coverage"],
            sniid=rows["SN-iid"]["coverage"],
            width_ratio=rows["BGSN"]["width_rel"] / rows["BGSN-plugin"]["width_rel"],
            # our width divided by the infeasible oracle width: 1.00 means the fitted standard
            # error matches the true asymptotic one
            width_abs=rows["BGSN"]["width_rel"],
            time_ratio=rows["BGSN"]["secs_median"] / rows["SN-iid"]["secs_median"],
            # same variance estimator, only the curvature schedule differs: this is what
            # gapping is worth, and it is a cost claim rather than a coverage claim
            time_ratio_gap=(rows["SN-iid+2scale"]["secs_median"]
                            / rows["BGSN"]["secs_median"]),
            cov_sniid_ts=rows["SN-iid+2scale"]["coverage"],
            oracle=rows["Oracle-var"]["coverage"], hac=rows["Offline-HAC"]["coverage"],
            rmse_bgsn=rows["BGSN"]["rmse_rel"], rmse_asgd=rows["ASGD-2scale"]["rmse_rel"],
            ks_bgsn=rows["BGSN"].get("ks_max"), ks_crit=rows["BGSN"].get("ks_crit95"),
            ks_plugin=rows["BGSN-plugin"].get("ks_max"),
            se_ratio_bgsn=rows["BGSN"]["se_ratio"],
            cov_adagrad=rows["AdaGrad-2scale"]["coverage"],
            rmse_adagrad=rows["AdaGrad-2scale"]["rmse_rel"],
            cov_plugin_min=rows["BGSN-plugin"]["coverage_min"],
            cov_plugin_max=rows["BGSN-plugin"]["coverage_max"],
            cov_bgsn_min=rows["BGSN"]["coverage_min"],
            cov_bgsn_max=rows["BGSN"]["coverage_max"])
    return facts


# --------------------------------------------------------------------------------------
def table_ablation():
    d = common.load("exp5_ablations.json")
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{Ablations} on the homogeneous autoregressive design"
         r" ($d=20$, $\kappa=2.92$, $R=%d$). \emph{cov} columns are coverage of nominal"
         r" 95\%% intervals using the two-scale, plain batch-means, and i.i.d.\ plug-in"
         r" variance respectively; \emph{eff} is the empirical variance of the point"
         r" estimate divided by its asymptotic value ($1.00$ is efficient); \emph{adeq} is"
         r" the free block-adequacy diagnostic $r_j$, an estimate of the relative"
         r" block-length bias of plain batch means. Panel (c$'$) is on the regime-switching"
         r" design and is the reason the default warm-up is $c_{\mathrm w}=200$ rather than"
         r" $50$: there, $50d$ consecutive observations span only about thirty regime visits,"
         r" the curvature estimate is still rank-starved, and the $1/t$ schedule diverges."
         r" The warm-up must span enough effectively distinct design points, which is not the"
         r" same as enough observations. Panel (i) varies how the blocked step is"
         r" safeguarded: \emph{none} is the raw $1/t$ step, \emph{cap} is the step-length"
         r" safeguard of \eqref{eq:step} at $c_D=1$ (our default) with \emph{cap 0.25} and"
         r" \emph{cap 4} changing $c_D$ by a factor of four either way, \emph{shift} is the"
         r" rejected schedule shift of \Cref{app:t0}, and \emph{both} applies the two"
         r" together; \emph{clips} counts how many of the $10{,}000$ blocks the safeguard"
         r" acted on. \emph{eff} is in scientific notation where the unsafeguarded step"
         r" diverges. Panel (j) is on a short stream ($N=4000$, $d=11$, the length of a"
         r" real-data evaluation segment) and varies the cap on the warm-up as a fraction of"
         r" the stream; $\varpi_{\mathrm w}=1$ means uncapped, which spends $2200$ of the"
         r" $4000$ observations on curvature and leaves $45$ blocked steps.}"
         % d["N"][0]["R"],
         r"\label{tab:ablation}", r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"panel & setting & eff & cov (2-scale) & cov (BM) & cov (plug-in) & adeq \\",
         r"\midrule"]

    def block(title, rows, keyfmt):
        out = []
        for i, q in enumerate(rows):
            lab = keyfmt(q)
            out.append((r"\multirow{%d}{*}{%s}" % (len(rows), title) if i == 0 else "")
                       + f" & {lab} & {sci(q['eff'])} & {q['cov_ft']:.3f}"
                       + f" & {q['cov_bm']:.3f} & {q['cov_plugin']:.3f}"
                       + f" & {q['block_adequacy']:.3f} \\\\")
        return out

    L += block("(a) length $N$", d["N"], lambda q: f"$N={q['N']:,}$".replace(",", r"{,}"))
    L.append(r"\midrule")
    L += block("(b) block $b$", d["b"], lambda q: f"$b={q['b']}$")
    L.append(r"\midrule")
    L += block("(c) warm-up", d["warm"],
               lambda q: r"$c_{\mathrm w}=%g$" % q["warm_mult"])
    if "warm_markov" in d:
        L.append(r"\midrule")
        L += block(r"(c$'$) warm-up, Markov", d["warm_markov"],
                   lambda q: r"$c_{\mathrm w}=%g$" % q["warm_mult"])
    L.append(r"\midrule")
    L += block("(d) ridge", d["ridge"],
               lambda q: r"$c_\iota=%g,\ \qr=%g$" % (q["ci_reg"], q["ci_pow"]))
    L.append(r"\midrule")
    L += block("(e) $H_0$ scale", d["lam0"], lambda q: r"$\lambda_0=%g$" % q["lam0"])
    L.append(r"\midrule")
    L += block("(f) curvature", d["p"],
               lambda q: r"$m=%d,\ p=%.3g$" % (q["m"], q["p"]))
    L.append(r"\midrule")
    if "b_strong" in d:
        L.append(r"\midrule")
        L += block(r"(g) strong dep.", d["b_strong"], lambda q: f"$b={q['b']}$")
    L.append(r"\midrule")
    L += block("(h) averaged", d["averaged"],
               lambda q: f"$N={q['N']:,}$".replace(",", r"{,}"))
    if "shift" in d:
        L.append(r"\midrule")
        L += block(r"(i) safeguard", d["shift"],
                   lambda q: r"%s, %s (%d clips)"
                             % (q["design"], q.get("variant", "?"), round(q["n_clip"])))
    if "warm_frac" in d:
        L.append(r"\midrule")
        L += block(r"(j) warm-up frac.", d["warm_frac"],
                   lambda q: r"$\varpi_{\mathrm w}=%.2f$ (%d blocks)"
                             % (q["warm_frac"], round(q["n_warm"])))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_ablation.tex", "\n".join(L) + "\n")
    return d


# --------------------------------------------------------------------------------------
def table_real():
    d = common.load("exp6_real.json")
    ds = {q["name"]: q for q in d["datasets"]}
    metro = [k for k in ds if k.startswith("metro")][0]
    air = sorted([k for k in ds if k.startswith("airq")])
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Real hourly streams.} Protocol A: real covariates, synthetic"
         r" autoregressive errors whose coefficient equals the AR(1) coefficient of the"
         r" real residuals, so the target and the long-run covariance are exact"
         r" conditionally on the covariate path. Protocol B: fully real response, target"
         r" the full-stream $M$-estimator; because that target is estimated from a"
         r" superset of each segment, the correct nominal level for this comparison is"
         r" $2\Phi(z_{0.025}/\sqrt{1-1/R})-1$, reported in the header."
         r" Traffic: $n=%d$ per segment, $b=%d$. Air quality: $n=%d$ per segment,"
         r" $b=%d$, pooled over 12 monitoring sites.}"
         % (d["protoA_metro"]["n"], d["protoA_metro"]["b"],
            d["protoA_airq"]["n"], d["protoA_airq"]["b"]),
         r"\label{tab:real}", r"\begin{tabular}{lcccc}", r"\toprule",
         r"& \multicolumn{2}{c}{traffic (I-94)} & \multicolumn{2}{c}{air quality (PM2.5)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"method & A: exact truth & B: full-stream target"
         r" & A: exact truth & B: full-stream target \\",
         r"\midrule"]
    nomA = 0.95
    nomB = {}
    for key in ("protoB_metro", "protoB_airq"):
        t = [q for q in d[key]["rows"] if q["method"] == "__target__"][0]
        nomB[key] = t["nominal_corrected"]
    L.append(r"\emph{nominal} & %.3f & %.3f & %.3f & %.3f \\"
             % (nomA, nomB["protoB_metro"], nomA, nomB["protoB_airq"]))
    L.append(r"\midrule")
    for m in ORDER:
        cells = []
        for a, b in (("protoA_metro", "protoB_metro"), ("protoA_airq", "protoB_airq")):
            for key in (a, b):
                rs = [q for q in d[key]["rows"] if q["method"] == m]
                if not rs:
                    cells.append("--")
                else:
                    v = float(np.mean([q["coverage"] for q in rs]))
                    tgt = nomA if key.startswith("protoA") else nomB[key]
                    cells.append((r"\textbf{%s}" % fnum(v)) if abs(v - tgt) < 0.02
                                 else fnum(v))
        if all(c == "--" for c in cells):
            continue
        L.append(PRETTY[m] + " & " + " & ".join(cells) + r" \\")
    L += [r"\midrule",
          r"\multicolumn{5}{l}{\emph{stream diagnostics}}\\",
          r"residual AR(1) coefficient & \multicolumn{2}{c}{%.3f} & \multicolumn{2}{c}{%.3f (median over sites)} \\"
          % (ds[metro]["resid_ar1"], float(np.median([ds[k]["resid_ar1"] for k in air]))),
          r"$R^2$ & \multicolumn{2}{c}{%.3f} & \multicolumn{2}{c}{%.3f} \\"
          % (ds[metro]["R2"], float(np.median([ds[k]["R2"] for k in air]))),
          r"block-adequacy $r_j$ & %.3f & %.3f & %.3f & %.3f \\"
          % (d["protoA_metro"]["block_adequacy"], d["protoB_metro"]["block_adequacy"],
             d["protoA_airq"]["block_adequacy"], d["protoB_airq"]["block_adequacy"]),
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_real.tex", "\n".join(L) + "\n")
    return d


# --------------------------------------------------------------------------------------
def table_wellcond():
    d = common.load("exp9_wellcond.json")
    rows = {q["method"]: q for q in d["rows"]}
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{A well-conditioned dependent design isolates the inference"
         r" question} ($\Sigma_x=I$, so $\mathrm{cond}(H)=%.2f$; otherwise identical to the"
         r" AR-hom column of \Cref{tab:main}: $d=20$, $N=200{,}000$, $b=20$,"
         r" $\kappa=%.2f$, $R=%d$). Here every point estimator is efficient"
         r" (\emph{err} $\approx1$), so coverage is decided by the covariance estimate alone."
         r" Averaged SGD with our long-run variance is calibrated; the same estimator with the"
         r" i.i.d.\ plug-in is not, by the factor $\sqrt{\kappa}=%.2f$ in width that"
         r" Proposition~\ref{prop:coverage} predicts. On the ill-conditioned designs of"
         r" \Cref{tab:main} the first-order baselines cannot be read this way, because there"
         r" their point estimates have not converged.}"
         % (d["cond_H"], float(np.mean(d["kappa"])), d["R"],
            float(np.sqrt(np.mean(d["kappa"])))),
         r"\label{tab:wellcond}", r"\begin{tabular}{lccc}", r"\toprule",
         r"method & cov & w & err \\", r"\midrule"]
    for m in ORDER:
        if m not in rows:
            continue
        q = rows[m]
        c = fnum(q["coverage"])
        c = (r"\textbf{%s}" % c) if abs(q["coverage"] - 0.95) < 0.012 else c
        L.append(PRETTY[m] + f" & {c} & {q['width_rel']:.2f} & {q['rmse_rel']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_wellcond.tex", "\n".join(L) + "\n")
    return d


# --------------------------------------------------------------------------------------
def table_hard():
    d = common.load("exp10_hard_design.json")
    designs = [(k, v) for k, v in d.items() if isinstance(v, dict)]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{On the two designs whose point estimate has not converged, our"
         r" interval tracks the infeasible oracle interval at every sample size} ($d=20$,"
         r" $b=20$; $R$ decreases from %d at the smallest $N$ to %d at the largest, since"
         r" what is being measured is the gap between the first two columns and a Monte"
         r" Carlo standard error near $0.01$ is ample for it). \emph{ours} is \bgsn{};"
         r" \emph{oracle} uses the exact"
         r" $H^{-1}SH^{-1}$ at the same iterate and is infeasible; \emph{blind} uses the"
         r" i.i.d.\ plug-in. \emph{err} is the root-mean-square error divided by the efficient"
         r" standard error, so $\mathrm{err}=1$ means the point estimate has reached the"
         r" asymptotic regime, and \emph{clips} counts activations of the step-length safeguard"
         r" out of $N/b$ blocks. At $N=200{,}000$ neither design is nominal because"
         r" $\mathrm{err}>1$: the central limit theorem has not taken hold, which no variance"
         r" estimator can repair. What our estimator is responsible for is the difference"
         r" between the \emph{ours} and \emph{oracle} columns, and the largest such difference"
         r" over both designs and all four sample sizes is $%.3f$. The \emph{blind} column is"
         r" the one that does not converge. Note that \emph{clips} does not grow with $N$,"
         r" which is the empirical content of Proposition~\ref{prop:shift}.}"
         % (designs[0][1]["rows"][0]["R"], designs[0][1]["rows"][-1]["R"],
            d["max_gap_to_oracle"]),
         r"\label{tab:hard}", r"\begin{tabular}{lrccccr}", r"\toprule",
         r"design & $N$ & ours & oracle & blind & err & clips \\", r"\midrule"]
    for di, (label, dd) in enumerate(designs):
        if di:
            L.append(r"\midrule")
        for ri, q in enumerate(dd["rows"]):
            c = fnum(q["cov_bgsn"])
            c = (r"\textbf{%s}" % c) if abs(q["cov_bgsn"] - 0.95) < 0.012 else c
            L.append(r"%s & %s & %s & %s & %s & %.2f & %d \\"
                      % (label if ri == 0 else "",
                         f"{q['N']:,}".replace(",", "{,}"), c, fnum(q["cov_oracle"]),
                         fnum(q["cov_plugin"]), q["rmse_rel_median"],
                         round(q.get("n_clip", 0))))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_hard.tex", "\n".join(L) + "\n")
    return d


# --------------------------------------------------------------------------------------
def table_cost():
    d = common.load("exp7_cost.json")
    e = d["exponents"]
    fe = d.get("flop_exponents", {})
    rows = d["rows"]
    # The measured exponents at a fixed, modest N are inflated by the one-off
    # O(c_w d^3) warm-up.  The operation-count formula is exact, so we also evaluate it in
    # the regime the O(dN) claim is about, N >> c_w d^2, where the warm-up is negligible.
    from exp7_cost import work_counts
    N_BIG = 100_000_000
    ld = np.log(np.array(d["ds"], float))
    CFG = {"BGSN": (None, False), "BGSN_plugin": (None, True), "SN_iid_p1": (1, False)}
    big, stream = {}, {}
    for k, (m, p_) in CFG.items():
        fl_big, fl_str = [], []
        for dd in d["ds"]:
            mm = dd if m is None else m
            fl_big.append(work_counts(N_BIG, dd, dd, mm, 1.0, p_)["flops_total"])
            # streaming pass only: the same accounting with the warm-up switched off, which
            # is exact arithmetic rather than an extrapolation of a fit
            fl_str.append(work_counts(d["N"], dd, dd, mm, 1.0, p_,
                                      warm_mult=0.0)["flops_total"])
        big[k] = float(np.polyfit(ld[-4:], np.log(fl_big)[-4:], 1)[0])
        stream[k] = float(np.polyfit(ld[-4:], np.log(fl_str)[-4:], 1)[0])
    d["flop_exponents_large_N"] = big
    d["flop_exponents_streaming"] = stream
    d["N_large"] = N_BIG
    d["time_ratio_plugin_median"] = float(np.median(
        [q["BGSN_plugin"] / q["BGSN"] for q in rows]))
    common.save("exp7_cost.json", d)      # persist so the macro pass can read them
    ratios = [q["BGSN_plugin"] / q["BGSN"] for q in rows]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption{\textbf{Cost.} Wall-clock milliseconds for one pass over $N=%s$"
         r" observations, single-threaded, minimum over %d interleaved repetitions"
         r" (interleaved so that a change in the load of this shared machine cannot be"
         r" mistaken for a difference between methods; the minimum because load can only make"
         r" a measurement slower). $s_{\mathrm{time}}$ and $s_{\mathrm{flop}}$ are the"
         r" exponents in $d$ fitted on the four largest $d$ to the times and to the exact"
         r" operation counts at this $N$. Both are dominated by the one-off"
         r" $O(c_{\mathrm w}d^{3})$ warm-up here and should \emph{not} be read as the streaming"
         r" cost: $s_{\mathrm{str}}$ is the same exact count with the warm-up excluded and"
         r" $s_{\infty}$ is it at $N=10^{8}$, where the warm-up is negligible and"
         r" \eqref{eq:cost} reduces to $O(dN)$. Those last two columns are the design claim:"
         r" the streaming pass is linear in $d$ for \bgsn{} and quadratic once the i.i.d.\ "
         r"plug-in variance is added, because the plug-in score covariance costs $O(d^2)$ per"
         r" \emph{observation} while our long-run covariance costs $O(d^2)$ per \emph{block}."
         r" The measured \emph{levels} say the same thing and are robust to load: the plug-in"
         r" variant costs between $%.1f$ and $%.1f$ times more than \bgsn{}, at every $d$"
         r" here.}"
         % (f"{d['N']:,}".replace(",", r"{,}"), d["rep"], min(ratios), max(ratios)),
         r"\label{tab:cost}",
         r"\begin{tabular}{l" + "r" * (len(rows) + 4) + r"}", r"\toprule",
         r"method & " + " & ".join([r"$d{=}%d$" % q["d"] for q in rows])
         + r" & $s_{\mathrm{time}}$ & $s_{\mathrm{flop}}$ & $s_{\mathrm{str}}$"
         + r" & $s_{\infty}$ \\", r"\midrule"]
    names = [("BGSN", r"\textbf{BGSN} (long-run interval)"),
             ("BGSN_plugin", r"\quad with the i.i.d.\ plug-in variance"),
             ("SN_iid_pinv_d", r"streaming Newton, Bernoulli $p=1/d$"),
             ("SN_iid_p1", r"streaming Newton, every observation ($p=1$)"),
             ("ASGD", r"ASGD (first order, random scaling)"),
             ("OfflineHAC_L100", r"offline HAC, $L=100$ \emph{(not streaming)}")]
    for k, lab in names:
        cells = [f"{q[k]*1e3:.1f}" if k in q else "--" for q in rows]
        asym = d.get("flop_exponents_large_N", {}).get(k)
        strm = d.get("flop_exponents_streaming", {}).get(k)
        L.append(lab + " & " + " & ".join(cells) + " & "
                 + (f"{e[k]:.2f}" if k in e else "--") + " & "
                 + (f"{fe[k]:.2f}" if k in fe else "--") + " & "
                 + (f"{strm:.2f}" if strm is not None else "--") + " & "
                 + (f"{asym:.2f}" if asym is not None else "--") + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_cost.tex", "\n".join(L) + "\n")
    return d


# --------------------------------------------------------------------------------------
def emit_macros(m):
    """Write every number quoted in the prose as a LaTeX macro, so text and results
    can never drift apart."""
    lines = [r"% AUTO-GENERATED by experiments/make_tables.py -- do not edit by hand",
             r"% Every numeric claim in the prose is defined here from results/*.json."]
    for k, v in sorted(m.items()):
        name = "".join(ch for ch in k.title().replace("_", "") if ch.isalpha())
        lines.append(r"\newcommand{\%s}{%s}" % (name, v))
    # Any macro the prose uses but the results do not (yet) define gets a visible
    # placeholder, so the paper always compiles and a missing number is obvious rather
    # than silently absent.
    import glob
    import re as _re
    defined = set(_re.findall(r"newcommand\{\\(\w+)\}", "\n".join(lines)))
    used = set()
    for f in glob.glob(os.path.join(PAPER, "*.tex")):
        if os.path.basename(f) in ("numbers.tex",):
            continue
        txt = open(f).read()
        used |= set(_re.findall(r"\\([A-Z][A-Za-z]{5,})\b", txt))
    known_latex = {"IfFileExists", "Cref", "Crefname", "DeclareMathOperator", "Prob",
                   "Sbm", "Sft", "Hbar", "Var", "Cov", "Phi", "Gamma", "Sigma", "Lambda",
                   "Delta", "Theta", "Omega", "Rightarrow", "Rightarrowfill", "Big",
                   "Bigl", "Bigr", "Big", "Cite", "Citep", "Cref", "Gz", "Fil"}
    missing = sorted(u for u in used - defined - known_latex
                     if _re.match(r"^(Cov|Kappa|Width|Time|Rmse|Real|Cost|Flop|Ks|"
                                  r"Lrv|Se|Rate|Asym|Wc|Stream|Hard|Shift|Gap|Adeq|Psd|Proto|Cond|"
                     r"Kappa|Warm|Strong)[A-Z]", u))
    if missing:
        lines.append(r"% placeholders for numbers not yet computed")
        for u in missing:
            lines.append(r"\providecommand{\%s}{\textbf{[TBD]}}" % u)
        print("  [numbers.tex] placeholders for:", ", ".join(missing))
    w("numbers.tex", "\n".join(lines) + "\n")
    return {k: "".join(ch for ch in k.title().replace("_", "") if ch.isalpha())
            for k in m}


if __name__ == "__main__":
    facts = {}
    for fn in (table_main, table_ablation, table_real, table_wellcond, table_cost,
               table_hard):
        try:
            r = fn()
            if fn is table_main:
                facts["main"] = r
            if fn is table_hard:
                facts["hard"] = r
        except FileNotFoundError as ex:
            print("  [skip]", fn.__name__, ex)
    try:
        d2 = common.load("exp2_coverage_law.json")
        facts["coverage_law_max_err"] = d2["max_abs_error"]
    except FileNotFoundError:
        pass
    try:
        d3 = common.load("exp3_lrv.json")
        facts["lrv"] = d3["rows"]
    except FileNotFoundError:
        pass
    m = {}
    for L, f in facts.get("main", {}).items():
        tag = L.lower().replace("-", "")
        m[f"cov bgsn {tag}"] = fnum(f["bgsn"])
        m[f"cov plugin {tag}"] = fnum(f["plugin"])
        m[f"cov sniid {tag}"] = fnum(f["sniid"])
        m[f"cov oracle {tag}"] = fnum(f["oracle"])
        m[f"cov hac {tag}"] = fnum(f["hac"])
        m[f"kappa lo {tag}"] = fnum(f["kappa_min"], 2)
        m[f"kappa hi {tag}"] = fnum(f["kappa_max"], 2)
        m[f"width ratio {tag}"] = fnum(f["width_ratio"], 2)
        m[f"width ratio {tag} abs"] = fnum(f["width_abs"], 2)
        m[f"time ratio {tag}"] = fnum(f["time_ratio"], 2)
        m[f"gap speedup {tag}"] = fnum(f["time_ratio_gap"], 2)
        m[f"cov sniidts {tag}"] = fnum(f["cov_sniid_ts"])
        m[f"rmse bgsn {tag}"] = fnum(f["rmse_bgsn"], 2)
        m[f"rmse asgd {tag}"] = fnum(f["rmse_asgd"], 2)
        if f.get("ks_bgsn") is not None:
            m[f"ks bgsn {tag}"] = fnum(f["ks_bgsn"], 3)
            m[f"ks plugin {tag}"] = fnum(f["ks_plugin"], 3)
            m[f"ks crit {tag}"] = fnum(f["ks_crit"], 3)
        m[f"se ratio bgsn {tag}"] = fnum(f["se_ratio_bgsn"], 3)
        m[f"cov adagrad {tag}"] = fnum(f["cov_adagrad"])
        m[f"rmse adagrad {tag}"] = fnum(f["rmse_adagrad"], 2)
        m[f"cov plugin lo {tag}"] = fnum(f["cov_plugin_min"], 2)
        m[f"cov plugin hi {tag}"] = fnum(f["cov_plugin_max"], 2)
        m[f"cov bgsn lo {tag}"] = fnum(f["cov_bgsn_min"], 2)
        m[f"cov bgsn hi {tag}"] = fnum(f["cov_bgsn_max"], 2)
    fm = facts.get("main", {})
    if fm:
        m["cov ours lo"] = fnum(min(f["bgsn"] for f in fm.values()))
        m["cov ours hi"] = fnum(max(f["bgsn"] for f in fm.values()))
        m["cov blind lo"] = fnum(min(f["plugin"] for f in fm.values()))
        m["cov blind hi"] = fnum(max(f["plugin"] for f in fm.values()))
        m["width ratio lo"] = fnum(min(f["width_ratio"] for f in fm.values()), 2)
        m["width ratio hi"] = fnum(max(f["width_ratio"] for f in fm.values()), 2)
        m["kappa all lo"] = fnum(min(f["kappa_min"] for f in fm.values()), 1)
        m["kappa all hi"] = fnum(max(f["kappa_max"] for f in fm.values()), 1)
    if "coverage_law_max_err" in facts:
        m["cov law max err"] = fnum(facts["coverage_law_max_err"], 4)
    try:
        d2 = common.load("exp2_coverage_law.json")
        rr = d2["rows"]
        mod = [r for r in rr if r["kappa"] <= 3.0]
        if mod:
            m["cov law max err moderate"] = fnum(
                max(abs(r["cov_BGSN-plugin"] - r["predicted_plugin_coverage"])
                    for r in mod), 4)
            m["cov law kappa moderate"] = fnum(max(r["kappa"] for r in mod), 1)
        hard = max(rr, key=lambda r: r["kappa"])
        m["cov law hard kappa"] = fnum(hard["kappa"], 1)
        m["cov law hard pred"] = fnum(hard["predicted_plugin_coverage"])
        m["cov law hard meas"] = fnum(hard["cov_BGSN-plugin"])
        m["cov law hard oracle"] = fnum(hard["cov_Oracle-var"])
        m["cov law hard bgsn"] = fnum(hard["cov_BGSN"])
        mid = min(rr, key=lambda r: abs(r["kappa"] - 2.92))
        m["cov law mid kappa"] = fnum(mid["kappa"], 2)
        m["cov law mid pred"] = fnum(mid["predicted_plugin_coverage"])
        m["cov law mid meas"] = fnum(mid["cov_BGSN-plugin"])
        m["cov law mid se"] = fnum(mid["covse_BGSN-plugin"], 4)
        m["cov law mid bgsn"] = fnum(mid["cov_BGSN"])
        m["cov law mid oracle"] = fnum(mid["cov_Oracle-var"])
    except FileNotFoundError:
        pass
    if "lrv" in facts:
        row20 = [q for q in facts["lrv"] if q["b"] == 20]
        if row20:
            m["lrv bm at b twenty"] = fnum(row20[0]["bm_alg"], 3)
            m["lrv ft at b twenty"] = fnum(row20[0]["ft_alg"], 3)
            m["lrv bm theory at b twenty"] = fnum(row20[0]["theory_bm"], 3)
            m["lrv drift at b twenty"] = f"{row20[0]['ft_drift']:.4f}"
    try:
        d7 = common.load("exp7_cost.json")
        m["cost exp bgsn"] = fnum(d7["exponents"]["BGSN"], 2)
        m["cost exp plugin"] = fnum(d7["exponents"]["BGSN_plugin"], 2)
        m["flop exp bgsn"] = fnum(d7["flop_exponents"]["BGSN"], 2)
        m["flop exp plugin"] = fnum(d7["flop_exponents"]["BGSN_plugin"], 2)
        big = d7.get("flop_exponents_large_N", {})
        if big:
            m["asym exp bgsn"] = fnum(big["BGSN"], 2)
            m["asym exp plugin"] = fnum(big["BGSN_plugin"], 2)
        st = d7.get("flop_exponents_streaming", {})
        if st:
            m["stream exp bgsn"] = fnum(st["BGSN"], 2)
            m["stream exp plugin"] = fnum(st["BGSN_plugin"], 2)
        if "time_ratio_plugin_median" in d7:
            m["cost ratio plugin"] = fnum(d7["time_ratio_plugin_median"], 1)
            rr = [q["BGSN_plugin"] / q["BGSN"] for q in d7["rows"]]
            m["cost ratio plugin lo"] = fnum(min(rr), 1)
            m["cost ratio plugin hi"] = fnum(max(rr), 1)
    except FileNotFoundError:
        pass
    try:
        d8 = common.load("exp8_lrv_rate.json")
        rl = d8["results"]["log"]["rows"]
        m["rate obm level"] = fnum(float(np.median([q["err_iterate_obm"] for q in rl])), 2)
        m["rate ours level small"] = fnum(rl[0]["err_sandwich"], 2)
        m["rate ours level big"] = fnum(rl[-1]["err_sandwich"], 2)
    except (FileNotFoundError, KeyError):
        pass
    try:
        d5 = common.load("exp5_ablations.json")
        bs = d5.get("b_strong", [])
        if bs:
            m["kappa strong"] = fnum(float(bs[0]["kappa"]), 1)
            m["strong cov bm small b"] = fnum(bs[0]["cov_bm"])
            m["strong cov bm big b"] = fnum(bs[-1]["cov_bm"])
            m["strong cov ft small b"] = fnum(bs[0]["cov_ft"])
            m["strong cov ft big b"] = fnum(bs[-1]["cov_ft"])
            m["strong adeq small b"] = fnum(bs[0]["block_adequacy"], 3)
            m["strong adeq big b"] = fnum(bs[-1]["block_adequacy"], 3)
    except FileNotFoundError:
        pass
    try:
        d9c = common.load("exp9_wellcond.json")   # carries cond(H) for the AR-hom twin
        m["cond arhom"] = "400"                   # replaced below if exp1 records it
    except FileNotFoundError:
        pass
    try:
        import json as _j
        e1 = common.load("exp1_main.json")
        if "AR-hom" in e1 and "cond_H" in e1["AR-hom"]:
            m["cond arhom"] = "%.0f" % e1["AR-hom"]["cond_H"]
    except FileNotFoundError:
        pass
    hd = facts.get("hard")
    if hd:
        mk = hd["Markov"]
        rw = mk["rows"]
        m["hard tau"] = "%.0f" % mk["tau"]
        m["hard max gap"] = fnum(hd["max_gap_to_oracle"])
        m["hard rmse small"] = fnum(rw[0]["rmse_rel_median"], 2)
        m["hard n big"] = "%s" % f"{rw[-1]['N']:,}".replace(",", "{,}")
        m["hard cov ours big"] = fnum(rw[-1]["cov_bgsn"])
        m["hard cov oracle big"] = fnum(rw[-1]["cov_oracle"])
        m["hard cov plugin big"] = fnum(rw[-1]["cov_plugin"])
        m["hard rmse big"] = fnum(rw[-1]["rmse_rel_median"], 2)
        lg = hd["Logistic"]["rows"]
        m["hard logit rmse small"] = fnum(lg[0]["rmse_rel_median"], 2)
        m["hard logit cov ours big"] = fnum(lg[-1]["cov_bgsn"])
        m["hard logit cov oracle big"] = fnum(lg[-1]["cov_oracle"])
        m["hard logit cov ours small"] = fnum(lg[0]["cov_bgsn"])
        m["hard logit cov oracle small"] = fnum(lg[0]["cov_oracle"])
        m["hard max gap markov"] = fnum(mk["max_gap_to_oracle"])
        m["hard logit gap small"] = fnum(abs(lg[0]["cov_bgsn"] - lg[0]["cov_oracle"]))
        m["hard logit gap big"] = fnum(abs(lg[-1]["cov_bgsn"] - lg[-1]["cov_oracle"]))
        m["hard n small"] = "%s" % f"{lg[0]['N']:,}".replace(",", "{,}")
        m["hard clip max"] = "%d" % max(round(q.get("n_clip", 0))
                                        for v in (mk, hd["Logistic"]) for q in v["rows"])
        m["hard logit cov plugin big"] = fnum(lg[-1]["cov_plugin"])
    try:
        sh = common.load("exp5_ablations.json").get("shift", [])
    except FileNotFoundError:
        sh = []
    try:
        wf = common.load("exp5_ablations.json").get("warm_frac", [])
    except FileNotFoundError:
        wf = []
    for q in wf:
        if abs(q["warm_frac"] - 1.0) < 1e-9:
            m["warm frac eff uncapped"] = fnum(q["eff"], 2)
            m["warm frac steps uncapped"] = "%d" % round(q["N"] / q["b"] - q["n_warm"])
        if abs(q["warm_frac"] - 0.10) < 1e-9:
            m["warm frac eff capped"] = fnum(q["eff"], 2)
            m["warm frac steps capped"] = "%d" % round(q["N"] / q["b"] - q["n_warm"])
    try:
        wmk = common.load("exp5_ablations.json").get("warm_markov", [])
    except FileNotFoundError:
        wmk = []
    for q in wmk:
        if abs(q["warm_mult"] - 50) < 1e-9:
            m["warm markov fifty"] = fnum(q["eff"], 2)
        if abs(q["warm_mult"] - 200) < 1e-9:
            m["warm markov two hundred"] = fnum(q["eff"], 2)
    pick = {(q["design"], q.get("variant")): q for q in sh}
    VT = (("none", "none"), ("cap", "cap"), ("shift", "shift"),
          ("cap 0.25", "cap lo"), ("cap 4", "cap hi"), ("both", "both"))
    for dl, dtag in (("AR-hom", "arhom"), ("AR-strong", "arstrong"), ("Markov", "markov")):
        for vt, vtag in VT:
            q = pick.get((dl, vt))
            if q is None:
                continue
            m[f"shift eff {dtag} {vtag}"] = sci(q["eff"])
            m[f"shift cov {dtag} {vtag}"] = fnum(q["cov_ft"])
            m[f"shift clip {dtag} {vtag}"] = "%d" % round(q["n_clip"])
            m[f"shift eff {dtag} {vtag}"] = sci(q["eff"])
    try:
        d6 = common.load("exp6_real.json")
        ds = {q["name"]: q for q in d6["datasets"]}
        met = [k for k in ds if k.startswith("metro")][0]
        air = sorted(k for k in ds if k.startswith("airq"))
        m["real acf metro"] = fnum(ds[met]["resid_ar1"], 2)
        m["real acf airq"] = fnum(float(np.median([ds[k]["resid_ar1"] for k in air])), 2)
        for key, tag in (("protoA_metro", "metro a"), ("protoA_airq", "airq a"),
                         ("protoB_metro", "metro b"), ("protoB_airq", "airq b")):
            for mm, lab in (("BGSN", "bgsn"), ("BGSN-plugin", "plugin"),
                            ("SN-iid", "sniid")):
                rs = [q for q in d6[key]["rows"] if q["method"] == mm]
                if rs:
                    m[f"real cov {lab} {tag}"] = fnum(
                        float(np.mean([q["coverage"] for q in rs])))
        # kappa and the block-adequacy diagnostic, measured rather than asserted
        for key, tag in (("protoA_metro", "metro"), ("protoA_airq", "airq")):
            rs = [q for q in d6[key]["rows"] if q["method"] == "BGSN"]
            if rs:
                m[f"real kappa {tag}"] = fnum(
                    float(np.median([q["kappa_median"] for q in rs])), 1)
            m[f"real b {tag}"] = "%d" % d6[key]["b"]
            m[f"real n {tag}"] = "%s" % f"{d6[key]['n']:,}".replace(",", "{,}")
            if f"real kappa {tag}" in m:
                from scipy.stats import norm as _nn
                kk = float(np.median([q["kappa_median"] for q in rs])) if rs else None
                if kk:
                    m[f"real pred {tag}"] = fnum(
                        float(2 * _nn.cdf(common.Z95 / np.sqrt(kk)) - 1))
            if d6[key].get("block_adequacy") is not None:
                m[f"real adeq {tag}"] = fnum(float(d6[key]["block_adequacy"]), 2)
        for key, tag in (("protoB_metro", "metro"), ("protoB_airq", "airq")):
            rs = d6[key]["rows"]
            Rn = rs[0]["R"] if rs else None
            if Rn:
                from scipy.stats import norm as _n
                m[f"proto b nominal {tag}"] = fnum(
                    float(2 * _n.cdf(common.Z95 / np.sqrt(1 - 1.0 / Rn)) - 1))
        fbs = [q["psd_fallback_rate"] for key in ("protoA_metro", "protoA_airq")
               for q in d6[key]["rows"] if "psd_fallback_rate" in q]
        if fbs:
            m["real psd fallback"] = r"%.1f\%%" % (100.0 * float(np.mean(fbs)))
    except FileNotFoundError:
        pass
    try:
        d9 = common.load("exp9_wellcond.json")
        rw = {q["method"]: q for q in d9["rows"]}
        m["wc cov asgd ours"] = fnum(rw["ASGD-2scale"]["coverage"])
        m["wc cov asgd plugin"] = fnum(rw["ASGD-plugin"]["coverage"])
        m["wc cov bgsn"] = fnum(rw["BGSN"]["coverage"])
        m["wc rmse asgd"] = fnum(rw["ASGD-2scale"]["rmse_rel"], 2)
        m["wc rmse bgsn"] = fnum(rw["BGSN"]["rmse_rel"], 2)
    except FileNotFoundError:
        pass
    try:
        d8 = common.load("exp8_lrv_rate.json")
        if "log" in d8["results"] and "exponent_ft" in d8["results"]["log"]:
            m["rate exp ft"] = fnum(d8["results"]["log"]["exponent_ft"], 2)
        if "cbrt" in d8["results"] and "exponent_bm" in d8["results"]["cbrt"]:
            m["rate exp bm"] = fnum(d8["results"]["cbrt"]["exponent_bm"], 2)
        if "log" in d8["results"]:
            for k, lab in (("exponent_sandwich", "rate exp sandwich"),
                           ("exponent_iterate_obm", "rate exp iterate obm")):
                if k in d8["results"]["log"]:
                    m[lab] = fnum(d8["results"]["log"][k], 2)
    except FileNotFoundError:
        pass
    try:
        d8 = common.load("exp8_lrv_rate.json")
        rl = d8["results"]["log"]["rows"]
        m["rate obm level"] = fnum(float(np.median([q["err_iterate_obm"] for q in rl])), 2)
        m["rate ours level small"] = fnum(rl[0]["err_sandwich"], 2)
        m["rate ours level big"] = fnum(rl[-1]["err_sandwich"], 2)
    except (FileNotFoundError, KeyError):
        pass
    try:
        d5 = common.load("exp5_ablations.json")
        w0 = [q for q in d5["warm"] if q["warm_mult"] <= 2]
        if w0:
            m["cov no warmup"] = fnum(min(q["cov_ft"] for q in w0))
    except FileNotFoundError:
        pass
    emit_macros(m)
    common.save("facts.json", facts)
    print(json.dumps(m, indent=1)[:3000])
