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
        r"\caption{\textbf{Coverage of nominal 95\% confidence intervals, interval width,"
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
        r" informative about inference."
        r"}",
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
            time_ratio=rows["BGSN"]["secs_median"] / rows["SN-iid"]["secs_median"],
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
         r" 95\% intervals using the two-scale, plain batch-means, and i.i.d.\ plug-in"
         r" variance respectively; \emph{eff} is the empirical variance of the point"
         r" estimate divided by its asymptotic value ($1.00$ is efficient); \emph{adeq} is"
         r" the free block-adequacy diagnostic $r_j$, an estimate of the relative"
         r" block-length bias of plain batch means. Panel (c$'$) is on the regime-switching"
         r" design and is the reason the default warm-up is $c_{\mathrm w}=200$ rather than"
         r" $50$: there, $50d$ consecutive observations span only about thirty regime visits,"
         r" the curvature estimate is still rank-starved, and the $1/t$ schedule diverges."
         r" The warm-up must span enough effectively distinct design points, which is not the"
         r" same as enough observations.}" % d["N"][0]["R"],
         r"\label{tab:ablation}", r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"panel & setting & eff & cov (2-scale) & cov (BM) & cov (plug-in) & adeq \\",
         r"\midrule"]

    def block(title, rows, keyfmt):
        out = []
        for i, q in enumerate(rows):
            lab = keyfmt(q)
            out.append((r"\multirow{%d}{*}{%s}" % (len(rows), title) if i == 0 else "")
                       + f" & {lab} & {q['eff']:.2f} & {q['cov_ft']:.3f}"
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
    big = {}
    for k, (m, p_) in (("BGSN", (None, False)), ("BGSN_plugin", (None, True)),
                       ("SN_iid_p1", (1, False))):
        fl = []
        for dd in d["ds"]:
            mm = dd if m is None else m
            pp = 1.0 if m is None else 1.0
            fl.append(work_counts(N_BIG, dd, dd, mm, pp, p_)["flops_total"])
        big[k] = float(np.polyfit(ld[-4:], np.log(fl)[-4:], 1)[0])
    d["flop_exponents_large_N"] = big
    d["N_large"] = N_BIG
    common.save("exp7_cost.json", d)      # persist so the macro pass can read them
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{Cost.} Wall-clock milliseconds for one pass over"
         r" $N=%s$ observations, single-threaded, median of %d runs, and the empirical"
         r" exponent $s$ in time $\propto d^{s}$ fitted on the four largest $d$."
         r" The i.i.d.\ plug-in score covariance that the independent-data theory"
         r" prescribes costs $O(d^2)$ per \emph{observation}, so the dependence-blind"
         r" interval is more expensive than our long-run interval, which costs $O(d^2)$"
         r" per \emph{block}.}" % (f"{d['N']:,}".replace(",", r"{,}"), d["rep"]),
         r"\ \emph{Asymptotic} exponents are the same exact counts evaluated at"
         r" $N=10^{8}$, where the one-off $O(c_{\mathrm w}d^{3})$ warm-up is negligible and"
         r" \eqref{eq:cost} reduces to $O(dN)$; at the $N$ used here the warm-up inflates"
         r" the exponent, which is why the measured value exceeds one.}",
         r"\label{tab:cost}",
         r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"method & " + " & ".join([r"$d{=}%d$" % q["d"] for q in rows])
         + r" & $s_{\mathrm{time}}$ & $s_{\mathrm{flop}}$ & $s_{\infty}$ \\",
         r"\midrule"]
    names = [("BGSN", r"\textbf{BGSN} (long-run interval)"),
             ("BGSN_plugin", r"\quad with the i.i.d.\ plug-in variance"),
             ("SN_iid_pinv_d", r"streaming Newton, Bernoulli $p=1/d$"),
             ("SN_iid_p1", r"streaming Newton, every observation ($p=1$)"),
             ("ASGD", r"ASGD (first order, random scaling)"),
             ("OfflineHAC_L100", r"offline HAC, $L=100$ \emph{(not streaming)}")]
    for k, lab in names:
        cells = [f"{q[k]*1e3:.1f}" if k in q else "--" for q in rows]
        asym = d.get("flop_exponents_large_N", {}).get(k)
        L.append(lab + " & " + " & ".join(cells) + " & "
                 + (f"{e[k]:.2f}" if k in e else "--") + " & "
                 + (f"{fe[k]:.2f}" if k in fe else "--") + " & "
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
                                  r"Lrv|Se|Rate|Asym|Wc)[A-Z]", u))
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
    for fn in (table_main, table_ablation, table_real, table_wellcond, table_cost):
        try:
            r = fn()
            if fn is table_main:
                facts["main"] = r
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
        m[f"time ratio {tag}"] = fnum(f["time_ratio"], 2)
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
    except FileNotFoundError:
        pass
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
        d5 = common.load("exp5_ablations.json")
        w0 = [q for q in d5["warm"] if q["warm_mult"] <= 2]
        if w0:
            m["cov no warmup"] = fnum(min(q["cov_ft"] for q in w0))
    except FileNotFoundError:
        pass
    emit_macros(m)
    common.save("facts.json", facts)
    print(json.dumps(m, indent=1)[:3000])
