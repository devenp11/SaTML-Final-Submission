"""Convert paper/draft.md to submission-ready IEEEtran conference LaTeX.

    \\documentclass[conference]{IEEEtran}   10pt, stock geometry, nothing overridden.

Handles: escaping, unicode -> LaTeX, the draft's plain-text math, booktabs tables
(table* for wide ones), figure/figure* placement, enumerate, sections/subsections,
abstract + IEEEkeywords, and citation placeholders -> \\cite keys.
"""
import re, pathlib, html

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "paper" / "draft.md"
import sys
ANON = "--anon" in sys.argv
TEX = ROOT / "paper" / "tex" / ("main_anon.tex" if ANON else "main.tex")

TITLE = ("Attackers Learn Twice: Survivorship Bias Hides\\\\ "
         "Within-Attacker Learning in Adaptive Red-Teaming")
KEYWORDS = ("adaptive red-teaming, prompt injection, attacker learning curves, "
            "survivorship bias, frailty, defense evaluation, large language models")

# bracketed placeholder in draft.md  ->  bibtex key
CITES = {
    "Attacker Moves Second": "attackerMovesSecond",
    "PISmith": "pismith",
    "GPT-Red": "gptred",
    "LLMail-Inject": "llmail",
    "Tensor Trust": "tensortrust",
    "Toyer et al.": "tensortrust",
    "Abdelnabi et al.": "llmail",
    "Balan & Putter": "balanputter",
    "Vaupel et al.": "vaupel",
    "Ma": "ma2011frailty",
    "When Benchmarks Lie": "benchmarksLie",
    "Defenses Learn Surface Heuristics": "surfaceHeuristics",
    "Layered LLM Defenses": "layeredDefenses",
}

FIGS = {  # md marker -> (file, caption, wide?)
    "Figure 1": ("F1_reversal", "(a) Pooled success hazard falls in every LLMail cell. "
                 "(b) Tensor Trust: 18.1\\% to 0.3\\%, monotone, all three target models. "
                 "(c) Conditioning on each attacker's own sequence, successes sit late under "
                 "every stopping-rule treatment: the within-attacker hazard rises.", True),
    "Figure 2": ("F2_fsr_variation", "Family survival rate, which measures local robustness under an observed "
                 "attacker mutation distribution. (a) 91 Tensor Trust "
                 "defenses span 11\\% to 100\\%. (b) LLMail detectors on both target models; "
                 "Spotlight is excluded because it is preventive and the detector flag is true "
                 "for it by construction.", False),
    "Figure 3": ("F3_paired_fsr", "Within-team paired FSR on LLMail: the same teams attacking "
                 "both defenses of a pair. Seven of twelve pairs differ; intervals excluding "
                 "zero are marked.", False),
}

TABLE_CAPS = {
    1: "Within-attacker success position under three treatments of the stopping rule. "
       "Mean of $T$ with 95\\% bootstrap CI over cells; cell counts in parentheses.",
    2: "Split-half rate difference, second half minus first, against a within-cell label "
       "permutation null (2{,}000 permutations).",
    3: "Conditional logistic regression with attacker fixed effects.",
    4: "Family survival rate on LLMail, detector-level evasion flag.",
    5: "Within-attacker split-half restricted by the wall-clock span of the attacker's "
       "whole sequence, ruling out community diffusion.",
    6: "Causal test of the determinism hypothesis.",
    7: "Pooled family survival under the two admissible family-seed definitions.",
}

UNI = [("\u207b\u2074", "$^{-4}$"),   # 10^-4; must precede the single-char rules
       ("\u207b", "$^{-}$"), ("\u2074", "$^{4}$"), ("\u00e8", "\\`{e}"),
       ("\u2014", "---"), ("\u2013", "--"), ("\u2212", "$-$"), ("\u00d7", "$\\times$"),
       ("\u2264", "$\\le$"), ("\u2265", "$\\ge$"), ("\u2248", "$\\approx$"),
       ("\u03b1", "$\\alpha$"), ("\u03b2", "$\\beta$"), ("\u03c4", "$\\tau$"),
       ("\u03a3", "$\\Sigma$"), ("\u2208", "$\\in$"), ("\u2026", "\\ldots"),
       ("\u00a7", "\\S"), ("\u00b7", "$\\cdot$"), ("\u2192", "$\\rightarrow$"),
       ("\u2019", "'"), ("\u2018", "`"), ("\u201c", "``"), ("\u201d", "''"),
       ("\u0302", ""), ("\u00b1", "$\\pm$"), ("\u2265", "$\\ge$")]


def esc(s):
    """Escape LaTeX specials in plain prose.

    < > | are not TeX specials but render as inverted punctuation and an em-dash in
    the OT1 text encoding IEEEtran uses, so "0 < s < W" comes out as "0 ¡ s ¡ W" and
    "|beta|" as an em-dash pair. They need the text commands.
    """
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"),
                 ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"),
                 ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"),
                 ("<", "\\textless{}"), (">", "\\textgreater{}"), ("|", "\\textbar{}")]:
        s = s.replace(a, b)
    # straight double quotes -> proper TeX open/close pairs
    out, open_q = [], True
    for ch in s:
        if ch == '"':
            out.append("``" if open_q else "''")
            open_q = not open_q
        else:
            out.append(ch)
    return "".join(out)


def unicode_fix(s):
    for a, b in UNI:
        s = s.replace(a, b)
    return s


def mathify(s):
    """The draft writes math in markdown emphasis; turn it into real LaTeX math.
    Runs before escaping so that _ and { } inside math survive."""
    s = re.sub(r"\*([A-Za-z])\*_\{?([A-Za-z0-9]+)\}?\(\*?([A-Za-z])\*?\)",
               r"$\1_{\\mathrm{\2}}(\3)$", s)                       # *h*_pop(*n*)
    s = re.sub(r"\*([A-Za-z])_\{?([A-Za-z0-9]{1,3})\}?\(([A-Za-z])\)\*",
               r"$\1_{\2}(\3)$", s)                                  # *h_i(n)*
    s = re.sub(r"\*([A-Za-z])\*_\{?([A-Za-z0-9]+)\}?", r"$\1_{\\mathrm{\2}}$", s)   # *h*_pop
    s = re.sub(r"\*([A-Za-z])_\{?([A-Za-z0-9]{1,3})\}?\*", r"$\1_{\2}$", s)         # *N_c*
    s = re.sub(r"(?<![A-Za-z0-9_$\\])([A-Za-z\u03b1-\u03c9])_\{?([A-Za-z0-9]{1,3})\}?(?![A-Za-z0-9_])",
               lambda m: "$" + (("\\" + {"\u03b1": "alpha", "\u03b2": "beta",
                                          "\u03c4": "tau"}.get(m.group(1), ""))
                                 or m.group(1)) + "_{" + m.group(2) + "}$", s)
    return s


def inline(s):
    """Markdown inline -> LaTeX, protecting math and code from the escaper."""
    keep = []

    def stash(txt):
        keep.append(txt)
        return f"\x00{len(keep)-1}\x00"

    s = re.sub(r"`([^`]+)`", lambda m: stash("\\texttt{" + esc(m.group(1)) + "}"), s)
    s = mathify(s)
    s = re.sub(r"\$[^$]+\$", lambda m: stash(m.group(0)), s)     # protect math
    # citations
    def cite(m):
        keys, unknown = [], []
        for part in re.split(r";\s*", m.group(1)):
            part = part.strip()
            (keys.append(CITES[part]) if part in CITES else unknown.append(part))
        if not keys:
            return m.group(0)
        out = "\\cite{" + ",".join(dict.fromkeys(keys)) + "}"
        return out + ("" if not unknown else " ")
    s = re.sub(r"\[([^\[\]]+?)\]", cite, s)
    s = re.sub(r"\\cite\{[^}]*\}", lambda m: stash(m.group(0)), s)
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: "\x01" + m.group(1) + "\x02", s)
    s = re.sub(r"(?<![A-Za-z*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", lambda m: "\x03" + m.group(1) + "\x04", s)
    s = esc(s)
    s = unicode_fix(s)
    s = s.replace("\x01", "\\textbf{").replace("\x02", "}")
    s = s.replace("\x03", "\\textit{").replace("\x04", "}")
    s = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)
    return s


def mk_table(rows, n):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    head, body = cells[0], cells[1:]
    ncol = len(head)
    # Column count alone is a bad predictor of width: Table 1 has three columns but
    # long labels plus two confidence intervals per row, and overflows a column by
    # 144pt. Estimate the set width from the widest row's character count too.
    widest = max(sum(len(c) for c in r) for r in cells)
    wide = ncol >= 4 or widest > 58
    env = "table*" if wide else "table"
    colspec = "l" + "c" * (ncol - 1)
    # \textsc{...\textit{x}...} asks for OT1/ptm/m/scit, which does not exist and
    # is silently substituted. Lift any italic run out of the small-caps group.
    def _hdr(c):
        parts = re.split(r"(\\textit\{[^}]*\})", inline(c))
        return "".join(p if p.startswith("\\textit{")
                       else ("\\textsc{" + p + "}" if p.strip() else p)
                       for p in parts)
    h = " & ".join(_hdr(c) for c in head) + " \\\\"
    b = "\n".join(" & ".join(inline(c) for c in r) + " \\\\" for r in body)
    # Captions bind POSITIONALLY: the most recent "**Table N.**" line in the draft
    # belongs to the next table emitted, whatever number was typed. Keying on the
    # typed number silently mis-assigned captions when the draft gained a table.
    cap = PENDING_CAP.pop("next", None) or TABLE_CAPS.get(n, "")
    if not cap:
        print(f"WARNING: table {n} has no caption -- add a **Table N.** line before it")
    return (f"\\begin{{{env}}}[t]\n\\centering\n\\caption{{{cap}}}\n\\label{{tab:{n}}}\n"
            f"\\small\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{h}\n\\midrule\n{b}\n"
            f"\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


EQUATIONS = {
    "has not succeeded before":
        r"h_{\mathrm{pop}}(n) \;=\; E\!\left[\, h_i(n) \;\middle|\; "
        r"i \text{ has not succeeded before } n \,\right].",
    "mean(success positions)":
        r"T \;=\; \operatorname{mean}(\text{success positions}) \,/\, N \;-\; 0.5",
    "logit Pr[success at attempt":
        r"\operatorname{logit} \Pr\!\left[\text{success at attempt } n\right] "
        r"\;=\; \alpha_c + \beta \log n,",
    "FSR(*D*)":
        r"\mathrm{FSR}(D) \;=\; \Pr\!\left[\, v \text{ evades } D \;\middle|\; "
        r"v \in V(P),\; P \text{ evaded } D \,\right],",
}

PENDING_CAP = {}
lines = SRC.read_text(encoding="utf-8").splitlines()
out, i, tno, in_abs, abs_paras = [], 0, 0, False, []

while i < len(lines):
    ln = lines[i]
    if ln.startswith("# ") or ln.startswith("*Draft v1") or ln.strip() == "---":
        i += 1; continue
    if ln.startswith("## "):
        h = ln[3:].strip()
        if h == "Abstract":
            in_abs = True; i += 1; continue
        in_abs = False
        m = re.match(r"\d+\.\s+(.*)", h)
        out.append("\n\\section{" + inline(m.group(1) if m else h) + "}")
        i += 1; continue
    if ln.startswith("### "):
        h = ln[4:].strip()
        m = re.match(r"\d+\.\d+\s+(.*)", h)
        out.append("\n\\subsection{" + inline(m.group(1) if m else h) + "}")
        i += 1; continue
    if ln.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append(lines[i]); i += 1
        tno += 1
        out.append(mk_table(rows, tno))
        continue
    if re.match(r"^\d+\.\s", ln):
        items = []
        while i < len(lines) and re.match(r"^\d+\.\s", lines[i]):
            items.append(re.sub(r"^\d+\.\s", "", lines[i])); i += 1
        out.append("\\begin{enumerate}\n" + "\n".join("\\item " + inline(x) for x in items)
                   + "\n\\end{enumerate}")
        continue
    if ln.startswith("  ") and ln.strip():
        # The draft's four display equations mix symbols with prose. A general
        # converter puts operators inside \text{} and sets them upright, which is
        # wrong; there are only four, so each is written out properly here, keyed
        # by a distinctive fragment of its source line.
        key = next((k for k in EQUATIONS if k in ln), None)
        if key:
            out.append("\\begin{equation*}\n" + EQUATIONS[key] + "\n\\end{equation*}")
        else:
            print("WARNING: unmapped display equation:", ln.strip()[:60])
            s = inline(ln.strip())
            parts = [t[1:-1] if t.startswith("$") else "\\text{" + t.strip() + "}"
                     for t in re.split(r"(\$[^$]*\$)", s) if t and t.strip()]
            out.append("\\begin{equation*}\n" + " ".join(parts) + "\n\\end{equation*}")
        i += 1; continue
    mcap = re.match(r"^\*\*Table (\d+)\.\*\*\s*(.+)$", ln.strip())
    if mcap:
        PENDING_CAP["next"] = inline(mcap.group(2).strip())
        i += 1; continue
    if ln.strip():
        para = [ln.strip()]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||\d+\.\s|  \S|\*Draft|---)", lines[i]):
            para.append(lines[i].strip()); i += 1
        p = " ".join(para)
        if in_abs:
            abs_paras.append(inline(p)); continue
        out.append(inline(p))
        for marker, (fn, cap, wide) in list(FIGS.items()):
            if marker in p:
                env = "figure*" if wide else "figure"
                width = "\\textwidth" if wide else "\\columnwidth"
                out.append(f"\n\\begin{{{env}}}[t]\n\\centering\n"
                           f"\\includegraphics[width={width}]{{figs/{fn}.pdf}}\n"
                           f"\\caption{{{cap}}}\n\\label{{fig:{fn}}}\n\\end{{{env}}}\n")
                del FIGS[marker]
        continue
    i += 1

body = "\n\n".join(out)
abstract = " ".join(abs_paras)

ANON_BLOCK = "\n".join([
    r"\author{\IEEEauthorblockN{Anonymous Author(s)}",
    r"\IEEEauthorblockA{\textit{Paper under double-blind review}\\",
    r"\textit{Submitted to IEEE SaTML}}",
    "}",
])

if ANON:
    AUTHOR_BLOCK = ANON_BLOCK
else:
    # The author's identity lives in author_identity.py, which is excluded from the
    # anonymous artifact so that this script carries no identifying information.
    # A missing file yields the anonymous block, so the failure mode is safe.
    try:
        sys.path.insert(0, str(ROOT / "paper"))
        from author_identity import AUTHOR_NAME, AUTHOR_AFFIL, AUTHOR_LOCATION
        AUTHOR_BLOCK = "\n".join([
            r"\author{\IEEEauthorblockN{" + AUTHOR_NAME + "}",
            r"\IEEEauthorblockA{\textit{" + AUTHOR_AFFIL + r"}\\",
            AUTHOR_LOCATION + "}",
            "}",
        ])
    except ImportError:
        print("NOTE: paper/author_identity.py absent -> ANONYMOUS author block")
        AUTHOR_BLOCK = ANON_BLOCK

doc = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts

\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{url}
\usepackage[hidelinks,pdfauthor={},pdftitle={},pdfsubject={},pdfkeywords={},pdfcreator={}]{hyperref}

\begin{document}

\title{""" + TITLE + r"""}

""" + AUTHOR_BLOCK + r"""

\maketitle

\begin{abstract}
""" + abstract + r"""
\end{abstract}

\begin{IEEEkeywords}
""" + KEYWORDS + r"""
\end{IEEEkeywords}

""" + body + r"""

\bibliographystyle{IEEEtran}
\bibliography{refs}

\end{document}
"""

TEX.parent.mkdir(parents=True, exist_ok=True)
TEX.write_text(doc, encoding="utf-8")
print(f"wrote {TEX} ({len(doc)/1024:.0f} KB), {tno} tables, {3-len(FIGS)} figures placed")
if FIGS:
    print("WARNING unplaced figures:", list(FIGS))
