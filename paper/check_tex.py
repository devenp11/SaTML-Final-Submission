"""Structural sanity checks on the generated LaTeX, before attempting a compile.

    python paper/check_tex.py                  # checks paper/tex/main_anon.tex
    python paper/check_tex.py <file.tex>       # or any .tex you name

Defaults to the anonymous manuscript, which is the one this artifact ships.
"""
import re
import sys
import pathlib

TEX = pathlib.Path(__file__).resolve().parent / "tex"
target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else TEX / "main_anon.tex"
if not target.is_absolute() and not target.exists():
    target = TEX / target.name
if not target.exists():
    raise SystemExit(
        f"\nNo such LaTeX file: {target}\n"
        f"Generate the anonymous manuscript first:\n"
        f"    python paper/build_tex.py --anon\n")

print(f"checking {target.name}\n")
t = target.read_text(encoding="utf-8")


def n(p):
    return len(re.findall(p, t))


COUNTS = [
    ("sections", r"\\section\{"),
    ("subsections", r"\\subsection\{"),
    ("table", r"\\begin\{table\}"),
    ("table*", r"\\begin\{table\*\}"),
    ("figure", r"\\begin\{figure\}"),
    ("figure*", r"\\begin\{figure\*\}"),
    ("enumerate", r"\\begin\{enumerate\}"),
    ("cite commands", r"\\cite\{"),
    ("inline math", r"\$[^$]+\$"),
]
for label, pat in COUNTS:
    print("{:20s}: {}".format(label, n(pat)))

dollars = t.count("$")
print("{:20s}: {} -> {}".format("$ count", dollars, "OK" if dollars % 2 == 0 else "UNBALANCED"))

for env in ("table", "table*", "figure", "figure*", "enumerate", "equation*", "abstract", "document"):
    b = n(r"\\begin\{" + re.escape(env) + r"\}")
    e = n(r"\\end\{" + re.escape(env) + r"\}")
    if b or e:
        flag = "OK" if b == e else "*** MISMATCH ***"
        print("  env {:11s} begin {}  end {}  {}".format(env, b, e, flag))

stripped = re.sub(r"\$[^$]*\$", "", t)
stripped = re.sub(r"\\texttt\{[^}]*\}", "", stripped)
print("{:20s}: {}".format("unescaped %", len(re.findall(r"(?<!\\)%", stripped))))
no_tab = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", stripped, flags=re.S)
print("{:20s}: {}".format("unescaped &", len(re.findall(r"(?<!\\)&", no_tab))))
print("{:20s}: {}".format("bare underscores", len(re.findall(r"(?<![\\\w])_", stripped))))

md = re.findall(r"\*\*|(?<![\w\\])\*[A-Za-z]", t)
print("{:20s}: {} {}".format("leftover md", len(md), md[:6]))
unres = [x for x in re.findall(r"\[[^\]\[]{3,45}\]", t) if not x.startswith("[t]")]
print("{:20s}: {} {}".format("unresolved [ ]", len(unres), unres[:6]))

prob = []
for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", t, re.S):
    ncol = len(re.sub(r"[^lcr]", "", m.group(1)))
    for row in m.group(2).split(r"\\"):
        row = re.sub(r"\\(top|mid|bottom)rule", "", row).strip()
        if not row:
            continue
        got = row.count("&") + 1
        if got != ncol:
            prob.append((ncol, got, row[:60]))
print("{:20s}: {}".format("tabular mismatches", len(prob)))
for p in prob[:6]:
    print("   expected {} cols, got {}: {}".format(p[0], p[1], p[2]))
