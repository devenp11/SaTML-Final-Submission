# Manuscript source (anonymous)

The compiled paper is `main_anon.pdf`. You do not need to build anything to read it.

Format: `\documentclass[conference]{IEEEtran}`, 10pt, stock geometry, nothing overridden.
10 pages, with the references beginning on page 9.

## Files

| file | what it is |
|---|---|
| `main_anon.tex` | the anonymous manuscript source, generated from `../draft.md` by `../build_tex.py --anon` |
| `main_anon.pdf` | the compiled anonymous paper |
| `main_anon.bbl` | the formatted bibliography, so a rebuild needs no `bibtex` pass |
| `refs.bib` | bibliography source: 11 entries, all verified against a primary source |
| `IEEEtran.cls` | the IEEE conference class, from CTAN |
| `IEEEtran.bst` | the matching BibTeX style (v1.14), from CTAN |
| `figs/*.pdf` | the three vector figures |

This artifact contains no author-identity files and no named build. The author block in
`main_anon.tex` is anonymous, and `build_tex.py` produces an anonymous block by default,
so rebuilding here cannot de-anonymize the paper.

## Rebuilding (optional)

Only needed if you want to confirm the PDF matches the source.

```bash
pdflatex main_anon && pdflatex main_anon
```

`main_anon.bbl` is included, so the `bibtex` pass is not required. If you would rather
regenerate the bibliography from `refs.bib`:

```bash
pdflatex main_anon && bibtex main_anon && pdflatex main_anon && pdflatex main_anon
```

Uploading this folder to Overleaf and compiling there also works.

To regenerate the `.tex` itself from the Markdown source:

```bash
python ../build_tex.py --anon      # ../draft.md -> main_anon.tex
python ../check_tex.py             # structural sanity checks on main_anon.tex
```
