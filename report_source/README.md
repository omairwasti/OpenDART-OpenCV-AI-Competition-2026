# OpenDART technical report source

Build from this directory with `tectonic --keep-logs main.tex`, or run `xelatex -interaction=nonstopmode -halt-on-error main.tex` twice. Figures are local to `images/`; the keyed bibliography is inline. Standard TeX fonts are used. English Babel supports XeLaTeX without a LuaLaTeX-only bidi option.

The 16-page report retains the dynamics/controller derivations and native equations. Body numbers and equations are unchanged by the final copy pass. No full DRACO source download is needed to build it.
