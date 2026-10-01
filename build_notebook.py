"""Convert analysis.py (percent format) into responsible_ai_analysis.ipynb. No extra libraries needed."""
import json, re

src = open("analysis.py").read()
chunks = re.split(r"^# %%", src, flags=re.M)[1:]
cells = []
for ch in chunks:
    first, _, rest = ch.partition("\n")
    if "[markdown]" in first:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in rest.strip("\n").splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
    else:
        code = rest.strip("\n")
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": code.splitlines(keepends=True)})
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
json.dump(nb, open("responsible_ai_analysis.ipynb", "w"), indent=1)
print(f"Wrote responsible_ai_analysis.ipynb with {len(cells)} cells")
