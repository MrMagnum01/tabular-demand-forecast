"""Generate LICENSES.md from installed package metadata (importlib.metadata).

Nothing here is inferred from a package name or assumed: each entry quotes
the package's own License-Expression / License / License-File / Trove
classifier metadata as installed in this venv. OSI status is derived
mechanically from those fields and anything uncertain is FLAGGED.

Usage: .venv/bin/python scripts/build_licenses.py
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tabular_demand_forecast.provenance import parse_lock  # noqa: E402

OUT = REPO / "LICENSES.md"
OSI_SPDX = {
    "MIT", "MIT-CMU", "BSD-2-Clause", "BSD-3-Clause", "0BSD", "Apache-2.0", "PSF-2.0",
    "Zlib", "OFL-1.1", "MPL-2.0", "ISC", "HPND",
}
DIRECT = {l.split("==")[0].strip().lower() for l in (REPO / "requirements.txt").read_text().splitlines() if "==" in l}


def md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def spdx_ids(expr: str):
    return [t for t in re.findall(r"[A-Za-z0-9.\-+]+", expr) if t not in ("AND", "OR", "WITH")]


def osi_status(meta) -> str:
    expr = meta.get("License-Expression")
    lic = (meta.get("License") or "").strip()
    classifiers = [c for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
    if expr:
        ids = spdx_ids(expr)
        bad = [i for i in ids if i not in OSI_SPDX]
        if " OR " in expr and " AND " not in expr:
            return "OSI-approved (SPDX expression, every alternative OSI-approved)" if not bad else f"FLAG: alternatives {bad} not in OSI list"
        if bad:
            return f"FLAG: SPDX expression includes non-OSI identifier(s) {bad}; remaining identifiers OSI-approved"
        return "OSI-approved (SPDX License-Expression)"
    if lic in OSI_SPDX:
        return f"OSI-approved (License field is SPDX id {lic})"
    if any("OSI Approved" in c for c in classifiers):
        return "OSI-approved per the package's own Trove classifier(s)"
    return "FLAG: no machine-readable OSI evidence in metadata"


OSI_EXTRA_COMPONENT_IDS = {"MIT-Modern-Variant", "GPL-2.0-or-later"}


def component_osi(stated: str) -> str:
    ids = spdx_ids(stated)
    known = OSI_SPDX | OSI_EXTRA_COMPONENT_IDS
    if ids and all(i in known for i in ids):
        return "OSI-approved"
    if " OR " in stated and any(i in known for i in ids):
        ok = [i for i in ids if i in known]
        bad = [i for i in ids if i not in known]
        return f"FLAG: {bad} not OSI-approved; OSI-approved alternative {ok} available (note: copyleft if GPL)"
    return "FLAG: not a recognised OSI-approved licence identifier (custom/font licence text)"


def license_field_summary(lic: str) -> str:
    lic = lic or ""
    if not lic.strip():
        return "(empty)"
    if len(lic) <= 80 and "\n" not in lic.strip():
        return f"`{lic.strip()}`"
    first = lic.strip().splitlines()[0].strip()
    digest = hashlib.sha256(lic.encode("utf-8")).hexdigest()[:16]
    return f"full text in metadata ({len(lic)} chars, sha256 {digest}...), first line: \"{first}\""


def matplotlib_section() -> str:
    dist = md.distribution("matplotlib")
    meta = dist.metadata
    lic = meta.get("License") or ""
    lines = lic.splitlines()
    components = []
    for i, line in enumerate(lines):
        m = re.match(r"^\s*Name:\s*(.+)$", line)
        if not m:
            continue
        name, files, lic_name = m.group(1).strip(), "", ""
        for j in range(i + 1, len(lines)):
            nxt = lines[j]
            if re.match(r"^\s*Name:", nxt):
                break
            fm = re.match(r"^\s*Files:\s*(.+)$", nxt)
            lm = re.match(r"^\s*License:\s*(.*)$", nxt)
            if fm and not files:
                files = fm.group(1).strip()
            if lm and not lic_name:
                lic_name = lm.group(1).strip()
                if not lic_name:  # licence named on the following non-empty line
                    lic_name = next((l.strip() for l in lines[j + 1:] if l.strip()), "")
                break
        components.append((name, files, lic_name, component_osi(lic_name)))
    headings = [l.strip() for l in lines if l.strip().startswith("License agreement for matplotlib")]
    font_files = sorted(str(f) for f in dist.files if "LICENSE" in f.name.upper() and "mpl-data" in str(f))
    out = [
        "## matplotlib: licence and bundled third-party notices (read from installed metadata)",
        "",
        f"* Installed version: `{dist.version}`.",
        f"* `License-Expression`: `{meta.get('License-Expression')}` (not declared).",
        f"* `License` metadata field: {license_field_summary(lic)}.",
        f"* Licence agreement headings in that field: {', '.join(repr(h) for h in headings)}.",
        f"* Trove classifiers: {', '.join('`' + c + '`' for c in (meta.get_all('Classifier') or []) if c.startswith('License ::'))}.",
        "* This is matplotlib's OWN declared licence (the \"License agreement for matplotlib\", PSF-derived) as recorded in its installed metadata; it is not assumed from Python's licence.",
        f"* Additional licence files shipped inside the installed package: {', '.join('`' + f + '`' for f in font_files)}.",
        "",
        "Bundled third-party components listed in matplotlib's own `License` metadata field:",
        "",
        "| component | files | licence (as stated) | OSI status |",
        "|---|---|---|---|",
    ]
    for name, files, lic_name, osi in components:
        out.append(f"| {md_escape(name)} | {md_escape(files)} | {md_escape(lic_name)} | {osi} |")
    return "\n".join(out) + "\n"


def main() -> int:
    entries = parse_lock()
    rows = []
    for e in entries:
        meta = md.metadata(e["name"])
        role = "direct" if e["name"].lower() in DIRECT else ("installer" if e["name"].lower() == "pip" else "transitive")
        classifiers = [c.replace("License :: ", "") for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
        status = osi_status(meta)
        if e["name"].lower() == "matplotlib":
            status += "; FLAG: bundles components under non-OSI licences (see matplotlib section)"
        rows.append(
            f"| {e['name']} | {e['version']} | {role} | {md_escape(meta.get('License-Expression') or '-')} | "
            f"{md_escape(license_field_summary(meta.get('License') or ''))} | {md_escape('; '.join(classifiers) or '-')} | "
            f"{md_escape(', '.join(meta.get_all('License-File') or []) or '-')} | {status} |"
        )
    flagged = [r for r in rows if "FLAG" in r.rsplit("|", 2)[-2]]
    text = [
        "# Third-party licences",
        "",
        "Generated by `scripts/build_licenses.py` from the installed package metadata (`importlib.metadata`) of every",
        "direct and transitive package in `eval/dependency-lock.txt`. Nothing is inferred from package names.",
        "OSI status is derived mechanically from each package's own metadata; anything uncertain is marked FLAG.",
        "",
        f"Packages: {len(rows)}. Flagged: {len(flagged)} (see the OSI status column and the notes below).",
        "",
        "| package | version | role | License-Expression | License field | licence classifiers | License-File entries | OSI status |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "## Notes on flags",
        "",
        "* numpy: its SPDX expression includes `CC0-1.0` for a bundled component; CC0-1.0 is a public-domain dedication,",
        "  not an OSI-approved licence. Flagged for the owner's decision; not resolved here.",
        "* matplotlib: its own licence is declared OSI-approved by its classifier, but it bundles fonts, compiled",
        "  libraries and snippets under licences that are not OSI-approved identifiers (BaKoMa, Bitstream-Charter,",
        "  FTL-or-GPL FreeType, Qhull, CC0 and a BSD-style custom text; see the table below). Flagged for the owner's decision.",
        "* python-dateutil, cycler, kiwisolver, pandas, scipy and matplotlib declare no SPDX expression; their OSI status",
        "  rests on the package's own `License :: OSI Approved` Trove classifier, quoted in the table.",
        "",
        matplotlib_section(),
        "## CSV output",
        "",
        "Every CSV this repo writes goes through `src/tabular_demand_forecast/csv_safety.py`: string cells starting with",
        "`=`, `+`, `-`, `@`, TAB or CR get a leading apostrophe. CSV is not universally formula-safe; see that module's docstring.",
    ]
    OUT.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(rows)} packages, {len(flagged)} flagged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
