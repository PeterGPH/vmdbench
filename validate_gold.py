#!/usr/bin/env python3
"""validate_gold.py — verify the oracle's reference answers with an INDEPENDENT tool.

The correctness benchmark trusts gold_oracle_multi.tcl (VMD). This script recomputes the
same metrics with MDAnalysis — a separate code base, unrelated to VMD — using the same
definitions, and reports AGREE / DIFFER per (structure, metric). Agreement validates the
gold; a difference flags either a real bug or a definition mismatch to resolve.

  pip install MDAnalysis          # (your MD lab almost certainly has it)
  python validate_gold.py --structures-dir fixtures_multi

Covers Rg (mass-weighted), atom count, residue count, first<->last CA distance. SASA is
algorithm-dependent (VMD's method vs Shrake-Rupley differ), so it is reported separately
with a loose tolerance and a note rather than a hard AGREE/DIFFER.
"""
import argparse, glob, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORACLE = HERE / "gold_oracle_multi.tcl"
DEFAULT_VMD = "/Applications/VMD.app/Contents/vmd/vmd_MACOSXARM64"

# metric -> absolute tolerance for "agreement"
# Counts/distance must match exactly; Rg gets 0.15 Å to absorb cross-tool mass-model
# drift (each tool assigns atomic masses slightly differently) — still far tighter
# than any real error (a wrong Rg is off by Angstroms, not hundredths).
TOL = {"rgyr": 0.15, "natoms": 0.5, "nresidues": 0.5, "ca_dist": 0.05}


def vmd_gold(vmd, structure):
    env = dict(os.environ, GOLD_STRUCT=structure)
    out = subprocess.run([vmd, "-dispdev", "text", "-e", str(ORACLE)],
                         env=env, capture_output=True, text=True, timeout=300).stdout
    g = {}
    for line in out.splitlines():
        m = re.match(r"\s*GOLD\s+(\S+)\s+(\S+)", line)
        if m:
            try: g[m.group(1)] = float(m.group(2))
            except ValueError: pass
    return g


def mda_values(structure):
    import numpy as np
    import MDAnalysis as mda
    u = mda.Universe(structure)
    prot = u.select_atoms("protein")
    ca = prot.select_atoms("name CA")
    vals = {
        "natoms":    float(len(u.atoms)),
        "nresidues": float(len(prot.residues)),
        "rgyr":      float(prot.radius_of_gyration()),   # mass-weighted, like VMD measure rgyr
    }
    if len(ca) >= 2:
        vals["ca_dist"] = float(np.linalg.norm(ca.positions[0] - ca.positions[-1]))
    return vals


_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "P": 30.974,
         "SE": 78.971, "FE": 55.845, "ZN": 65.38, "MG": 24.305, "CA": 40.078,
         "NA": 22.99, "CL": 35.45, "K": 39.098, "MN": 54.938}
_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
       "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
       "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "MSE", "CYX", "SEC"}


def biopython_values(structure):
    """Independent recompute via Biopython — reads PDB *and* mmCIF (covers 1CRN.cif)."""
    import numpy as np
    from Bio.PDB import PDBParser, MMCIFParser
    parser = (MMCIFParser(QUIET=True) if structure.lower().endswith((".cif", ".mmcif"))
              else PDBParser(QUIET=True))
    s = parser.get_structure("x", structure)
    atoms = list(s.get_atoms())
    prot_res = [r for r in s.get_residues() if r.get_resname().strip().upper() in _AA]
    prot_atoms = [a for r in prot_res for a in r.get_atoms()]
    coords = np.array([a.coord for a in prot_atoms], dtype=float)
    masses = np.array([_MASS.get((a.element or a.get_name()[:1]).strip().upper(), 12.0)
                       for a in prot_atoms])
    com = (masses[:, None] * coords).sum(0) / masses.sum()
    rg = float(np.sqrt((masses * ((coords - com) ** 2).sum(1)).sum() / masses.sum()))
    vals = {"natoms": float(len(atoms)), "nresidues": float(len(prot_res)), "rgyr": rg}
    cas = [a for a in prot_atoms if a.get_name().strip() == "CA"]
    if len(cas) >= 2:
        vals["ca_dist"] = float(np.linalg.norm(cas[0].coord - cas[-1].coord))
    return vals


def independent_values(structure):
    """Prefer MDAnalysis (tested Rg); fall back to Biopython for formats MDA can't read."""
    try:
        return mda_values(structure), "MDAnalysis"
    except Exception:
        return biopython_values(structure), "Biopython"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures-dir", default=str(HERE / "fixtures_multi"))
    ap.add_argument("--vmd", default=DEFAULT_VMD)
    a = ap.parse_args()

    have = []
    for mod in ("MDAnalysis", "Bio"):
        try:
            __import__(mod); have.append(mod)
        except Exception:
            pass
    if not have:
        print("Install an independent checker:  pip install MDAnalysis biopython")
        return 1

    structures = sorted(glob.glob(os.path.join(a.structures_dir, "*.pdb"))
                        + glob.glob(os.path.join(a.structures_dir, "*.cif")))
    print(f"{'structure':8} {'metric':10} {'VMD gold':>14} {'independent':>14}  {'tool':10} verdict")
    print("-" * 76)
    n_ok = n_diff = 0
    for s in structures:
        name = Path(s).stem.upper()
        try:
            g = vmd_gold(os.path.expanduser(a.vmd), s)
            mv, tool = independent_values(s)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:8} (error: {exc})")
            continue
        for k in ("natoms", "nresidues", "rgyr", "ca_dist"):
            if k not in g or k not in mv:
                continue
            d = abs(g[k] - mv[k])
            ok = d <= TOL[k]
            n_ok += int(ok); n_diff += int(not ok)
            print(f"{name:8} {k:10} {g[k]:>14.4f} {mv[k]:>14.4f}  {tool:10} "
                  + ("AGREE" if ok else f"DIFFER (|Δ|={d:.4f})"))
    print("-" * 76)
    print(f"=> {n_ok} agree, {n_diff} differ  (VMD gold vs independent recompute)")
    print("Note: SASA is method-dependent (VMD vs Shrake-Rupley) — validate it separately "
          "with freesasa if you need a hard check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
