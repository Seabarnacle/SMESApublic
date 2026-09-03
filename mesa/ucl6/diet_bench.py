

import json, os, psi4, pandas as pd
from pathlib import Path

JSON_TABLE = "DietGMTKN55_100.json"
GEOM_ROOT  = Path("gmtkn_geom")  
BASIS      = "def2-TZVP"     

def scf_iters_for_xyz(xyz_file: Path) -> int | None:
    mol = psi4.geometry(xyz_file.read_text())
    psi4.core.set_output_file("scratch.out", False)

    psi4.set_options({
        "basis": BASIS,
        "scf_type": "df",
        "d_convergence": 1e-8,
    })

    try:
        _, wfn = psi4.energy("scf", return_wfn=True, molecule=mol)
        return int(wfn.get_variable("SCF ITERATIONS"))
    except psi4.SCFConvergenceError:
        return None       # mark failure

def main():
    tasks = json.load(open(JSON_TABLE))
    rows  = []

    for rec in tasks:
        system_id = f"{rec['set']}_{rec['id']}"
        xyz_path  = GEOM_ROOT / rec["set"] / f"{rec['id']}.xyz"

        if not xyz_path.exists():
            print(f"!! missing geometry {xyz_path}")
            nit = None
        else:
            nit = scf_iters_for_xyz(xyz_path)

        rows.append({"system": system_id, "n_iter": nit})
        print(f"{system_id:<20s}  {nit}")

    df = pd.DataFrame(rows)
    df.to_excel("DIIS_benchmark.xlsx", index=False)
    print("\nWrote DIIS_benchmark.xlsx")

if __name__ == "__main__":
    main()