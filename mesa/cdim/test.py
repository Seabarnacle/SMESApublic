import json
import sys

import psi4


def run_method(method, do_scf, mesa_error):
    default_options = {
        "basis": BASIS_SET,
        "SCF_INITIAL_ACCELERATOR": "NONE",
        "LIST": "NONE",
        "DIIS": False,
        "MESA": False,
        "GUESS": "CORE",
        "SCF_TYPE": "PK",
    }
    if method == "MESA":
        default_options["MESA"] = True
        default_options["MESA_SCF_METHODS"] = MESA_SCF_METHODS
        default_options["MESA_ERROR"] = mesa_error
    elif method == "DIIS":
        default_options["DIIS"] = True
    elif method in ["FDIIS", "LISTB", "LISTF", "LISTR"]:
        default_options["LIST"] = method
    elif method in ["EDIIS", "ADIIS"]:
        default_options["SCF_INITIAL_ACCELERATOR"] = method
    elif method.startswith("DIIS") and method.endswith("ADIIS") or method.endswith("EDIIS"):
        default_options["DIIS"] = True
        default_options["SCF_INITIAL_ACCELERATOR"] = method.split("/")[1]

    psi4.set_options(default_options)
    psi4.energy("scf" if do_scf else "b3lyp")
    # psi4.energy("scf" if do_scf else "M08-HX")


BASIS_SET = "3-21G"

cdim = psi4.geometry("""
N
C     1    1.424500000
C     2    1.424500000   1    108.0000000
N     3    1.424500000   2    108.0000000   1    0.000000000
C     4    1.424500000   3    108.0000000   2    0.000000000
H     5    1.424500000   4    120.0000000   3    180.0000000
Cd    4    1.424500000   3    120.0000000   2    180.0000000
H     3    1.090000000   2    120.0000000   1    180.0000000
H     2    1.090000000   1    120.0000000   5    180.0000000
H     1    1.090000000   2    120.0000000   3    180.0000000
""")

MESA_SCF_METHODS = [
    "FDIIS",
    "EDIIS",
    "LISTB",
    "LISTF",
    "ADIIS",
    "DIIS",
    "LISTR",
]
eval_methods = [
    "DIIS",
    "ADIIS",
    "EDIIS",
    "DIIS/ADIIS",
    "DIIS/EDIIS",
    "FDIIS",
    "LISTB",
    "LISTF",
    "LISTR",
    "MESA"
]

scf_conv_method = sys.argv[1]
try:
    if scf_conv_method != "MESA":
        run_method(scf_conv_method, do_scf=False, mesa_error="")
        psi4.core.clean()
        print("Run done: completed " + scf_conv_method)
    else:
        for error_type in ["INTRA_D", "INTER_D", "INTER_F", "D_F", "COMMUTATOR"]:
            run_method("MESA", do_scf=False, mesa_error=error_type)
            psi4.core.clean()
            print("Run done: completed " + scf_conv_method + " with error type " + error_type)
except Exception as e:
    print(e)
    print("Run done: failed to converge " + scf_conv_method)
