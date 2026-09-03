import json
import sys

import psi4
from psi4 import SCFConvergenceError
#TESTETSTETESTTEST

def run_method(method, do_scf, mesa_error):
    default_options = {
        "basis": BASIS_SET,
        "SCF_INITIAL_ACCELERATOR": "NONE",
        "LIST": "NONE",
        "DIIS": False,
        "MESA": False,
        "GUESS": "CORE",
        "SCF_TYPE": "PK",
        "reference": "UHF"
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

    print(default_options)
    psi4.set_options(default_options)
    #psi4.energy("scf" if do_scf else "svwn")
    psi4.energy("scf")

#BASIS_SET = "cc-pVDZ"
BASIS_SET = "LANL2DZ"
# h2o = psi4.geometry("""
#   O
#   H 1 0.965
#   H 1 0.965 2 103.75
# """)
#
no_molecule = psi4.geometry("""
0 2
N
O 1 1.15
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
    "FDIIS",
    "EDIIS",
    "LISTB",
    "LISTF",
    "LISTR",
    "ADIIS",
    "DIIS",
    "DIIS/ADIIS",
    "DIIS/EDIIS",
    "MESA"
]

scf_conv_method = "MESA"
try:
    if scf_conv_method != "MESA":
        run_method(scf_conv_method, do_scf=False, mesa_error="")
        psi4.core.clean()
        print("Run done: completed " + scf_conv_method)
    else:
        for error_type in ["INTRA_D","COMMUTATOR", "D_F", "INTER_F", "INTER_D"]:
            run_method("MESA", do_scf=False, mesa_error=error_type)
            psi4.core.clean()
            print("Run done: completed " + scf_conv_method + " with error type " + error_type)
except SCFConvergenceError as e:
    print(e)
    print("Run done: failed to converge " + scf_conv_method)
