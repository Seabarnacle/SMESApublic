import json
import sys
#w

import os


#psi4.set_options({'SCF__PRINT': 5})


import psi4
from psi4 import SCFConvergenceError


#psi4.set_memory("16 GB")


def run_method(method, do_scf, mesa_error):
    default_options = {
        "basis": BASIS_SET,
        "SCF_INITIAL_ACCELERATOR": "NONE",
        "LIST": "NONE",
        "DIIS": False,
        "MESA": False,
        "GUESS": "CORE",
        "SCF_TYPE": "PK",
        "MAXITER": 200,
        #"reference": "UHF"
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
    #psi4.energy("scf" if do_scf else "B3LYP")
    psi4.energy("scf")

# BASIS_SET = "3-21G"

# uc11l6 = psi4.geometry("""
# 0 1
# H  -0.97193796934  -1.23012761042  0.594000437264
# H  -1.4204758981  2.36938924571  0.112708925663
# H  -0.175911158088  -0.714392186745  -0.924670541634
# H  -0.719047137293  2.26806661386  -0.138326240518
# C  -0.0518142770964  -1.2909450799  -5.06115778309e-05
# H  0.783840401585  -0.877820246889  0.577908955421
# H  0.156404935873  -2.33889698571  -0.246971008857
# H  2.39894110246  1.8147262501  0.0254000842379
# """)

# BASIS_SET = "LANL2DZ"
# ucl6 = psi4.geometry("""
# U  
# Cl    1    2.39
# Cl    1    2.39   2    90
# Cl    1    2.39   2    90   3    -90
# Cl    1    2.39   3    90   2    180
# Cl    1    2.39   2    90   3    90
# Cl    1    2.39   2    90   3    180
# """)

# BASIS_SET = "cc-pVDZ"
# stretched_h8 = psi4.geometry("""
# 0 1
# H   0.0   0.0   0.0
# H   0.0   0.0   1.5
# H   0.0   0.0   3.0
# H   0.0   0.0   4.5
# H   0.0   0.0   6.0
# H   0.0   0.0   7.5
# H   0.0   0.0   9.0
# H   0.0   0.0  10.5
# symmetry c1
# """)

BASIS_SET = "LANL2DZ"
Rh4LCO = psi4.geometry("""
    Rh        0.07032        0.03018        0.01635
    Rh        0.07203        0.03309        2.50923
    Rh        2.26863       -0.06179        1.26137
    Rh        0.63158       -2.04779        1.26487
    C        2.77649        1.75042        1.26215
    O        2.95253        2.89861        1.26714
""")


# zero122 = psi4.geometry("""
#     0 1   
#     C         -0.21673        1.39210        0.44478
#     C          0.33720       -0.00231        0.75980
#     C         -1.61012        1.46871       -0.11967
#     C         -2.58179        0.51846       -0.19001
#     C         -0.09323       -1.02610       -0.32354
#     C         -2.51607       -0.91225        0.11531
#     C         -1.39294       -1.66637        0.03140
#     C          0.86934        1.95516       -0.45831
#     C          1.87687        0.07980        0.59747
#     H         -1.90445        2.47193       -0.43547
#     C          1.13488       -1.88337       -0.49180
#     H         -3.55542        0.86732       -0.53621
#     H         -3.46019       -1.39914        0.35453
#     H         -1.43339       -2.73326        0.24834
#     H          0.72787        2.83770       -1.07845
#     H          1.13491       -2.83617       -1.01618
#     C          2.01724        1.25619       -0.35204
#     C          2.23016       -1.26182       -0.01116
#     H          2.94135        1.49513       -0.87271
#     H          3.24646       -1.64186       -0.07817
#     H          0.03867       -0.37388        1.74557
#     H         -0.22596        1.99145        1.37279
#     H          2.40909        0.26571        1.54100
#     H         -0.24776       -0.45404       -1.25847
# """)

# BASIS_SET = "cc-pVDZ"

# c6h6 = psi4.geometry("""
#      C            0.000000000000    -1.202581711137    -0.694310875655
#      C            0.000000000000    -1.202581711137     0.694310875655
#      C            0.000000000000     0.000000000000     1.388621750949
#      C           -0.000000000000     1.202581711137     0.694310875655
#      C           -0.000000000000     1.202581711137    -0.694310875655
#      C            0.000000000000     0.000000000000    -1.388621750949
#      H            0.000000000000     0.000000000000    -2.470818697970
#      H           -0.000000000000     2.139791761953    -1.235409348982
#      H           -0.000000000000     2.139791761953     1.235409348982
#      H            0.000000000000     0.000000000000     2.470818697970
#      H            0.000000000000    -2.139791761953     1.235409348982
#      H            0.000000000000    -2.139791761953    -1.235409348982
#  """)

# BASIS_SET = "6-31G*"

# sih4 = psi4.geometry("""
# SI
# H             1    4.00
# H             1    1.47      2  109.47
# H             1    1.47      2  109.47      3  120.000000
# H             1    1.47      2  109.47      3 -120.000000
#     """)

# BASIS_SET = "3-21G"

# cdim = psi4.geometry("""
# N
# C     1    1.424500000
# C     2    1.424500000   1    108.0000000
# N     3    1.424500000   2    108.0000000   1    0.000000000
# C     4    1.424500000   3    108.0000000   2    0.000000000
# H     5    1.424500000   4    120.0000000   3    180.0000000
# Cd    4    1.424500000   3    120.0000000   2    180.0000000
# H     3    1.090000000   2    120.0000000   1    180.0000000
# H     2    1.090000000   1    120.0000000   5    180.0000000
# H     1    1.090000000   2    120.0000000   3    180.0000000
# """)

# BASIS_SET = "cc-pVDZ"

# WATER27_OHmH2O4cs = psi4.geometry("""
# -1 1
# O      0.794126907446    -0.158049501951     2.06603946245
# O     -0.0265990025726   -1.46862558046      0.0
# H      0.853543729125     1.31756814974      0.76326952645
# H     -0.0492091726494    0.0556878844845    2.47385636803
# O      0.706019653451     1.9061094238       0.0
# H      0.853543729125     1.31756814974     -0.76326952645
# O      0.794126907446    -0.158049501951    -2.06603946245
# H     -0.0492091726494    0.0556878844845   -2.47385636803
# H      0.525302934075    -0.76678112566     -1.28987424987
# O     -2.03773029565      0.390588797273     0.0
# H     -1.39115337784      1.10999294994      0.0
# H     -1.45458484121     -0.407955345836     0.0
# H      0.525302934075    -0.76678112566      1.28987424987
# H     -0.0434809321712   -2.42696105794      0.0
# """)

# BASIS_SET = "LANL2DZ"
# na8 = psi4.geometry("""
# 0 1
# Na    2.0763096    2.0763096   -2.0763096 
# Na    1.2670936   -1.2670936   -1.2670936 
# Na   -2.0763096   -2.0763096   -2.0763096 
# Na    2.0763096   -2.0763096    2.0763096 
# Na    1.2670936    1.2670936    1.2670936 
# Na   -2.0763096    2.0763096    2.0763096 
# Na   -1.2670936   -1.2670936    1.2670936 
# Na   -1.2670936    1.2670936   -1.2670936 
# """)

# BASIS_SET = "cc-pVDZ"
# o2_minus = psi4.geometry("""
# -1 2
# O  0.0  0.0  0.6
# O  0.0  0.0 -0.6
# """)

# BASIS_SET = "def2-SVP"
# Cu = psi4.geometry("""
# 0 2
# Cu  0.0  0.0  0.0
# """)

#BASIS_SET = "cc-pVDZ"
# N2stretch = psi4.geometry("""
# 0 1
# N 0.0 0.0 -1.5
# N 0.0 0.0  1.5
# """)

# BASIS_SET = "LANL2DZ"
# Cr2Evil = psi4.geometry("""
# 0 1
# Cr 0.0 0.0 -1.68
# Cr 0.0 0.0  1.68
# """)

# BASIS_SET = "LANL2DZ"
# Fe2S2 = psi4.geometry("""
# 0 1
# Fe   0.000000    0.000000    1.340534 
# S   0.000000    1.588225    0.000000 
# Fe   0.000000    0.000000   -1.340534 
# S   0.000000   -1.588225    0.000000
# """)

#BASIS_SET = "DZP"


# BASIS_SET = "LANL2DZ"
# Fe2S2 = psi4.geometry("""
# U    0.00000000  0.00000000  0.00000000
# F   -0.03815118  1.97962841  0.00398177
# F   -1.85358012 -0.69588532  0.01960555
# F    0.96405330 -0.64477254  1.60476466
# F    0.92767800 -0.63897056 -1.62835198
# """)

# test2 = psi4.geometry("""
# 0 1   
# C          0.00000        0.87845        1.79652
# C         -0.76076       -0.43922        1.79652
# H         -1.33206       -0.76905        2.65845
# C          0.76076       -0.43922        1.79652
# C         -1.19981       -0.69271        0.35389
# H          0.00001        1.53812        2.65845
# C          1.19981       -0.69271        0.35389
# H          1.33205       -0.76907        2.65845
# H          0.00000        2.47465        0.24764
# C         -0.76076        0.43922       -1.79652
# H         -2.14311       -1.23733        0.24764
# C          1.19981        0.69271       -0.35389
# C          0.00000       -1.38542       -0.35389
# C          0.76076        0.43922       -1.79652
# H          2.14311       -1.23733        0.24764
# C          0.00000        1.38542        0.35389
# C         -1.19981        0.69271       -0.35389
# C         -0.00000       -0.87845       -1.79652
# H         -2.14311        1.23733       -0.24764
# H         -0.00000       -2.47465       -0.24764
# H          2.14311        1.23733       -0.24764
# H         -0.00001       -1.53812       -2.65845
# H         -1.33205        0.76907       -2.65845
# H          1.33206        0.76905       -2.65845
# """)

# BASIS_SET = "cc-pVDZ"
# CCl4 = psi4.geometry("""
# 0 1
# C 0.0000000 0.0000000 0.0000000
# Cl 0.0000000 0.0000000 1.7673000
# Cl 1.6662300 0.0000000 -0.5891000
# Cl -0.8331100 -1.4429900 -0.5891000
# Cl -0.8331100 1.4429900 -0.5891000
# """)

# BASIS_SET = "LANL2DZ"
# molecule = psi4.geometry("""
# 0 1
# O     0.000000     0.000000     1.361423
# Cu    0.000000    -1.229699     0.000000
# Cu    0.000000     1.229699     0.000000
# O     0.000000     0.000000    -1.361423
# H     0.000000     0.000000    -2.327232
# H     0.000000     0.000000     2.327232
# """)

# done w/ SCAN
MESA_SCF_METHODS = [
    "LISTB",
    "LISTF",
    "FDIIS",
    "EDIIS",    
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

scf_conv_method = "LIST"
#scf_conv_method = "MESA"
#scf_conv_method = "LISTB"

try:
    if scf_conv_method != "MESA":
        run_method(scf_conv_method, do_scf=False, mesa_error="")
        psi4.core.clean()
        print("Run done: completed " + scf_conv_method)
    else:
        #for error_type in ["ZYMETA", "YMETA", "ZMETA", "EMETA","INTRA_D", "INTER_D", "INTER_F", "D_F", "COMMUTATOR"]:
        #for error_type in ["INTER_D", "INTER_F", "D_F", "INTRA_D"]:
        #for error_type in ["INTRA_D"]:
        #for error_type in ["EMETA"]:
        #for error_type in ["YMETA"]:
        for error_type in ["ZYMETA"]:
        #for error_type in ["YZMETA"]:
        #for error_type in ["ZMETA"]:
        #for error_type in ["LMETA"]:
        #for error_type in ["LISTF"]:
        #for error_type in ["EMETA", "INTRA_D"]:
        #for error_type in ["COMMUTATOR"]:
        #for error_type in ["D_F-"]:
        #for error_type in ["EMETA"]:
            run_method("MESA", do_scf=False, mesa_error=error_type)
            psi4.core.clean()
            print("Run done: completed " + scf_conv_method + " with error type " + error_type)

except Exception as e:
    print(e)
    print("Run done: failed to converge " + scf_conv_method)
