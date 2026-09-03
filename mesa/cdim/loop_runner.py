import os

eval_methods = [
    # "LIST3",
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

for method in eval_methods:
    os.system(f'python test.py {method}')
