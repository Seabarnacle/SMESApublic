# Supporting information: SMESA calculations

This directory contains the calculation inputs and iteration traces supporting
**“Secondary Minimal Error Sampling Algorithms for Accelerating
Self-Consistent Field Calculations”** by Paul Calistrate-Petre and Yan Alexander
Wang. It is self-contained supporting-information (SI) entry point; the
repository-level README describes the Psi4 project rather than this
study.

The calculations use a modified **Psi4 1.9** source tree. The source snapshot in
this repository is authoritative implementation of SMESA. 

## Directory contents

Each named subdirectory is one molecular test case. Its contents are:

- `<case>.txt`: charge, multiplicity, geometry, and optional calculation
  settings read by `metarunner.py`. Geometries use angstroms (Psi4's default
  molecular-coordinate unit where no explicit `units` line is present).
- `mesa*.json`: archived per-iteration results for a MESA or SMESA variant.
- `diis.json`, `adiis.json`, and similarly named files, where present: baseline
  SCF-accelerator traces in the same JSON format.
- `<case>_convergence.png`: a plot generated from the JSON `RMS` arrays by
  `plotter.py`. These images are derived artifacts; the JSON is the numerical
  record.
  Also plots made by plotter.py are rough and not made using same code as in-text plots. 
- `SMESA_geometries.pdf`: illustrated geometry supplement covering every case
  input, including Cartesian coordinates, charge, multiplicity, method,
  reference, basis, initial guess, and SCF type. It contains no energy values.
- `SMESA_geometries.tex`: editable LaTeX source for the same geometry
  supplement. The rendered molecule panels are in `geometry_figures/`.
- `make_geometry_si.py`: regenerates the geometry figures, PDF, and TeX source
  directly from the case `.txt` inputs.

Runtime scratch misc (`psi.*.clean`), `timer.dat`, local logs, and ad-hoc
files named `* copy.*` are intentionally excluded. 

## Build the modified Psi4 source

From the repository root, create the development environment and a new build
directory. The exact CMake generator may be changed to suit the local system.

```bash
conda env create -f env_p4dev.yaml
conda activate p4dev
eval "$(conda/psi4-path-advisor.py cmake --objdir objdir_smesa)"
```

The path advisor generates a new dependency cache from the active environment,
configures the source, and builds it. Expose the staged build to Python and
verify that Python imports this source build rather than another Psi4
installation:

```bash
eval "$(objdir_smesa/stage/bin/psi4 --psiapi)"
python -c 'import psi4; print(psi4.__version__); print(psi4.__file__)'
```

The development environment supplies NumPy and SciPy. Install Matplotlib in
that environment if convergence plots or the geometry supplement are to be
regenerated.

```bash
conda install -c conda-forge matplotlib
```

## Reproduce calculations with `metarunner.py`

Run the driver from this directory. The following example reruns the principal
published variants for one case using four Psi4 threads and 8 GB of memory:

```bash
cd mesa/SMESACASES
SMESA_CASES=Cr3_linear \
SMESA_ERROR_TYPES=INTRA_D,COMMUTATOR,INTER_D,INTER_F,EMETA,ZMETA,YMETA,ZYMETA \
SMESA_THREADS=4 \
SMESA_MEMORY="8 GB" \
python metarunner.py
```

`SMESA_CASES` is a comma-separated list of case-directory names. Set it to
`ALL` to process every subdirectory containing a `.txt` input. The runner's
default is the focused `Os3CO9` run with `ZYMETA` and `YMETA` when these
environment settings are omitted.

`SMESA_ERROR_TYPES` is a comma-separated list of the internal identifiers in
the table below. Calculations are serial across cases and variants. Existing
JSON with the same name can be overwritten, so work on a copy when the
archived traces must be preserved byte-for-byte.

The runner reads these optional assignments from each case's `.txt` input:
`BASIS_SET`, `GUESS`, `SCF_TYPE`, `DFT`, and `REFERENCE`. Defaults are
`cc-pVDZ`, `CORE`, `PK`, no DFT functional (SCF/HF), and `RHF`, respectively.
It then constructs the embedded `psi4.geometry(...)`, resets Psi4 state between
cases, and runs up to 200 SCF iterations. The candidate accelerator pool is
`LISTB`, `LISTF`, `FDIIS`, `EDIIS`, `ADIIS`, `DIIS`, and `LISTR`.

The study treats a calculation as converged only when both the ordinary Psi4
SCF commutator residual and the absolute change in total energy are below
`1e-6`. A run that reaches the 200-iteration cutoff is recorded as
nonconverged. Owing to the development loop's bookkeeping, some archived
nonconverged JSON traces contain 201 aligned records; this does not denote a
different convergence cutoff.

Regenerate all plots after the calculations with:

```bash
python plotter.py
```

The plotting script uses a noninteractive backend, reads whichever recognized
`mesa*.json` files exist in each case, and replaces that case's convergence
PNG.

## Regenerate the geometry supplement

After exposing the staged Psi4 build to Python as described above, run the
geometry-document generator from this directory:

```bash
python make_geometry_si.py
```

The script reads all case `.txt` files, uses Psi4 to convert any Z-matrix input
to Cartesian coordinates in angstrom, and writes `geometry_figures/`,
`SMESA_geometries.tex`, and `SMESA_geometries.pdf`. It never reads calculation
result JSON. If a TeX installation is available, the editable source can also
be compiled from this directory with:

```bash
pdflatex SMESA_geometries.tex
```

## Internal names and paper labels

The JSON filename is formed from the internal option by converting it to lower
case and prefixing `mesa`.

| Internal option | Result filename | Label used in the paper/SI | Meaning |
| --- | --- | --- | --- |
| `INTRA_D` | `mesaintra_d.json` | MESA (INTRA-D) | Single-diagnostic MESA using the within-cycle density change. |
| `COMMUTATOR` | `mesacommutator.json` | MESA (COMM) | Single-diagnostic MESA using the generalized SCF commutator. |
| `INTER_D` | `mesainter_d.json` | MESA (INTER-D) | Single-diagnostic MESA using density change between cycles. |
| `INTER_F` | `mesainter_f.json` | MESA (INTER-F) | Single-diagnostic MESA using Fock change between cycles. |
| `EMETA` | `mesaemeta.json` | **SMESAv** | Normalized consensus-voting selector, using INTRA-D and COMM. |
| `ZMETA` | `mesazmeta.json` | **SMESAm** | Metric-space least-squares selector; uses INTRA-D, COMM, and, above the switching threshold, INTER-F. |
| `YMETA` | `mesaymeta.json` | **SMESAf** | Weighted hybridization of sampled Fock matrices for the COMM target diagnostic. |
| `ZYMETA` | `mesazymeta.json` | **SMESAc** | Multi-method, multi-diagnostic Fock hybrid. |

COMM abbreviates the generalized commutator
`F D S - S D F`. INTRA-D is the Frobenius norm of the difference between the
output and input densities for a candidate within one SCF cycle. INTER-D is the
Frobenius norm of the candidate output density minus the preceding accepted
output density. INTER-F is the Frobenius norm of the candidate input Fock
matrix minus the preceding accepted output Fock matrix. SMESAm and SMESAc omit
INTER-F after the ordinary convergence error falls below `1e-3`, as described
in the manuscript.

## JSON schema and units

Every archived JSON object has exactly three same-length arrays:

| Field | Type | Definition and units |
| --- | --- | --- |
| `RMS` | array of numbers | Standard Psi4 RMS SCF generalized-commutator/orbital-gradient residual for each iteration, in Psi4's atomic-unit convention. This is the common convergence trace, not necessarily the diagnostic named by the filename. |
| `ENERGIES` | array of numbers | Total SCF energy after each iteration, in hartree (`E_h`). |
| `METHODS` | array of arrays of strings | Unitless internal status/method labels for each iteration. For selectors these identify the chosen candidate method; hybrid variants may report their internal hybrid label rather than mixture weights. |

Array element zero corresponds to SCF iteration 1. The three arrays therefore
provide aligned iteration records. Small last-iteration differences between
variants can reflect convergence to distinct SCF stationary points and should
not automatically be interpreted as serialization or unit errors.
