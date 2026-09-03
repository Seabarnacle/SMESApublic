#!/usr/bin/env python3
"""Generate molecular renderings plus TeX and PDF geometry supplements.

Run this after exposing the staged modified-Psi4 build to Python, as described
in this directory's README. The molecular coordinates are read only from the
case input files; calculation result JSON files are never opened.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import psi4


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "geometry_figures"
TEX_PATH = ROOT / "SMESA_geometries.tex"
PDF_PATH = ROOT / "SMESA_geometries.pdf"
BOHR_TO_ANGSTROM = 0.529177210903

CASE_ORDER = [
    "C4H4square",
    "Co2",
    "Cr2",
    "Cr3_linear",
    "Fe2S2",
    "Mn2_stretched",
    "Os3CO9",
    "Rh4LCO",
    "Rh4_square",
    "UCl6",
    "V3_Scalene",
    "cdim",
    "square_H4",
    "stretched_N2",
]

DISPLAY_NAMES = {
    "C4H4square": "C4H4 square",
    "Co2": "Co2",
    "Cr2": "Cr2",
    "Cr3_linear": "Cr3 linear",
    "Fe2S2": "Fe2S2",
    "Mn2_stretched": "Mn2 stretched",
    "Os3CO9": "Os3CO9",
    "Rh4LCO": "Rh4LCO",
    "Rh4_square": "Rh4 square",
    "UCl6": "UCl6",
    "V3_Scalene": "V3 scalene",
    "cdim": "cdim",
    "square_H4": "H4 square",
    "stretched_N2": "N2 stretched",
}

COLORS = {
    "H": "#e8e8e8",
    "C": "#4b4b4b",
    "N": "#3159d8",
    "O": "#e31a1c",
    "S": "#e4d900",
    "Cl": "#43aa4b",
    "V": "#7586ae",
    "Cr": "#8c68bb",
    "Mn": "#d8879d",
    "Fe": "#bd5527",
    "Co": "#2651db",
    "Rh": "#087b86",
    "Cd": "#d7b970",
    "Os": "#236889",
    "U": "#77933c",
}

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "Cl": 1.02,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.24,
    "Co": 1.26,
    "Rh": 1.42,
    "Cd": 1.44,
    "Os": 1.44,
    "U": 1.96,
}

ATOM_SIZES = {
    "H": 420,
    "C": 760,
    "N": 800,
    "O": 790,
    "S": 980,
    "Cl": 1050,
    "V": 1200,
    "Cr": 1240,
    "Mn": 1240,
    "Fe": 1190,
    "Co": 1200,
    "Rh": 1260,
    "Cd": 1320,
    "Os": 1320,
    "U": 1500,
}

MANUAL_BONDS = {
    "C4H4square": [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7)],
    "Co2": [(0, 1)],
    "Cr2": [(0, 1)],
    "Cr3_linear": [(0, 1), (1, 2)],
    "Fe2S2": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "Mn2_stretched": [(0, 1)],
    "Os3CO9": [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (1, 5), (5, 6), (2, 7), (7, 8)],
    "Rh4LCO": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (4, 5)],
    "Rh4_square": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "UCl6": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)],
    "V3_Scalene": [(0, 1), (1, 2), (2, 0)],
    "cdim": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (4, 5), (3, 6), (2, 7), (1, 8), (0, 9)],
    "square_H4": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "stretched_N2": [(0, 1)],
}


def setting(text: str, name: str, default: str) -> tuple[str, bool]:
    match = re.search(
        rf"^\s*{re.escape(name)}\s*=\s*[\"']([^\"']+)[\"']",
        text,
        flags=re.MULTILINE,
    )
    if match:
        return match.group(1), False
    return default, True


def formula(symbols: list[str]) -> str:
    counts = Counter(symbols)
    parts = []
    for symbol in dict.fromkeys(symbols):
        count = counts[symbol]
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def load_case(case_name: str) -> dict:
    input_path = ROOT / case_name / f"{case_name}.txt"
    text = input_path.read_text(encoding="utf-8")
    geometry_match = re.search(
        r"psi4\.geometry\(\s*\"\"\"(.*?)\"\"\"\s*\)",
        text,
        flags=re.DOTALL,
    )
    if not geometry_match:
        raise ValueError(f"No psi4.geometry block found in {input_path}")

    molecule = psi4.geometry(geometry_match.group(1))
    molecule.update_geometry()
    symbols = [molecule.symbol(index).title() for index in range(molecule.natom())]
    coordinates = molecule.geometry().to_array(dense=True) * BOHR_TO_ANGSTROM

    basis, basis_default = setting(text, "BASIS_SET", "cc-pVDZ")
    guess, guess_default = setting(text, "GUESS", "CORE")
    scf_type, scf_type_default = setting(text, "SCF_TYPE", "PK")
    reference, reference_default = setting(text, "REFERENCE", "RHF")
    dft, dft_default = setting(text, "DFT", "HF")

    return {
        "case": case_name,
        "name": DISPLAY_NAMES[case_name],
        "input_path": input_path.relative_to(ROOT),
        "symbols": symbols,
        "coordinates": coordinates,
        "formula": formula(symbols),
        "charge": int(round(molecule.molecular_charge())),
        "multiplicity": molecule.multiplicity(),
        "basis": basis,
        "basis_default": basis_default,
        "guess": guess.upper(),
        "guess_default": guess_default,
        "scf_type": scf_type.upper(),
        "scf_type_default": scf_type_default,
        "reference": reference.upper(),
        "reference_default": reference_default,
        "method": dft if not dft_default else "HF",
        "method_default": dft_default,
    }


def inferred_bonds(symbols: list[str], coordinates: np.ndarray) -> list[tuple[int, int]]:
    bonds = []
    for left in range(len(symbols)):
        for right in range(left + 1, len(symbols)):
            distance = np.linalg.norm(coordinates[left] - coordinates[right])
            cutoff = 1.22 * (
                COVALENT_RADII.get(symbols[left], 0.8)
                + COVALENT_RADII.get(symbols[right], 0.8)
            )
            if 0.2 < distance <= cutoff:
                bonds.append((left, right))
    return bonds


def projected_coordinates(case: dict) -> tuple[np.ndarray, np.ndarray]:
    centered = case["coordinates"] - np.mean(case["coordinates"], axis=0)
    if case["case"] == "UCl6":
        right = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
        up = np.array([1.0, 1.0, -2.0]) / math.sqrt(6.0)
        depth_axis = np.cross(right, up)
        return centered @ np.column_stack((right, up)), centered @ depth_axis
    if case["case"] == "Rh4LCO":
        right = np.array([1.0, 0.0, 0.35])
        right /= np.linalg.norm(right)
        up = np.array([0.0, 1.0, 0.35])
        up -= np.dot(up, right) * right
        up /= np.linalg.norm(up)
        depth_axis = np.cross(right, up)
        return centered @ np.column_stack((right, up)), centered @ depth_axis

    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    right = right_vectors[0]
    up = right_vectors[1] if len(right_vectors) > 1 else np.array([0.0, 1.0, 0.0])
    depth_axis = right_vectors[2] if len(right_vectors) > 2 else np.cross(right, up)
    projected = centered @ np.column_stack((right, up))
    depth = centered @ depth_axis

    if np.ptp(projected[:, 0]) < np.ptp(projected[:, 1]):
        projected = projected[:, ::-1]
    return projected, depth


def render_case(case: dict) -> Path:
    projected, depth = projected_coordinates(case)
    bonds = MANUAL_BONDS.get(
        case["case"], inferred_bonds(case["symbols"], case["coordinates"])
    )

    figure_path = FIGURE_DIR / f"{case['case']}.png"
    figure, axis = plt.subplots(figsize=(8.4, 4.8), dpi=200)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    for left, right in sorted(bonds, key=lambda pair: float(np.mean(depth[list(pair)]))):
        x_values = projected[[left, right], 0]
        y_values = projected[[left, right], 1]
        midpoint = (projected[left] + projected[right]) / 2.0
        axis.plot(x_values, y_values, color="#343a40", linewidth=13, solid_capstyle="round", zorder=1)
        axis.plot(
            [projected[left, 0], midpoint[0]],
            [projected[left, 1], midpoint[1]],
            color=COLORS.get(case["symbols"][left], "#777777"),
            linewidth=8,
            solid_capstyle="round",
            zorder=2,
        )
        axis.plot(
            [midpoint[0], projected[right, 0]],
            [midpoint[1], projected[right, 1]],
            color=COLORS.get(case["symbols"][right], "#777777"),
            linewidth=8,
            solid_capstyle="round",
            zorder=2,
        )
        axis.plot(x_values, y_values, color="white", alpha=0.22, linewidth=2, zorder=3)

    atom_order = np.argsort(depth)
    extent = max(float(np.ptp(projected[:, 0])), float(np.ptp(projected[:, 1])), 1.0)
    highlight_shift = 0.035 * extent
    for index in atom_order:
        symbol = case["symbols"][index]
        color = COLORS.get(symbol, "#888888")
        size = ATOM_SIZES.get(symbol, 850)
        x_value, y_value = projected[index]
        axis.scatter(
            [x_value],
            [y_value],
            s=size,
            color=color,
            edgecolor="#30343b",
            linewidth=0.8,
            zorder=5,
        )
        axis.scatter(
            [x_value - highlight_shift],
            [y_value + highlight_shift],
            s=size * 0.12,
            color="white",
            alpha=0.72,
            linewidth=0,
            zorder=6,
        )
        label_color = "#202020" if symbol == "H" else "white"
        axis.text(
            x_value,
            y_value,
            str(index + 1),
            ha="center",
            va="center",
            color=label_color,
            fontsize=7.5,
            weight="bold",
            zorder=7,
        )

    x_span = max(float(np.ptp(projected[:, 0])), 1.0)
    y_span = max(float(np.ptp(projected[:, 1])), 0.75)
    axis.set_xlim(float(np.min(projected[:, 0])) - 0.16 * x_span, float(np.max(projected[:, 0])) + 0.16 * x_span)
    axis.set_ylim(float(np.min(projected[:, 1])) - 0.24 * y_span, float(np.max(projected[:, 1])) + 0.24 * y_span)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    figure.savefig(figure_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return figure_path


def default_suffix(is_default: bool) -> str:
    return " (default)" if is_default else ""


def pdf_page(pdf: PdfPages, case: dict, figure_path: Path, page_number: int) -> None:
    page = plt.figure(figsize=(11.0, 8.5), facecolor="white")
    page.text(0.055, 0.935, case["name"], fontsize=23, weight="bold", color="#17365d")
    page.text(
        0.945,
        0.94,
        f"Geometry {page_number} of {len(CASE_ORDER)}",
        ha="right",
        fontsize=9,
        color="#68717a",
    )
    page.add_artist(plt.Line2D([0.055, 0.945], [0.91, 0.91], color="#4472c4", linewidth=2.2))

    image_axis = page.add_axes([0.055, 0.30, 0.47, 0.56])
    image_axis.imshow(mpimg.imread(figure_path))
    image_axis.axis("off")

    settings_axis = page.add_axes([0.56, 0.66, 0.38, 0.20])
    settings_axis.axis("off")
    settings = [
        ("Input", str(case["input_path"])),
        ("Atoms", f"{case['formula']} ({len(case['symbols'])} atoms)"),
        ("Charge / multiplicity", f"{case['charge']} / {case['multiplicity']}"),
        ("Method / reference", f"{case['method']} / {case['reference']}{default_suffix(case['reference_default'])}"),
        ("Basis", f"{case['basis']}{default_suffix(case['basis_default'])}"),
        ("Initial guess", f"{case['guess']}{default_suffix(case['guess_default'])}"),
        ("SCF type", f"{case['scf_type']}{default_suffix(case['scf_type_default'])}"),
    ]
    table = settings_axis.table(
        cellText=settings,
        colWidths=[0.38, 0.62],
        cellLoc="left",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2f3")
        cell.set_linewidth(0.7)
        cell.set_facecolor("#f6f8fc" if row % 2 == 0 else "white")
        if column == 0:
            cell.set_text_props(weight="bold", color="#17365d")

    coordinate_axis = page.add_axes([0.56, 0.265, 0.38, 0.34])
    coordinate_axis.axis("off")
    coordinate_axis.text(
        0.0,
        1.06,
        "Cartesian geometry (angstrom)",
        fontsize=12,
        weight="bold",
        color="#17365d",
        transform=coordinate_axis.transAxes,
    )
    rows = [
        [str(index + 1), symbol, f"{xyz[0]: .6f}", f"{xyz[1]: .6f}", f"{xyz[2]: .6f}"]
        for index, (symbol, xyz) in enumerate(zip(case["symbols"], case["coordinates"]))
    ]
    coordinates_table = coordinate_axis.table(
        cellText=rows,
        colLabels=["#", "Atom", "x", "y", "z"],
        colWidths=[0.08, 0.14, 0.26, 0.26, 0.26],
        cellLoc="right",
        bbox=[0, 0, 1, 1],
    )
    coordinates_table.auto_set_font_size(False)
    coordinates_table.set_fontsize(8.6 if len(rows) <= 9 else 7.8)
    for (row, column), cell in coordinates_table.get_celld().items():
        cell.set_edgecolor("#d9e2f3")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#dbe5f1")
            cell.set_text_props(weight="bold", color="#17365d", ha="center")
        elif column in (0, 1):
            cell.set_text_props(ha="center")

    page.text(
        0.055,
        0.075,
        "Atom numbers in the rendering correspond to the coordinate table. "
        "Default labels identify settings supplied by metarunner.py when absent from the input.",
        fontsize=8.6,
        color="#5b6570",
    )
    pdf.savefig(page, bbox_inches="tight")
    plt.close(page)


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_tex(cases: list[dict]) -> None:
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[letterpaper,landscape,margin=0.55in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{array}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\definecolor{smesablue}{HTML}{17365D}",
        r"\definecolor{smesalight}{HTML}{F1F5FA}",
        r"\pagestyle{empty}",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\LARGE\bfseries\color{smesablue} SMESA calculation geometries}\par\vspace{0.35in}",
        r"{\large Supporting information for \emph{Secondary Minimal Error Sampling Algorithms for Accelerating Self-Consistent Field Calculations}}\par\vspace{0.2in}",
        r"Paul Calistrate-Petre and Yan Alexander Wang\par\vspace{0.5in}",
        r"\begin{minipage}{0.78\textwidth}\large",
        r"This document records the Cartesian geometry and starting calculation settings for every case input in \texttt{mesa/SMESACASES}. Coordinates are in angstrom. Inputs supplied as Z-matrices were converted to the matching Cartesian representation by Psi4. Values marked ``default'' are supplied by \texttt{metarunner.py} when the case file does not override them.",
        r"\end{minipage}",
        r"\end{center}",
        r"\clearpage",
    ]

    for index, case in enumerate(cases, start=1):
        image_name = f"geometry_figures/{case['case']}.png"
        settings = [
            ("Input", str(case["input_path"])),
            ("Atoms", f"{case['formula']} ({len(case['symbols'])} atoms)"),
            ("Charge / multiplicity", f"{case['charge']} / {case['multiplicity']}"),
            ("Method / reference", f"{case['method']} / {case['reference']}{default_suffix(case['reference_default'])}"),
            ("Basis", f"{case['basis']}{default_suffix(case['basis_default'])}"),
            ("Initial guess", f"{case['guess']}{default_suffix(case['guess_default'])}"),
            ("SCF type", f"{case['scf_type']}{default_suffix(case['scf_type_default'])}"),
        ]
        lines.extend(
            [
                rf"{{\LARGE\bfseries\color{{smesablue}} {tex_escape(case['name'])}}}\hfill Geometry {index} of {len(cases)}\par",
                r"\vspace{0.08in}\color{smesablue}\hrule\color{black}\vspace{0.15in}",
                r"\begin{minipage}[t]{0.48\textwidth}",
                rf"\centering\includegraphics[width=\linewidth,height=5.7in,keepaspectratio]{{\detokenize{{{image_name}}}}}",
                r"\end{minipage}\hfill",
                r"\begin{minipage}[t]{0.49\textwidth}",
                r"\rowcolors{1}{smesalight}{white}",
                r"\begin{tabular}{>{\bfseries\color{smesablue}}p{1.65in}p{3.15in}}",
            ]
        )
        for label, value in settings:
            lines.append(rf"{tex_escape(label)} & {tex_escape(value)} \\")
        lines.extend(
            [
                r"\end{tabular}",
                r"\vspace{0.18in}",
                r"{\large\bfseries\color{smesablue} Cartesian geometry (angstrom)}\par\vspace{0.06in}",
                r"\rowcolors{2}{smesalight}{white}",
                r"\begin{tabular}{r c r r r}",
                r"\toprule",
                r"\# & Atom & $x$ & $y$ & $z$ \\",
                r"\midrule",
            ]
        )
        for atom_index, (symbol, xyz) in enumerate(zip(case["symbols"], case["coordinates"]), start=1):
            lines.append(
                rf"{atom_index} & {tex_escape(symbol)} & {xyz[0]: .6f} & {xyz[1]: .6f} & {xyz[2]: .6f} \\"
            )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{minipage}",
                r"\vfill",
                r"\footnotesize Atom numbers in the rendering correspond to the coordinate table.",
                r"\clearpage",
            ]
        )

    lines.extend([r"\end{document}", ""])
    TEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_pdf(cases: list[dict], figure_paths: list[Path]) -> None:
    metadata = {
        "Title": "SMESA calculation geometries",
        "Author": "Paul Calistrate-Petre and Yan Alexander Wang",
        "Subject": "Geometry and starting calculation settings",
    }
    with PdfPages(PDF_PATH, metadata=metadata) as pdf:
        cover = plt.figure(figsize=(11.0, 8.5), facecolor="white")
        cover.text(0.5, 0.70, "SMESA calculation geometries", ha="center", fontsize=29, weight="bold", color="#17365d")
        cover.text(
            0.5,
            0.60,
            "Supporting information for\nSecondary Minimal Error Sampling Algorithms for Accelerating Self-Consistent Field Calculations",
            ha="center",
            va="center",
            fontsize=15,
            linespacing=1.5,
        )
        cover.text(0.5, 0.48, "Paul Calistrate-Petre and Yan Alexander Wang", ha="center", fontsize=13)
        cover.text(
            0.5,
            0.31,
            "Cartesian coordinates and starting calculation settings\nCoordinates in angstrom",
            ha="center",
            fontsize=12,
            color="#5b6570",
            linespacing=1.5,
        )
        cover.add_artist(plt.Line2D([0.23, 0.77], [0.42, 0.42], color="#4472c4", linewidth=2.5))
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        for index, (case, figure_path) in enumerate(zip(cases, figure_paths), start=1):
            pdf_page(pdf, case, figure_path, index)


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    cases = [load_case(case_name) for case_name in CASE_ORDER]
    figure_paths = [render_case(case) for case in cases]
    write_tex(cases)
    write_pdf(cases, figure_paths)
    print(f"Wrote {len(cases)} geometry figures to {FIGURE_DIR}")
    print(f"Wrote {TEX_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
