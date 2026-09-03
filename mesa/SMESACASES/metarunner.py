import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import glob
import psi4
from psi4 import SCFConvergenceError


def csv_environment_setting(name, default):
    """Return a comma-separated environment setting as a list."""
    value = os.environ.get(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


SMESA_THREADS = int(os.environ.get("SMESA_THREADS", "16"))
SMESA_MEMORY = os.environ.get("SMESA_MEMORY", "16 GB")

psi4.set_num_threads(SMESA_THREADS)

# --- Global Settings ---
psi4.set_memory(SMESA_MEMORY)

MESA_SCF_METHODS = [
    "LISTB",
    "LISTF",
    "FDIIS",
    "EDIIS",    
    "ADIIS",
    "DIIS",
    "LISTR",
]

# --- Filtering Settings ---
# Uncomment to ONLY run calculations on these specific folders
#INCLUDE_FOLDERS = ["W4F2","Pd4F2", "Cr4NO", "Mn4O2"]
#INCLUDE_FOLDERS = ["Be2","H4line","LiF","Ni2"]
#INCLUDE_FOLDERS = ["Mn4O2", "Ni2","CrMo","H3F3","C5H5+"]
#INCLUDE_FOLDERS = ["C4H4square","cdim","square_H4","Co2"]
#INCLUDE_FOLDERS = ["Ir4LNO"]
#INCLUDE_FOLDERS = ["Cr4NO"]
#INCLUDE_FOLDERS = ["Cr2"]
# Uncomment to SKIP calculations on these specific folders
#EXCLUDE_FOLDERS = ["Rh4LCO", "Cr2", "Fe2S2"]
#INCLUDE_FOLDERS = ["Pb4","Sn2","SnCl2_NH3","Pd4LCN"]
#INCLUDE_FOLDERS = ["Cr3_Scalene","Fe2C2","Th2","Y3_Anion"]
#INCLUDE_FOLDERS = ["V3_Scalene"]
#INCLUDE_FOLDERS = ["Fe4","H10_chain","H4_square","Mo2_compressed","Cu3_JT","Fe4_tetra","Mn2_stretched","Co2","Rh3CO3","V4O4"]
#INCLUDE_FOLDERS = ["Mn2_stretched"]
#INCLUDE_FOLDERS = ["Rh3CO3","Cr3_linear","W2","Rh4_square","Os3CO9"]
#INCLUDE_FOLDERS = ["Cr3_linear","Rh4_square","Cr2","Fe2S2","Mn2_stretched","Os3CO9","V3_Scalene","Rh4LCO"]
configured_cases = os.environ.get("SMESA_CASES")
if configured_cases and configured_cases.strip().upper() == "ALL":
    INCLUDE_FOLDERS = None
else:
    INCLUDE_FOLDERS = csv_environment_setting("SMESA_CASES", ["Os3CO9"])


def run_method(method, do_scf, mesa_error, basis_set, custom_guess=None, custom_scf_type=None, dft_functional=None, custom_reference=None):
    default_options = {
        "basis": basis_set,
        "SCF_INITIAL_ACCELERATOR": "NONE",
        "LIST": "NONE",
        "DIIS": False,
        "MESA": False,
        "GUESS": custom_guess if custom_guess else "CORE",
        "SCF_TYPE": custom_scf_type if custom_scf_type else "PK",
        "REFERENCE": custom_reference if custom_reference else "RHF",
        "MAXITER": 200,
        #"DIIS_MAX_VECS": 15
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

    print(f"Options for this run: {default_options}")
    psi4.set_options(default_options)
    
    # Run DFT if a functional was specified, otherwise default to standard SCF
    if dft_functional:
        print(f"Executing DFT energy calculation with functional: {dft_functional}")
        psi4.energy(dft_functional)
    else:
        psi4.energy("scf")


# --- Main Execution Loop ---
def process_cases():
    # Automatically get the directory where metarunner.py is located
    cases_dir = os.path.dirname(os.path.abspath(__file__))
    error_types = csv_environment_setting(
        "SMESA_ERROR_TYPES", ["ZYMETA", "YMETA"]
    )
    #error_types = ["D_F-"]
    #error_types = ["ZYMETA", "YMETA"]
    #error_types = ["INTRA_D"]
    #error_types = ["INTER_D", "INTER_F", "D_F", "D_F-"]
    #error_types = ["ZYMETA", "EMETA", "INTRA_D", "INTER_F", "COMMUTATOR","INTER_D", "D_F-"]

    for root, dirs, files in os.walk(cases_dir):
        if root == cases_dir:
            continue
            
        geometry_name = os.path.basename(root)
        
        # --- DEEP CLEAN STATE FOR FRESH RUN ---
        # Wipes all inherited global options and variables from the previous folder
        psi4.core.clean_options()
        psi4.core.clean_variables()
        psi4.core.clean()
        psi4.set_memory(SMESA_MEMORY)
        
        # --- Filtering Logic ---
        if INCLUDE_FOLDERS is not None and geometry_name not in INCLUDE_FOLDERS:
            continue
            
        if 'EXCLUDE_FOLDERS' in globals() and geometry_name in EXCLUDE_FOLDERS:
            continue
        # -----------------------
        
        txt_files = glob.glob(os.path.join(root, "*.txt"))
        
        if not txt_files:
            print(f"No text files found in {geometry_name}. Skipping.")
            continue
            
        geom_file = txt_files[0]
        
        # Re-initialize variables to None before parsing the new text file
        basis_set = "cc-pVDZ" 
        custom_guess = None
        custom_scf_type = None
        dft_functional = None
        custom_reference = None
        
        geom_lines = []
        in_geom = False
        
        with open(geom_file, 'r') as f:
            for line in f:
                line = line.lstrip("#").strip()
                if not line:
                    continue
                
                if "BASIS_SET" in line:
                    parts = line.split("=")
                    if len(parts) > 1:
                        basis_set = parts[1].replace('"', '').replace("'", "").strip()
                elif "GUESS" in line:
                    parts = line.split("=")
                    if len(parts) > 1:
                        custom_guess = parts[1].replace('"', '').replace("'", "").strip()
                elif "SCF_TYPE" in line:
                    parts = line.split("=")
                    if len(parts) > 1:
                        custom_scf_type = parts[1].replace('"', '').replace("'", "").strip()
                elif "DFT" in line:
                    parts = line.split("=")
                    if len(parts) > 1:
                        dft_functional = parts[1].replace('"', '').replace("'", "").strip()
                elif "REFERENCE" in line:
                    parts = line.split("=")
                    if len(parts) > 1:
                        custom_reference = parts[1].replace('"', '').replace("'", "").strip()
                        
                elif 'psi4.geometry' in line:
                    in_geom = True
                elif in_geom:
                    if '""")' in line or '"""' in line:
                        in_geom = False
                    else:
                        geom_lines.append(line)
                        
        geom_string = "\n".join(geom_lines)
        
        if not geom_string:
            print(f"Could not parse geometry in {geom_file}. Skipping.")
            continue

        try:
            mol = psi4.geometry(geom_string)
        except Exception as e:
            print(f"Failed to load geometry for {geometry_name}: {e}")
            continue

        original_cwd = os.getcwd()
        os.chdir(root)
        
        for error_type in error_types:
            print(f"\n--- Running MESA with {error_type} on {geometry_name} ---")
            try:
                run_method(
                    "MESA", 
                    do_scf=False, 
                    mesa_error=error_type, 
                    basis_set=basis_set,
                    custom_guess=custom_guess,
                    custom_scf_type=custom_scf_type,
                    dft_functional=dft_functional,
                    custom_reference=custom_reference
                )
                psi4.core.clean()
            except SCFConvergenceError as e:
                print(f"Convergence error on {error_type}: {e}")
            except Exception as e:
                print(f"Unexpected error on {error_type}: {e}")
                
        os.chdir(original_cwd)

if __name__ == "__main__":
    process_cases()
