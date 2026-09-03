import os
import json
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DISPLAY_LABELS = {
    "ZYMETA": "SMESAc",
    "YMETA": "SMESAf",
    "ZMETA": "SMESAm",
    "EMETA": "SMESAv",
    "INTRA_D": "MESA (INTRA-D)",
    "INTER_D": "MESA (INTER-D)",
    "INTER_F": "MESA (INTER-F)",
    "D_F": "MESA (D-F)",
    "D_F-": "MESA (D-F-minus)",
    "COMMUTATOR": "MESA (COMM)",
}


def plot_cases():
    cases_dir = os.path.dirname(os.path.abspath(__file__))
    error_types = ["ZYMETA", "YMETA", "ZMETA", "EMETA", "INTRA_D", "INTER_D", "INTER_F", "D_F", "D_F-", "COMMUTATOR"]
    #error_types = ["ZYMETA", "ZMETA"]

    for root, dirs, files in os.walk(cases_dir):
        if root == cases_dir:
            continue
            
        geometry_name = os.path.basename(root)
        print(f"Plotting folder: {geometry_name}")
        
        plt.figure(figsize=(10, 8))
        
        # 'o'=circle, 's'=square, '^'=triangle_up, 'D'=diamond, 'v'=triangle_down, 
        # 'p'=pentagon, '*'=star, 'h'=hexagon, 'X'=cross
        markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X'])
        
        # '-'=solid, '--'=dashed, '-.'=dash-dot, ':'=dotted
        linestyles = itertools.cycle(['-', '--', '-.', ':'])
        
        plotted_anything = False
        
        for err in error_types:
            json_file = os.path.join(root, f"mesa{err.lower()}.json")
            
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        
                    if "RMS" in data:
                        rms = data["RMS"]
                        
                        current_marker = next(markers)
                        current_linestyle = next(linestyles)
                        
                        plt.plot(
                            range(1, len(rms) + 1), 
                            rms, 
                            label=DISPLAY_LABELS.get(err, err),
                            marker=current_marker, 
                            linestyle=current_linestyle,
                            markersize=5,
                            linewidth=1.5
                        )
                        plotted_anything = True
                except Exception as e:
                    print(f"Failed to read/plot {json_file}: {e}")
                    
        if plotted_anything:
            plt.yscale('log')
            plt.xlabel('SCF Iteration', fontsize=12)
            plt.ylabel('RMS Error', fontsize=12)
            plt.title(f'SCF Convergence for {geometry_name}', fontsize=14, fontweight='bold')
            plt.legend(loc='upper right', frameon=True)
            plt.grid(True, which="both", ls="--", alpha=0.5)
            
            save_path = os.path.join(root, f"{geometry_name}_convergence.png")
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
            
        plt.close() 

if __name__ == "__main__":
    plot_cases()
