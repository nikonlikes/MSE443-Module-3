"""
MSE 433 - Module 3: Analysis of Greedy Tote Simulation Results

Reads greedy/outputs/simulation_results.csv (produced by greedy_sim.py) and
computes summary statistics by size (small, medium, large) and method (greedy, baseline):
  - Average time to complete (avg completion time per order)
  - Average makespan
  - Average sum of completion times
  - Count of completed runs (finite makespan) vs total
  - Min, median, max where applicable

Run after greedy_sim.py. If simulation_results.csv is missing, run:
  python greedy_sim.py
"""

import csv
import os
import statistics

# Path to results file (same output dir as greedy_sim)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, 'outputs')
RESULTS_CSV = os.path.join(OUTPUTS_DIR, 'simulation_results.csv')


def load_results(path):
    """Load simulation_results.csv. Returns list of dicts with numeric fields where possible."""
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            d = dict(row)
            # Parse numeric columns (empty = inf/missing)
            for key in ('makespan', 'sum_completion_times', 'avg_completion_time'):
                raw = d.get(key, '')
                try:
                    d[key] = float(raw) if raw else None
                except ValueError:
                    d[key] = None
            for key in ('dataset_id', 'n_orders'):
                raw = d.get(key, '')
                try:
                    d[key] = int(float(raw)) if raw else None
                except ValueError:
                    d[key] = None
            rows.append(d)
    return rows


def analyze(rows):
    """Compute stats by size and method. Returns structured dict for reporting."""
    sizes = ('small', 'medium', 'large')
    methods = ('greedy', 'baseline')

    by_size_method = {s: {m: [] for m in methods} for s in sizes}
    for r in rows:
        s, m = r.get('size'), r.get('method')
        if s not in by_size_method or m not in by_size_method[s]:
            continue
        by_size_method[s][m].append(r)

    def stats(vals, key):
        """vals = list of row dicts, key = 'makespan' | 'sum_completion_times' | 'avg_completion_time'."""
        nums = [r[key] for r in vals if r.get(key) is not None]
        if not nums:
            return {'count': 0, 'count_inf': len(vals), 'mean': None, 'median': None, 'min': None, 'max': None}
        return {
            'count': len(nums),
            'count_inf': len(vals) - len(nums),
            'mean': statistics.mean(nums),
            'median': statistics.median(nums),
            'min': min(nums),
            'max': max(nums),
        }

    out = {}
    for size in sizes:
        out[size] = {}
        for method in methods:
            data = by_size_method[size][method]
            out[size][method] = {
                'n_datasets': len(data),
                'makespan': stats(data, 'makespan'),
                'sum_completion': stats(data, 'sum_completion_times'),
                'avg_completion': stats(data, 'avg_completion_time'),
            }
    return out


def format_val(x, decimals=1):
    if x is None:
        return "n/a"
    return f"{x:.{decimals}f}"


def print_report(analysis):
    print("=" * 72)
    print("  SIMULATION RESULTS — Analysis by Size and Method")
    print("=" * 72)

    for size in ('small', 'medium', 'large'):
        print(f"\n  --- {size.upper()} ({analysis[size]['greedy']['n_datasets']} datasets) ---\n")
        for method in ('greedy', 'baseline'):
            M = analysis[size][method]
            ms = M['makespan']
            sc = M['sum_completion']
            ac = M['avg_completion']

            print(f"  {method.upper()}")
            print(f"    Makespan:           mean = {format_val(ms['mean']):>8}  median = {format_val(ms['median']):>8}  min = {format_val(ms['min']):>8}  max = {format_val(ms['max']):>8}")
            print(f"    Sum completion:     mean = {format_val(sc['mean']):>8}  median = {format_val(sc['median']):>8}")
            print(f"    Avg completion:     mean = {format_val(ac['mean']):>8}  (avg time to complete per order)")
            print(f"    Completed (finite makespan): {ms['count']} / {ms['count'] + ms['count_inf']}  (infeasible: {ms['count_inf']})")
            print()

    # Cross-size summary: average time to complete (avg completion) by size (both methods)
    print("  " + "=" * 68)
    print("  AVERAGE TIME TO COMPLETE (per order) — by size")
    print("  " + "=" * 68)
    print(f"  {'Size':<10} {'Greedy (mean)':>16} {'Baseline (mean)':>16} {'Greedy (median)':>16} {'Baseline (median)':>16}")
    print("  " + "-" * 76)
    for size in ('small', 'medium', 'large'):
        g_ac = analysis[size]['greedy']['avg_completion']
        b_ac = analysis[size]['baseline']['avg_completion']
        g_mean = format_val(g_ac['mean']) if g_ac['mean'] is not None else "n/a"
        b_mean = format_val(b_ac['mean']) if b_ac['mean'] is not None else "n/a"
        g_med = format_val(g_ac['median']) if g_ac['median'] is not None else "n/a"
        b_med = format_val(b_ac['median']) if b_ac['median'] is not None else "n/a"
        print(f"  {size:<10} {g_mean:>16} {b_mean:>16} {g_med:>16} {b_med:>16}")
    print()

    # Makespan summary by size
    print("  " + "=" * 68)
    print("  MAKESPAN (mean) — by size and method")
    print("  " + "=" * 68)
    print(f"  {'Size':<10} {'Greedy':>14} {'Baseline':>14}  (n/a = all infeasible)")
    print("  " + "-" * 40)
    for size in ('small', 'medium', 'large'):
        g_ms = analysis[size]['greedy']['makespan']['mean']
        b_ms = analysis[size]['baseline']['makespan']['mean']
        print(f"  {size:<10} {format_val(g_ms):>14} {format_val(b_ms):>14}")
    print()


def write_analysis_csv(analysis, path):
    """Write a flat summary CSV for further use."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['size', 'method', 'makespan_mean', 'makespan_median', 'makespan_min', 'makespan_max',
                    'sum_completion_mean', 'avg_completion_mean', 'n_completed', 'n_total'])
        for size in ('small', 'medium', 'large'):
            for method in ('greedy', 'baseline'):
                M = analysis[size][method]
                ms = M['makespan']
                sc = M['sum_completion']
                ac = M['avg_completion']
                n_comp = ms['count']
                n_tot = ms['count'] + ms['count_inf']
                w.writerow([
                    size, method,
                    ms['mean'], ms['median'], ms['min'], ms['max'],
                    sc['mean'], ac['mean'],
                    n_comp, n_tot,
                ])
    print(f"  Summary CSV written: {path}")


if __name__ == '__main__':
    if not os.path.isfile(RESULTS_CSV):
        print(f"Results file not found: {RESULTS_CSV}")
        print("Run the simulation first:  python greedy_sim.py")
        exit(1)

    rows = load_results(RESULTS_CSV)
    analysis = analyze(rows)
    print_report(analysis)
    write_analysis_csv(analysis, os.path.join(OUTPUTS_DIR, 'analysis_summary.csv'))
