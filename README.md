# MSE 443 Module 3 — Warehousing Optimization

Comparison of four conveyor scheduling heuristics (SPT, LPT, Greedy, Local Search) across 90 randomly generated datasets.

---

## Project Structure

```
ranDataGen/                         # Data generator notebook + one sample dataset
ranDataGen copy/                    # All 90 generated datasets (used for batch runs)
  small sized samples/              # 30 datasets
  medium sized samples/             # 30 datasets
  large sized samples/              # 30 datasets

scheduling_heuristic/               # SPT & LPT batch runner
  Scripts/                         # batch runner entrypoint
  outputs/simulation_results.csv   # Results for all 90 datasets

greedy/                             # Greedy heuristic batch runner
  outputs/simulation_results.csv

local_search/                       # Local Search batch runner
  scripts/                          # local search implementation + batch runner
  simulations/simulation_results.csv # precomputed results used by the dashboard
  outputs/simulation_results.csv   # where `run_local_search_batch.py` writes if re-run

comparison_output/                  # Aggregated cross-method comparison CSVs
dashboard.py                        # Streamlit dashboard
```

---

## Datasets

Datasets were generated using `ranDataGen/MSE433_M3_data_generator.ipynb`. Each dataset consists of three headerless CSV files where each row is one order:

| File | Contents |
|---|---|
| `order_itemtypes_N.csv` | Item type code (0–7) for each item slot |
| `order_quantities_N.csv` | Quantity for each item slot |
| `orders_totes_N.csv` | Tote ID containing each item |

Item type codes: `0=circle, 1=pentagon, 2=trapezoid, 3=triangle, 4=star, 5=moon, 6=heart, 7=cross`

90 datasets were generated in three size classes (30 each) and stored in `ranDataGen copy/`.

---

## Running the Batch Scripts

Each heuristic has its own batch script that reads all 90 datasets and writes a `simulation_results.csv`.

**SPT & LPT:**
```bash
python scheduling_heuristic/Scripts/run_scheduling_heuristic_batch.py
# Output: scheduling_heuristic/outputs/simulation_results.csv
```

**Greedy:**
```bash
python greedy/greedy_sim.py
# Output: greedy/outputs/simulation_results.csv
```

**Local Search:**
```bash
python local_search/scripts/run_local_search_batch.py
# Output (script): local_search/outputs/simulation_results.csv
# Note (dashboard): dashboard reads local_search/simulations/simulation_results.csv
```

SPT/LPT, Greedy, and Local Search CSVs must exist for the dashboard's "Results & Analysis" page to populate.
For Local Search specifically, ensure your generated CSV matches the folder the dashboard expects.

---

## Running the Dashboard

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Launch:**
```bash
streamlit run dashboard.py
```

The dashboard opens in your browser at `http://localhost:8501` and has two pages:

### Results & Analysis
Loads the pre-computed batch results and shows:
- Summary statistics (mean, std, min, max) by size and method
- Bar chart of mean sum of completion times
- Winner counts per dataset size
- Box plots of completion time distributions
- Per-dataset line chart comparing all methods
- Key takeaways table with ratios relative to SPT

### Run on Your Data
Upload your own three CSV files (`order_itemtypes`, `order_quantities`, `orders_totes`) and run all four heuristics live. You can adjust timing parameters (belt segment time, tote load time, item spacing) and Local Search iteration count before running.

Expected CSV format (for all 3 files):
- **No headers**.
- Each row corresponds to an order (1-indexed order id in code).
- Each column corresponds to an item slot; the same row/column position across the 3 files must match the same item.
- Cells may be blank; blank entries are treated as “no item in that slot”.

For convenience, `generate_sample_uploads.py` can generate small/medium/large example CSV folders under `sample_uploads/` for testing the dashboard.

Notes about implementation:
- `dashboard.py` runs self-contained heuristics for uploaded data (it does not call the batch scripts).
- For "Results & Analysis", `dashboard.py` loads precomputed CSV outputs from `scheduling_heuristic/outputs`, `greedy/outputs`, and `local_search/simulations`.
- In "Run on Your Data", SPT/LPT completion times are computed in item-count units (cumulative items per conveyor), while Greedy/Local Search are simulated in seconds.
- Local Search objectives differ slightly:
  - Batch local search optimizes **makespan** (while also reporting sum of completion times).
  - Dashboard local search optimizes **sum of completion times**.
