"""
MSE 443 Module 3 — Warehousing Optimization Dashboard
  1. Results & Analysis   — pre-computed batch results
  2. Run on Your Data     — upload CSVs, run all heuristics live
"""

import copy
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NUM_CONVEYORS  = 4
NUM_ITEM_TYPES = 8
ITEM_NAMES     = ['circle', 'pentagon', 'trapezoid', 'triangle',
                  'star', 'moon', 'heart', 'cross']

METHOD_COLORS = {
    'SPT':           '#2196F3',
    'LPT':           '#4CAF50',
    'Greedy':        '#FF9800',
    'Local Search':  '#9C27B0',
    'scheduling_heuristic': '#2196F3',
    'greedy':        '#FF9800',
    'local_search':  '#9C27B0',
}

SIZE_ORDER = ['small', 'medium', 'large']

# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(__file__)

SH_RESULTS  = os.path.join(BASE, 'scheduling_heuristic', 'outputs', 'simulation_results.csv')
GR_RESULTS  = os.path.join(BASE, 'greedy', 'outputs', 'simulation_results.csv')
LS_RESULTS  = os.path.join(BASE, 'local_search', 'simulations', 'simulation_results.csv')
SUMMARY_CSV = os.path.join(BASE, 'comparison_output', 'summary_by_size_and_method.csv')
WINNERS_CSV = os.path.join(BASE, 'comparison_output', 'winner_counts_by_size.csv')
PER_DS_CSV  = os.path.join(BASE, 'comparison_output', 'comparison_per_dataset.csv')


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM IMPLEMENTATIONS (self-contained, no file I/O)
# ─────────────────────────────────────────────────────────────────────────────

def parse_uploaded_csvs(it_file, qt_file, tt_file):
    """
    Parse three uploaded file objects into orders and totes dicts.
    Returns (orders, totes, processing_times).
      orders : {order_id: {'items': [(item_type, qty), ...], 'total_items': int}}
      totes  : {tote_id:  [{'order': oid, 'item_type': it, 'quantity': qty}, ...]}
      processing_times : list[int]  (total items per order, same order as orders)
    """
    def read_raw(f):
        rows = []
        text = f.read().decode('utf-8') if hasattr(f, 'read') else f
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            row = []
            for p in parts:
                p = p.strip()
                row.append(float(p) if p != '' else None)
            rows.append(row)
        return rows

    it_data = read_raw(it_file)
    qt_data = read_raw(qt_file)
    tt_data = read_raw(tt_file)

    orders = {}
    totes  = {}

    for order_idx, (it_row, qt_row, tt_row) in enumerate(zip(it_data, qt_data, tt_data)):
        order_id = order_idx + 1
        orders[order_id] = {'items': [], 'total_items': 0}
        for it, qt, tt in zip(it_row, qt_row, tt_row):
            if it is None or qt is None or tt is None:
                continue
            tote_id   = int(tt)
            item_type = int(it)
            quantity  = int(qt)
            orders[order_id]['items'].append((item_type, quantity))
            orders[order_id]['total_items'] += quantity
            totes.setdefault(tote_id, []).append({
                'order':     order_id,
                'item_type': item_type,
                'quantity':  quantity,
            })

    processing_times = [orders[oid]['total_items'] for oid in sorted(orders)]
    return orders, totes, processing_times


# ── SPT / LPT ────────────────────────────────────────────────────────────────

def schedule_orders(processing_times, n_conveyors, reverse=False):
    sorted_indices = sorted(range(len(processing_times)),
                            key=lambda i: processing_times[i], reverse=reverse)
    loads      = [0] * n_conveyors
    assignment = {c: [] for c in range(n_conveyors)}
    for idx in sorted_indices:
        c = loads.index(min(loads))
        assignment[c].append(idx)
        loads[c] += processing_times[idx]
    return assignment


def compute_sh_metrics(assignment, processing_times, n_conveyors):
    loads     = [sum(processing_times[i] for i in assignment[c]) for c in range(n_conveyors)]
    makespan  = max(loads)
    sum_ct    = sum(loads)
    active    = [l for l in loads if l > 0]
    avg_ct    = sum_ct / len(active) if active else 0
    return makespan, sum_ct, avg_ct


def run_spt_lpt(processing_times, orders_list):
    results = {}
    for name, reverse in [('SPT', False), ('LPT', True)]:
        assign     = schedule_orders(processing_times, NUM_CONVEYORS, reverse)
        conv_loads = {c: sum(processing_times[i] for i in assign[c])
                      for c in range(NUM_CONVEYORS)}

        # Per-order completion times: cumulative within each conveyor queue.
        # Order j finishes after all orders before it on the same conveyor are done.
        # This correctly differs between SPT (short jobs first → earlier completions)
        # and LPT (long jobs first → later completions for most orders).
        order_completion = {}
        for c in range(NUM_CONVEYORS):
            cumtime = 0
            for order_idx in assign[c]:
                cumtime += processing_times[order_idx]
                order_completion[order_idx + 1] = cumtime  # order IDs are 1-indexed

        makespan = max(order_completion.values()) if order_completion else 0
        sum_ct   = sum(order_completion.values())
        avg_ct   = sum_ct / len(order_completion) if order_completion else 0

        results[name] = {
            'makespan':               makespan,
            'sum_completion_times':   sum_ct,
            'avg_completion_time':    avg_ct,
            'assignment':             assign,
            'conv_loads':             conv_loads,
            'order_completion_times': order_completion,
        }
    return results


# ── GREEDY + LOCAL SEARCH (shared helpers) ────────────────────────────────────

def build_round_robin_queues(orders):
    order_list = sorted(orders.keys())
    queues     = [[] for _ in range(NUM_CONVEYORS)]
    for i, oid in enumerate(order_list):
        queues[i % NUM_CONVEYORS].append(oid)
    return queues


def _greedy_sequence(orders, totes, conv_queues):
    remaining = {oid: dict({it: qty for it, qty in odata['items']})
                 for oid, odata in orders.items()}
    queues    = [list(q) for q in conv_queues]
    qpos      = [0] * NUM_CONVEYORS

    def active():
        return {queues[c][qpos[c]] for c in range(NUM_CONVEYORS)
                if qpos[c] < len(queues[c])}

    rem_totes = list(totes.keys())
    seq       = []

    while rem_totes:
        act    = active()
        best   = None
        bscore = (-1, -1, float('inf'))

        for tid in rem_totes:
            primary  = 0
            tmp      = {o: dict(d) for o, d in remaining.items()}
            for e in totes[tid]:
                oid, it, qty = e['order'], e['item_type'], e['quantity']
                if oid in act:
                    useful   = min(qty, max(tmp.get(oid, {}).get(it, 0), 0))
                    primary += useful
                    if it in tmp.get(oid, {}):
                        tmp[oid][it] -= qty
            secondary = 0
            for c in range(NUM_CONVEYORS):
                if qpos[c] < len(queues[c]):
                    oid = queues[c][qpos[c]]
                    if oid in act and all(v <= 0 for v in tmp.get(oid, {}).values()):
                        secondary += 2 if (qpos[c] + 1) < len(queues[c]) else 1
            score = (primary, secondary, -tid)
            if score > bscore:
                bscore = score
                best   = tid

        seq.append(best)
        rem_totes.remove(best)
        for e in totes[best]:
            oid, it, qty = e['order'], e['item_type'], e['quantity']
            if oid in remaining and it in remaining[oid]:
                remaining[oid][it] -= qty
        for c in range(NUM_CONVEYORS):
            if qpos[c] < len(queues[c]):
                oid = queues[c][qpos[c]]
                if all(v <= 0 for v in remaining.get(oid, {}).values()):
                    qpos[c] += 1

    return seq


def _simulate(tote_sequence, orders, totes, conv_queues,
              time_per_segment, tote_load_time, time_per_item_spacing):
    remaining = {oid: {it: qty for it, qty in odata['items']}
                 for oid, odata in orders.items()}
    queues    = [list(q) for q in conv_queues]
    qpos      = [0] * NUM_CONVEYORS
    oconv     = {oid: ci for ci, q in enumerate(queues) for oid in q}

    def active_order(ci):
        return queues[ci][qpos[ci]] if qpos[ci] < len(queues[ci]) else None

    order_completion = {}
    current_time     = 0.0

    for tote_id in tote_sequence:
        load_start = current_time + tote_load_time

        conv_rows = {}
        for e in totes[tote_id]:
            oid, it, qty = e['order'], e['item_type'], e['quantity']
            ci = oconv.get(oid)
            if ci is None or active_order(ci) != oid:
                continue
            if ci not in conv_rows:
                conv_rows[ci] = [0] * NUM_ITEM_TYPES
            conv_rows[ci][it] += qty

        if not conv_rows:
            current_time = load_start
            continue

        rows = sorted(conv_rows.items(), key=lambda x: x[0], reverse=True)  # furthest first
        belt_cursor   = load_start
        conv_last_arr = {}

        for ci, counts in rows:
            total = sum(counts)
            for k in range(total):
                place_t   = belt_cursor + k * time_per_item_spacing
                arrival_t = place_t + time_per_segment * (ci + 1)
                conv_last_arr[ci] = max(conv_last_arr.get(ci, 0), arrival_t)
            belt_cursor += total * time_per_item_spacing

        for ci, last_arr in conv_last_arr.items():
            oid = active_order(ci)
            if oid is None or oid in order_completion:
                continue
            for it_idx, qty in enumerate(conv_rows[ci]):
                if qty > 0 and it_idx in remaining.get(oid, {}):
                    remaining[oid][it_idx] -= qty
            if all(v <= 0 for v in remaining[oid].values()):
                order_completion[oid] = last_arr
                qpos[ci] += 1

        current_time = belt_cursor

    for oid in orders:
        if oid not in order_completion:
            order_completion[oid] = float('inf')

    makespan  = max(order_completion.values())
    finite    = [t for t in order_completion.values() if t != float('inf')]
    sum_ct    = sum(finite)
    avg_ct    = sum_ct / len(finite) if finite else float('inf')

    return {
        'order_completion_times': order_completion,
        'makespan':               makespan,
        'sum_completion_times':   sum_ct,
        'avg_completion_time':    avg_ct,
    }


def run_greedy(orders, totes, time_per_segment, tote_load_time, time_per_item_spacing):
    queues = build_round_robin_queues(orders)
    seq    = _greedy_sequence(orders, totes, queues)
    result = _simulate(seq, orders, totes, queues,
                       time_per_segment, tote_load_time, time_per_item_spacing)
    result['tote_sequence'] = seq
    result['conv_queues']   = queues
    return result


def run_local_search(orders, totes, time_per_segment, tote_load_time,
                     time_per_item_spacing, max_iterations=200):
    queues    = build_round_robin_queues(orders)
    greedy_seq = _greedy_sequence(orders, totes, queues)

    best_queues   = [list(q) for q in queues]
    best_seq      = list(greedy_seq)
    best_ms       = _simulate(best_seq, orders, totes, best_queues,
                              time_per_segment, tote_load_time,
                              time_per_item_spacing)['sum_completion_times']
    best_result   = _simulate(best_seq, orders, totes, best_queues,
                              time_per_segment, tote_load_time, time_per_item_spacing)

    def eval_sol(qs, seq):
        r = _simulate(seq, orders, totes, qs,
                      time_per_segment, tote_load_time, time_per_item_spacing)
        return r['sum_completion_times'], r

    history = [best_ms]

    for _ in range(max_iterations):
        improved = False

        # Tote swaps
        n = len(best_seq)
        for i in range(n - 1):
            if improved:
                break
            for j in range(i + 1, n):
                new_seq       = list(best_seq)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                ms, res       = eval_sol(best_queues, new_seq)
                if ms < best_ms:
                    best_seq  = new_seq
                    best_ms   = ms
                    best_result = res
                    improved  = True
                    break

        # Order moves
        if not improved:
            for from_ci in range(NUM_CONVEYORS):
                if improved:
                    break
                if len(best_queues[from_ci]) <= 1:
                    continue
                for oid in list(best_queues[from_ci]):
                    if improved:
                        break
                    for to_ci in range(NUM_CONVEYORS):
                        if to_ci == from_ci or improved:
                            continue
                        for pos in range(len(best_queues[to_ci]) + 1):
                            new_q = [list(q) for q in best_queues]
                            new_q[from_ci].remove(oid)
                            new_q[to_ci].insert(pos, oid)
                            ms, res = eval_sol(new_q, best_seq)
                            if ms < best_ms:
                                best_queues = new_q
                                best_ms     = ms
                                best_result = res
                                improved    = True
                                break

        history.append(best_ms)
        if not improved:
            break

    best_result['tote_sequence'] = best_seq
    best_result['conv_queues']   = best_queues
    best_result['history']       = history
    return best_result


# ─────────────────────────────────────────────────────────────────────────────
# LOAD PRE-COMPUTED RESULTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    dfs = []
    for path, label in [(SH_RESULTS, 'sh'), (GR_RESULTS, 'gr'), (LS_RESULTS, 'ls')]:
        if os.path.exists(path):
            df       = pd.read_csv(path)
            df['src'] = label
            dfs.append(df)
    if not dfs:
        return None, None, None, None, None

    all_df = pd.concat(dfs, ignore_index=True)
    # normalise method names for display
    all_df['method_display'] = all_df['method'].map({
        'SPT':          'SPT',
        'LPT':          'LPT',
        'greedy':       'Greedy',
        'baseline':     'Greedy (baseline)',
        'local_search': 'Local Search',
    }).fillna(all_df['method'])

    summary = pd.read_csv(SUMMARY_CSV) if os.path.exists(SUMMARY_CSV) else None
    winners = pd.read_csv(WINNERS_CSV) if os.path.exists(WINNERS_CSV) else None
    per_ds  = pd.read_csv(PER_DS_CSV)  if os.path.exists(PER_DS_CSV)  else None

    return all_df, summary, winners, per_ds


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: RESULTS & ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def page_results():
    st.title("Results & Analysis")
    st.caption("Pre-computed results across 90 datasets (30 small · 30 medium · 30 large)")

    all_df, summary, winners, per_ds = load_results()

    if all_df is None:
        st.error("Could not find result CSV files. Run the batch scripts first.")
        return

    # Build a clean working dataframe — SPT and LPT kept separate
    KEEP    = ['SPT', 'LPT', 'greedy', 'local_search']
    DISPLAY = {'SPT': 'SPT', 'LPT': 'LPT', 'greedy': 'Greedy', 'local_search': 'Local Search'}
    METHOD_ORDER = ['SPT', 'LPT', 'Greedy', 'Local Search']

    df = all_df[all_df['method'].isin(KEEP)].copy()
    df['method_display'] = df['method'].map(DISPLAY)
    df['size'] = pd.Categorical(df['size'], SIZE_ORDER, ordered=True)

    DISPLAY_COLORS = {
        'SPT':          METHOD_COLORS['SPT'],
        'LPT':          METHOD_COLORS['LPT'],
        'Greedy':       METHOD_COLORS['Greedy'],
        'Local Search': METHOD_COLORS['Local Search'],
    }

    # ── Summary Stats ───────────────────────────────────────────────────────
    st.header("Summary Statistics")

    agg = (
        df.groupby(['size', 'method_display'])['sum_completion_times']
        .agg(Mean='mean', Std='std', Min='min', Max='max', N='count')
        .reset_index()
        .rename(columns={'size': 'Size', 'method_display': 'Method'})
    )
    agg['Method'] = pd.Categorical(agg['Method'], METHOD_ORDER, ordered=True)
    agg = agg.sort_values(['Size', 'Method'])
    st.dataframe(agg.set_index(['Size', 'Method']).round(2), use_container_width=True)

    # ── Mean Sum Completion Times by Size & Method ─────────────────────────
    st.header("Mean Sum of Completion Times")

    cols = st.columns(3)
    for col, size in zip(cols, SIZE_ORDER):
        size_agg = agg[agg['Size'] == size]
        fig = px.bar(
            size_agg,
            x='Method', y='Mean',
            color='Method',
            error_y='Std',
            category_orders={'Method': METHOD_ORDER},
            color_discrete_map=DISPLAY_COLORS,
            labels={'Mean': 'Mean Sum Completion Times (s)', 'Method': ''},
            title=size.capitalize(),
        )
        fig.update_layout(height=380, showlegend=False)
        col.plotly_chart(fig, use_container_width=True)

    # ── Winner Counts ────────────────────────────────────────────────────────
    st.header("Winner Counts (Best Sum Completion Times per Dataset)")

    # Compute winners from raw data so SPT and LPT are treated separately
    pivot = df.pivot_table(
        index=['size', 'dataset_id'],
        columns='method_display',
        values='sum_completion_times',
        aggfunc='min',
    ).reset_index()

    win_rows = []
    for _, row in pivot.iterrows():
        vals = {m: row[m] for m in METHOD_ORDER if m in row.index and pd.notna(row[m])}
        if vals:
            best_val = min(vals.values())
            for m, v in vals.items():
                if v == best_val:
                    win_rows.append({'size': row['size'], 'method': m})
                    break  # one winner per dataset

    win_df = pd.DataFrame(win_rows)
    win_counts = (
        win_df.groupby(['size', 'method']).size()
        .reset_index(name='wins')
    )
    win_counts['size'] = pd.Categorical(win_counts['size'], SIZE_ORDER, ordered=True)
    win_counts['method'] = pd.Categorical(win_counts['method'], METHOD_ORDER, ordered=True)
    win_counts = win_counts.sort_values(['size', 'method'])

    fig2 = px.bar(
        win_counts,
        x='size', y='wins', color='method', barmode='group',
        category_orders={'size': SIZE_ORDER, 'method': METHOD_ORDER},
        color_discrete_map=DISPLAY_COLORS,
        labels={'size': 'Dataset Size', 'wins': 'Number of Wins', 'method': 'Method'},
        title='Number of Datasets Each Method Wins (out of 30)',
    )
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Box Plots ────────────────────────────────────────────────────────────
    st.header("Distribution of Sum Completion Times")

    size_choice = st.selectbox("Select dataset size", SIZE_ORDER, index=0, key='box_size')
    subset = df[df['size'] == size_choice].copy()
    subset['method_display'] = pd.Categorical(subset['method_display'], METHOD_ORDER, ordered=True)

    fig3 = px.box(
        subset.sort_values('method_display'),
        x='method_display', y='sum_completion_times',
        color='method_display',
        color_discrete_map=DISPLAY_COLORS,
        labels={'method_display': 'Method', 'sum_completion_times': 'Sum Completion Times (s)'},
        title=f'Distribution of Sum Completion Times — {size_choice.capitalize()} Datasets',
        points='all',
    )
    fig3.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Per-Dataset Comparison ───────────────────────────────────────────────
    st.header("Per-Dataset Comparison")

    size2  = st.selectbox("Select dataset size", SIZE_ORDER, index=0, key='ds_size')
    sub_ds = df[df['size'] == size2].copy()

    fig4 = go.Figure()
    for method in METHOD_ORDER:
        m_data = sub_ds[sub_ds['method_display'] == method].sort_values('dataset_id')
        if not m_data.empty:
            fig4.add_trace(go.Scatter(
                x=m_data['dataset_id'],
                y=m_data['sum_completion_times'],
                mode='lines+markers',
                name=method,
                line=dict(color=DISPLAY_COLORS[method]),
            ))
    fig4.update_layout(
        title=f'Sum Completion Times per Dataset — {size2.capitalize()}',
        xaxis_title='Dataset ID',
        yaxis_title='Sum Completion Times (s)',
        height=420,
    )
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("View raw per-dataset table"):
        tbl = sub_ds[['dataset_id', 'method_display', 'makespan',
                       'sum_completion_times', 'avg_completion_time', 'n_orders']]
        st.dataframe(tbl.sort_values(['dataset_id', 'method_display'])
                       .reset_index(drop=True), use_container_width=True)

    # ── Key Takeaways ────────────────────────────────────────────────────────
    st.header("Key Takeaways")

    # Mean sum completion times — all methods side by side, relative to best
    means = (
        df.groupby(['size', 'method_display'])['sum_completion_times']
        .mean()
        .reset_index()
    )

    pivot = means.pivot(index='size', columns='method_display', values='sum_completion_times')
    pivot = pivot.reindex(SIZE_ORDER)[METHOD_ORDER]

    summary_rows = []
    for size in SIZE_ORDER:
        row_vals = pivot.loc[size]
        best_val = row_vals.min()
        for method in METHOD_ORDER:
            val = row_vals[method]
            summary_rows.append({
                'Size':         size.capitalize(),
                'Method':       method,
                'Mean (s)':     round(val, 1),
                'vs Best':      f"{val / best_val:.2f}x" if val != best_val else 'BEST',
                'Rank':         int(row_vals.rank()[method]),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df['Size'] = pd.Categorical(
        summary_df['Size'], [s.capitalize() for s in SIZE_ORDER], ordered=True
    )
    summary_df['Method'] = pd.Categorical(summary_df['Method'], METHOD_ORDER, ordered=True)
    summary_df = summary_df.sort_values(['Size', 'Method'])

    def highlight_best(row):
        return ['background-color: #d4edda; font-weight: bold'
                if row['vs Best'] == 'BEST' else '' for _ in row]

    st.markdown("**Mean sum of completion times by method and dataset size (lower is better):**")
    st.dataframe(
        summary_df.set_index(['Size', 'Method']).style.apply(highlight_best, axis=1),
        use_container_width=True,
    )

    # Winner count summary
    spt_wins  = win_counts[win_counts['method'] == 'SPT']['wins'].sum()
    lpt_wins  = win_counts[win_counts['method'] == 'LPT']['wins'].sum()
    gr_wins   = win_counts[win_counts['method'] == 'Greedy']['wins'].sum()
    ls_wins   = win_counts[win_counts['method'] == 'Local Search']['wins'].sum()
    total     = 90
    best_overall = max(
        [('SPT', spt_wins), ('LPT', lpt_wins), ('Greedy', gr_wins), ('Local Search', ls_wins)],
        key=lambda x: x[1]
    )[0]

    st.markdown(f"""
| Finding | Detail |
|---|---|
| **Best overall** | {best_overall} wins the most datasets ({max(spt_wins, lpt_wins, gr_wins, ls_wins)}/{total}) |
| **SPT** | Wins {spt_wins}/{total} datasets total |
| **LPT** | Wins {lpt_wins}/{total} datasets total |
| **Greedy** | Wins {gr_wins}/{total} datasets total |
| **Local Search** | Wins {ls_wins}/{total} datasets total |
""")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: RUN ON YOUR DATA
# ─────────────────────────────────────────────────────────────────────────────

def page_upload():
    st.title("Run on Your Data")
    st.markdown(
        "Upload your three dataset CSVs and each heuristic will be run once on your data."
    )

    # ── File Upload ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        f_types = st.file_uploader("order_itemtypes.csv", type='csv', key='itemtypes')
    with col2:
        f_qtys  = st.file_uploader("order_quantities.csv", type='csv', key='quantities')
    with col3:
        f_totes = st.file_uploader("orders_totes.csv", type='csv', key='totes')

    # ── Timing Parameters ───────────────────────────────────────────────────
    with st.expander("Timing Parameters (calibrate after physical testing)"):
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            time_per_seg = st.number_input(
                "Time per belt segment (s)", min_value=1.0, max_value=120.0,
                value=20.0, step=0.5,
                help="Travel time LOAD→conv0, conv0→conv1, etc."
            )
        with col_t2:
            tote_load_t = st.number_input(
                "Tote load time (s)", min_value=0.5, max_value=30.0,
                value=5.0, step=0.5,
                help="Seconds to physically place a tote on the belt"
            )
        with col_t3:
            item_spacing = st.number_input(
                "Item spacing (s)", min_value=0.5, max_value=20.0,
                value=2.0, step=0.5,
                help="Gap between consecutive items placed on belt"
            )
        max_iter = st.slider(
            "Local Search max iterations", min_value=10, max_value=500,
            value=100, step=10,
            help="Higher = better solution, slower runtime"
        )

    if not (f_types and f_qtys and f_totes):
        st.info("Upload all three CSV files to run the heuristics.")
        _show_csv_format_help()
        return

    # ── Parse & Run ──────────────────────────────────────────────────────────
    if st.button("Run All Heuristics", type="primary"):
        try:
            with st.spinner("Parsing uploaded files..."):
                orders, totes, ptimes = parse_uploaded_csvs(
                    f_types, f_qtys, f_totes
                )
            n_orders = len(orders)
            n_totes  = len(totes)
            st.success(f"Loaded **{n_orders} orders** and **{n_totes} totes**.")

            results = {}

            # SPT & LPT
            with st.spinner("Running SPT and LPT..."):
                sh = run_spt_lpt(ptimes, orders)
                results['SPT'] = sh['SPT']
                results['LPT'] = sh['LPT']

            # Greedy
            with st.spinner("Running Greedy heuristic..."):
                gr = run_greedy(orders, totes, time_per_seg, tote_load_t, item_spacing)
                results['Greedy'] = gr

            # Local Search
            with st.spinner(f"Running Local Search (max {max_iter} iterations)..."):
                ls = run_local_search(orders, totes, time_per_seg, tote_load_t,
                                      item_spacing, max_iterations=max_iter)
                results['Local Search'] = ls

            # ── Results Table ────────────────────────────────────────────────
            st.header("Results Comparison")

            rows = []
            for method, res in results.items():
                ms  = res['makespan']
                sct = res['sum_completion_times']
                act = res['avg_completion_time']
                rows.append({
                    'Method':               method,
                    'Makespan (s)':         round(ms, 2) if ms != float('inf') else '∞',
                    'Sum Completion (s)':   round(sct, 2) if sct != float('inf') else '∞',
                    'Avg Completion (s)':   round(act, 2) if act != float('inf') else '∞',
                })

            res_df = pd.DataFrame(rows)
            best_sct = min(
                r['sum_completion_times'] for r in results.values()
                if r['sum_completion_times'] != float('inf')
            )

            def highlight_best(row):
                val = results[row['Method']]['sum_completion_times']
                if val == best_sct:
                    return ['background-color: #d4edda'] * len(row)
                return [''] * len(row)

            st.dataframe(
                res_df.style.apply(highlight_best, axis=1),
                use_container_width=True, hide_index=True
            )

            # ── Bar Chart ────────────────────────────────────────────────────
            sct_vals = {m: r['sum_completion_times'] for m, r in results.items()
                        if r['sum_completion_times'] != float('inf')}
            if sct_vals:
                fig = px.bar(
                    x=list(sct_vals.keys()),
                    y=list(sct_vals.values()),
                    color=list(sct_vals.keys()),
                    color_discrete_map=METHOD_COLORS,
                    labels={'x': 'Method', 'y': 'Sum Completion Times (s)'},
                    title='Sum of Completion Times by Method',
                )
                fig.update_layout(showlegend=False, height=380)
                st.plotly_chart(fig, use_container_width=True)

            # ── Per-Order Completion Times ───────────────────────────────────
            st.header("Per-Order Completion Times")
            st.caption("SPT/LPT times are in item-count units (cumulative items processed per conveyor). "
                       "Greedy and Local Search times are in seconds (belt simulation).")

            order_ids = sorted(orders.keys())
            oct_data  = {'Order': [f'Order {oid}' for oid in order_ids]}
            for method, res in results.items():
                oct = res.get('order_completion_times', {})
                oct_data[method] = [
                    round(oct.get(oid, float('inf')), 1)
                    if oct.get(oid, float('inf')) != float('inf')
                    else '∞'
                    for oid in order_ids
                ]
            oct_df = pd.DataFrame(oct_data)
            st.dataframe(oct_df.set_index('Order'), use_container_width=True)

            # line chart for order completion times (only finite values)
            fig2 = go.Figure()
            for method, res in results.items():
                oct = res.get('order_completion_times', {})
                ys  = [oct.get(oid, None) for oid in order_ids]
                ys  = [y if y != float('inf') else None for y in ys]
                fig2.add_trace(go.Scatter(
                    x=[f'O{oid}' for oid in order_ids],
                    y=ys,
                    mode='lines+markers',
                    name=method,
                    line=dict(color=METHOD_COLORS.get(method)),
                ))
            fig2.update_layout(
                title='Per-Order Completion Times',
                xaxis_title='Order',
                yaxis_title='Completion Time (s)',
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)

            # ── Conveyor Assignments ─────────────────────────────────────────
            st.header("Conveyor Queue Assignments")

            tabs = st.tabs(['SPT', 'LPT', 'Greedy', 'Local Search'])
            for tab, method in zip(tabs, ['SPT', 'LPT', 'Greedy', 'Local Search']):
                with tab:
                    res = results[method]
                    if method in ('SPT', 'LPT'):
                        assign = res['assignment']
                        for c in range(NUM_CONVEYORS):
                            order_indices = assign[c]
                            order_ids_c   = [idx + 1 for idx in order_indices]
                            load          = res['conv_loads'][c]
                            st.write(f"**Conv {c}** — Orders {order_ids_c} "
                                     f"(total items: {load})")
                    else:
                        queues = res.get('conv_queues', [])
                        for c, q in enumerate(queues):
                            st.write(f"**Conv {c}** — Orders {q}")

            # ── Local Search Convergence ─────────────────────────────────────
            if 'history' in results.get('Local Search', {}):
                st.header("Local Search Convergence")
                hist = results['Local Search']['history']
                fig3 = px.line(
                    x=list(range(len(hist))),
                    y=hist,
                    labels={'x': 'Iteration', 'y': 'Sum Completion Times (s)'},
                    title='Local Search: Sum Completion Times vs Iteration',
                    color_discrete_sequence=[METHOD_COLORS['Local Search']],
                )
                fig3.update_layout(height=350)
                st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)


def _show_csv_format_help():
    st.subheader("Expected CSV Format")
    st.markdown("""
Each dataset requires **three CSV files** with no headers. Each row = one order.
Columns within a row correspond across all three files.

**order_itemtypes.csv** — item type code (0–7) for each item slot:
```
3, 1
2, 3, 0
```

**order_quantities.csv** — quantity for each item slot:
```
3, 2
3, 1, 1
```

**orders_totes.csv** — tote ID containing each item:
```
1, 1
2, 3, 2
```

Item type codes: `0=circle, 1=pentagon, 2=trapezoid, 3=triangle, 4=star, 5=moon, 6=heart, 7=cross`
""")


# ─────────────────────────────────────────────────────────────────────────────
# APP ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="MSE 443 — Warehousing Optimization",
        page_icon="📦",
        layout="wide",
    )

    with st.sidebar:
        st.title("📦 MSE 443 Module 3")
        st.caption("Warehousing Optimization")
        st.divider()
        page = st.radio(
            "Navigate",
            ["Results & Analysis", "Run on Your Data"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("**Methods compared**")
        st.markdown("""
- **SPT** — Shortest Processing Time
- **LPT** — Longest Processing Time
- **Greedy** — Marginal gain tote sequencing
- **Local Search** — Tote swaps + order moves
""")

    if page == "Results & Analysis":
        page_results()
    else:
        page_upload()


if __name__ == "__main__":
    main()
