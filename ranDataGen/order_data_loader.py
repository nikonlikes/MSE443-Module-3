"""
Load order data from order_itemtypes.csv, order_quantities.csv, orders_totes.csv
into an efficient pandas DataFrame.

Each cell (order row × item_slot column) represents one item: itemtype, quantity, tote.
Empty cells are omitted from the DataFrame.
"""

import pandas as pd


def load_order_data(
    itemtypes_path='order_itemtypes.csv',
    quantities_path='order_quantities.csv',
    totes_path='orders_totes.csv',
    base_dir='',
):
    """
    Load the three CSV files and build a single long-format DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: order, item_slot, itemtype, quantity, tote
        - order    : 1-indexed order ID (row in the CSV)
        - item_slot: 0-indexed position within that order
        - itemtype : item type ID
        - quantity : number of units
        - tote     : tote ID this item comes from

    Example
    -------
    >>> df = load_order_data(base_dir='ranDataGen/')
    >>> df[(df['order']==1) & (df['item_slot']==0)]
    ...  # 3 instances of item type 3 in tote 1
    """
    base = base_dir.rstrip('/') + '/' if base_dir else ''

    it_df = pd.read_csv(base + itemtypes_path, header=None)
    qt_df = pd.read_csv(base + quantities_path, header=None)
    tt_df = pd.read_csv(base + totes_path, header=None)

    rows = []
    for order_idx, (it_row, qt_row, tt_row) in enumerate(zip(
        it_df.itertuples(index=False),
        qt_df.itertuples(index=False),
        tt_df.itertuples(index=False),
    )):
        order_id = order_idx + 1
        for item_slot, (it, qt, tt) in enumerate(zip(it_row, qt_row, tt_row)):
            # Skip empty cells (NaN or empty string)
            if pd.isna(it) or pd.isna(qt) or pd.isna(tt):
                continue
            if it == '' or qt == '' or tt == '':
                continue
            rows.append({
                'order': order_id,
                'item_slot': item_slot,
                'itemtype': int(it),
                'quantity': int(qt),
                'tote': int(tt),
            })

    return pd.DataFrame(rows)


def load_order_data_indexed(base_dir='ranDataGen/'):
    """
    Same as load_order_data but returns a DataFrame indexed by (order, item_slot)
    for fast lookups: df.loc[(order_id, item_slot)].
    """
    df = load_order_data(base_dir=base_dir)
    return df.set_index(['order', 'item_slot'])


if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else 'ranDataGen/'
    df = load_order_data(base_dir=base)
    print(df.to_string())
    print(f'\nShape: {df.shape}')
    print('Example: order 1, item_slot 0 (cell 1,1)')
    print(df[(df['order'] == 1) & (df['item_slot'] == 0)])