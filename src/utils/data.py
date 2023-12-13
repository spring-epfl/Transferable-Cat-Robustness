import numpy as np
import pandas as pd


def one_hot_encode(
    df,
    binary_vars=None,
    cat_cols=None,
    num_cols=None,
    dummy_na=True,
    quantiles=None,
    standardize=False,
    prefix_sep='_',
):
    """
    One-hot encode the categorical features.

    Assumes df only contains categorical or numeric features.
    """
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["category"]).columns.tolist()
    if num_cols is None:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if binary_vars is None:
        binary_vars = []
    cat_cols = [col for col in cat_cols if col not in binary_vars]

    col_sets = []

    # Quantize or standardize numeric features
    if quantiles is not None:
        for col in num_cols:
            qdf = pd.qcut(df[col], q=quantiles, duplicates="drop")
            cat_cols.extend(qdf.columns)
            df = pd.concat([df, qdf], axis=1)
    elif standardize:
        col_sets.append((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
    else:
        col_sets.append(df.drop(columns=cat_cols + binary_vars))

    col_sets.extend([pd.get_dummies(df[cat_cols], dummy_na=dummy_na,prefix_sep=prefix_sep), df[binary_vars]])

    df = pd.concat(col_sets, axis=1)
    cols_to_delete = []
    for col in df.columns:
        if col.endswith("_nan") and df[col].sum() == 0:
            cols_to_delete.append(col)

    df.drop(columns=cols_to_delete, inplace=True)
    return df


def diff(x, x_prime, feature_specs, show_only_diff=True):
    """
    Show diff between two (modified) examples.
    """
    diffs = []
    for spec in feature_specs:
        values = spec.infer_range(x)
        a = spec.get_example_value(x)
        b = spec.get_example_value(x_prime)
        if show_only_diff and (a != b):
            diffs.append(pd.DataFrame({spec.name: [a, b]}))

    df = pd.concat(diffs, axis=1)
    if len(df) > 0:
        df.index = ["original", "transformation"]
    return df
