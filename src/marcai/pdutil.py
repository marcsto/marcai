import pandas as pd

def expand_col(df, col_name):
  """
    Expands a cell that are lists into individual columns labeled col_name_0, col_name_1 ...
  """
  expanded = df[col_name].apply(pd.Series)
  expanded = expanded.add_prefix(col_name + "_")
  return pd.concat([df, expanded], axis = 1)


def expand_col_to_rows(df, col_to_expand):
    res = df[col_to_expand].apply(pd.Series).stack()
    res = res.reset_index()
    res.columns = ['idx', 'seq_idx', col_to_expand + '_exp']
    res = res.set_index(['idx', 'seq_idx'])
    return res

def expand_cols_to_rows(df, cols):
    joined = None
    for col in cols:
        a = expand_col_to_rows(df, col)
        if joined is not None:
            joined = joined.join(a)
        else:
            joined = a
    joined = joined.reset_index()
    joined = joined.set_index('idx')
    return df.join(joined)