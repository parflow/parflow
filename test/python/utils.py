import xarray as xr
from math import floor, log10


def pfbs_are_equal_to_n_sig_figs(pfb1, pfb2):
    n=4
    df1 = xr.open_dataarray(pfb1).to_dataframe()
    df1 = round_df_to_n_sig_figs(df1, n)
    df2 = xr.open_dataarray(pfb2).to_dataframe()
    df2 = round_df_to_n_sig_figs(df2, n)
    return df1.equals(df2)


def round_df_to_n_sig_figs(df, n):
    return df.applymap(lambda x: round(x, n - int(floor(log10(abs(x))))))