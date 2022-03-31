import xarray as xr
from math import floor, log10
import numpy as np

# path_to_pfb = "../correct_output/default_richards.out.press.00000.pfb"
# path_to_pfb2 = "../correct_output/default_richards.out.press.00001.pfb"
#
# da1 = xr.open_dataarray(path_to_pfb)
# da2 = xr.open_dataarray(path_to_pfb)
# da3 = xr.open_dataarray(path_to_pfb2)
#
# print(da1.equals(da2))
# print(da1.equals(da3))
# df1 = da1.to_dataframe()
#
# print("hello world")

def pfbs_are_equal_to_n_sig_figs(pfb1, pfb2):
    n=4
    df1 = xr.open_dataarray(pfb1).to_dataframe()
    df1 = round_df_to_n_sig_figs(df1, n)
    df2 = xr.open_dataarray(pfb2).to_dataframe()
    df2 = round_df_to_n_sig_figs(df2, n)
    return df1.equals(df2)

def round_df_to_n_sig_figs(df, n):
    return df.applymap(lambda x: round(x, n - int(floor(log10(abs(x))))))
    # return da.apply_ufunc(round_to_n, n)


# print(are_pfbs_equal(path_to_pfb, path_to_pfb))
# print(are_pfbs_equal(path_to_pfb, path_to_pfb2))