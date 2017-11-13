#!/usr/bin/python
# coding: utf-8
import sys
import pandas as pd
from matplotlib import pyplot as plt
def draw(fn, style, ax=None):
    df = pd.read_csv(fn, skiprows=4)
    df = df.loc[df['i'] > 1]
    return df.plot(x="NX", y=["_1000"], legend=True, style=style, ax=ax, label=fn)

ax = draw(sys.argv[1], '+')
draw(sys.argv[2], 'o', ax)

plt.show()

