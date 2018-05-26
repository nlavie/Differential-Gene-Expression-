#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy.stats

df = pd.read_csv('wilcoxon.csv')

def WRS_in_steps(df, column):
    rank1 = df[df['rank'] == 1][column]
    T = sum(df.rank()[df['rank'] == 1][column])
    print('%s WRS: %s' % (column, T))

    B = rank1.count()
    N = df[column].count()

    # \[\mu_T=\frac{B(N+1)}{2}\]
    mu = B * (N + 1) / 2.0
    # \[\sigma_T=\sqrt{\frac{B(N-B)(N+1)}{12}}\]
    sigma = np.sqrt((B * (N - B) * (N + 1)) / 12.0)
    # \[Z(T)=\frac{T-\mu_T}{\sigma_T} \sim N(0,1)\]
    Z = (T - mu) / (sigma)

    print('%s Z: %s' % (column, Z))

    pvalue = scipy.stats.norm.sf(Z)
    print('%s 1s p-value: %s' % (column, pvalue))
    print('%s bonferroni p-value: %s' % (column, pvalue * 5))

    print('')

WRS_in_steps(df, 'M1')
WRS_in_steps(df, 'M2')
WRS_in_steps(df, 'M3')
WRS_in_steps(df, 'M4')
WRS_in_steps(df, 'M5')
