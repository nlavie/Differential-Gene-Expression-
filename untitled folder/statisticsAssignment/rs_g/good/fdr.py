#!/usr/bin/env python3

import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests

def compute_fdr(data):
    _, data['1-fdr m2h'], _, _ = multipletests(data['m2h 1-pvalue'], method='fdr_bh')
    _, data['1-fdr h2m'], _, _ = multipletests(data['m2h stat'], method='fdr_bh')
    _, data['2-fdr h2m'], _, _ = multipletests(data['m2h2-pvalue'], method='fdr_bh')
    # _, data['1-fdr h2hm'], _, _ = multipletests(data['h2hm 1-pvalue'], method='fdr_bh')
    # _, data['1-fdr hm2h'], _, _ = multipletests(data['hm2h 1-pvalue'], method='fdr_bh')
    # _, data['2-fdr h2hm'], _, _ = multipletests(data['h2hm 2-pvalue'], method='fdr_bh')
    # _, data['1-fdr h2f'], _, _ = multipletests(data['h2f 1-pvalue'], method='fdr_bh')
    # _, data['1-fdr f2h'], _, _ = multipletests(data['f2h 1-pvalue'], method='fdr_bh')
    # _, data['2-fdr h2f'], _, _ = multipletests(data['h2f 2-pvalue'], method='fdr_bh')

ttest = pd.read_csv('ttest.csv')
compute_fdr(ttest)
ttest.to_csv('ttest.fdr.csv')

# Skipping for performance
ranksum = pd.read_csv('ranksum.csv')
compute_fdr(ranksum)
ranksum.to_csv('ranksum.fdr.csv')
