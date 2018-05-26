#!/usr/bin/env python3

from scipy import stats
import pandas as pd

dl = pd.read_csv('labels3.csv')
df = pd.read_csv('AMI_GSE66360_series_matrix.csv')

ids_to_state = dl.set_index('id').disease_state.to_dict()
state_to_ids = dl.set_index('disease_state')

m = [id for id, state in ids_to_state.items() if state == 'M']
h = [id for id, state in ids_to_state.items() if state == 'H']
#hemorrhagic = [id for id, state in ids_to_state.items() if state == 'Dengue Hemorrhagic Fever']
#fever = [id for id, state in ids_to_state.items() if state == 'Dengue Fever']

import sys, random
def ttest_row(gene, test):
    if random.randint(1, 100) == 1:
        sys.stdout.write('.')
        sys.stdout.flush()

    #h2c = test(gene.ix[healthy], gene.ix[convalescent])
    #h2h = test(gene.ix[healthy], gene.ix[hemorrhagic])
    #h2f = test(gene.ix[healthy], gene.ix[fever])
    m2h = test(gene.ix[m], gene.ix[h])
    return pd.Series({
        'ID_REF': gene.loc['ID_REF'],
        #'h2c stat': h2c.statistic,
        #'h2c 1-pvalue': h2c.pvalue / 2.0 if h2c.statistic > 0 else (1 - (h2c.pvalue / 2.0)),
        #'c2h 1-pvalue': h2c.pvalue / 2.0 if h2c.statistic < 0 else (1 - (h2c.pvalue / 2.0)),
        #'h2c 2-pvalue': h2c.pvalue,
        #'h2hm stat': h2h.statistic,
        #'h2hm 1-pvalue': h2h.pvalue / 2.0 if h2h.statistic > 0 else (1 - (h2h.pvalue / 2.0)),
        #'hm2h 1-pvalue': h2h.pvalue / 2.0 if h2h.statistic < 0 else (1 - (h2h.pvalue / 2.0)),
        #'h2hm 2-pvalue': h2h.pvalue,
        #'h2f stat': h2f.statistic,
        #'h2f 1-pvalue': h2f.pvalue / 2.0 if h2f.statistic > 0 else (1 - (h2f.pvalue / 2.0)),
        #'f2h 1-pvalue': h2f.pvalue / 2.0 if h2f.statistic < 0 else (1 - (h2f.pvalue / 2.0)),
        #'h2f 2-pvalue': h2f.pvalue,
        'm2h stat': m2h.statistic,
        'm2h 1-pvalue': m2h.pvalue / 2.0 if m2h.statistic > 0 else (1 - (m2h.pvalue / 2.0)),
        'm2h 1-pvalue': m2h.pvalue / 2.0 if m2h.statistic < 0 else (1 - (m2h.pvalue / 2.0)),
        'm2h2-pvalue': m2h.pvalue,
    })

print('Computing ttest for data')
ttest = df.apply(ttest_row, axis=1, args=(stats.ttest_ind,))
ttest.to_csv('ttest.csv')

#Skipping ranksums for performance
print('')
print('Computing ranksums for data')
ranksum = df.apply(ttest_row, axis=1, args=(stats.ranksums,))
ranksum.to_csv('ranksum.csv')

# Implement TNoM and use as a third statistical measure
# http://bioinfo.cs.technion.ac.il/projects/Kahana-Navon/TNoM.htm
