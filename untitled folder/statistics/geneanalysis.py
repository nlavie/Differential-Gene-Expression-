#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd

def analyze(data, column):
    print('%s: %s < 0.05' % (column, data[data[column] < 0.05].shape[0]))
    print('%s: %s < 0.1' % (column, data[data[column] < 0.1].shape[0]))

ttest = pd.read_csv('ttest.fdr.csv')
print('Results by ttest:')
analyze(ttest, '2-fdr h2c')
analyze(ttest, '2-fdr h2hm')
analyze(ttest, '2-fdr h2f')

# Skipping for performance
# ranksum = pd.read_csv('ranksum.fdr.csv')
# print('Results by ranksums:')
# analyze(ranksum, '2-fdr h2c')
# analyze(ranksum, '2-fdr h2hm')
# analyze(ranksum, '2-fdr h2f')

def plot_DE(data, column, title):
    pvals = data.sort_values(column)[column]
    plt.scatter(list(range(pvals.count())), pvals, c='r', marker='x')
    plt.axhline(y = 0.05)
    plt.xlabel('rank')
    plt.ylabel(column)
    plt.title(title)
    plt.savefig('%s.png' % column)
    plt.clf()

plot_DE(ttest, 'h2c 1-pvalue', 'Healthy to Convalescent Differential Expression 1-tailed')
plot_DE(ttest, '1-fdr h2c', 'Healthy to Convalescent Differential Expression 1-tailed FDR')
plot_DE(ttest, 'c2h 1-pvalue', 'Convalescent to Healthy Differential Expression 1-tailed')
plot_DE(ttest, '1-fdr c2h', 'Convalescent to Healthy Differential Expression 1-tailed FDR')
plot_DE(ttest, 'h2c 2-pvalue', 'Healthy to Convalescent Differential Expression 2-tailed')
plot_DE(ttest, '2-fdr h2c', 'Healthy to Convalescent Differential Expression 2-tailed FDR')

plot_DE(ttest, 'h2hm 1-pvalue', 'Healthy to Hemorrhagic Differential Expression 1-tailed')
plot_DE(ttest, '1-fdr h2hm', 'Healthy to Hemorrhagic Differential Expression 1-tailed FDR')
plot_DE(ttest, 'hm2h 1-pvalue', 'Hemorrhagic to Healthy Differential Expression 1-tailed')
plot_DE(ttest, '1-fdr hm2h', 'Hemorrhagic to Healthy Differential Expression 1-tailed FDR')
plot_DE(ttest, 'h2hm 2-pvalue', 'Healthy to Hemorrhagic Differential Expression 2-tailed')
plot_DE(ttest, '2-fdr h2hm', 'Healthy to Hemorrhagic Differential Expression 2-tailed FDR')

plot_DE(ttest, 'h2f 1-pvalue', 'Healthy to Fever Differential Expression 1-tailed')
plot_DE(ttest, '1-fdr h2f', 'Healthy to Fever Differential Expression 1-tailed FDR')
plot_DE(ttest, 'f2h 1-pvalue', 'Fever to Healthy Differential Expression 1-tailed')
plot_DE(ttest, '1-fdr f2h', 'Fever to Healthy Differential Expression 1-tailed FDR')
plot_DE(ttest, 'h2f 2-pvalue', 'Healthy to Fever Differential Expression 2-tailed')
plot_DE(ttest, '2-fdr h2f', 'Healthy to Fever Differential Expression 2-tailed FDR')
