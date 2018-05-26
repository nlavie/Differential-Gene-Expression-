#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd

def analyze(data, column, threshold):
    print('%s: %s < %s' % (column, data[data[column] < threshold].shape[0], threshold))

ttest = pd.read_csv('ttest.fdr.csv')
print('Results by ttest:')
for threshold in [0.05, 0.1]:
    for col in ['m2h 1-pvalue', 'm2h stat', 'm2h2-pvalue', '1-fdr m2h', '1-fdr h2m', '2-fdr h2m']:
        analyze(ttest, col, threshold)

# Skipping for performance
ranksum = pd.read_csv('ranksum.fdr.csv')
print('Results by ranksums:')
analyze(ranksum, '2-fdr h2m', threshold)
# analyze(ranksum, '2-fdr h2hm')
# analyze(ranksum, '2-fdr h2f')

def plot_FDR(data, column, title):
    pvals = data.sort_values(column)[column]
    plt.scatter(list(range(pvals.count())), pvals, c='r', marker='x')
    plt.axhline(y = 0.05)
    plt.xlabel('rank')
    plt.ylabel('1-tailed p-value' if column.endswith('1-pvalue') else '2-tailed p-value')
    plt.title(title)
    plt.savefig('%s.png' % column.replace(' ', '_'))
    plt.clf()

def plot_OA(data, column, title):
    pvals = data.sort_values(column)[column]
    plt.scatter(pvals, list(range(pvals.count())), c='r', marker='x')
    plt.xlabel('1-tailed p-value' if column.endswith('1-pvalue') else '2-tailed p-value')
    plt.ylabel('rank')
    plt.title(title)
    plt.savefig('%s.png' % column.replace(' ', '_'))
    plt.clf()

plot_OA(ttest, 'm2h 1-pvalue', 'Healthy to Convalescent DE (Healthy Over expressed)')
plot_OA(ttest, '1-fdr m2h', 'Convalescent to Healthy DE (Healthy Under expressed)')
plot_OA(ttest, '2-fdr h2m', 'Healthy to Convalescent DE')

# plot_OA(ttest, 'h2hm 1-pvalue', 'Healthy to Hemorrhagic DE (Healthy Over expressed)')
# plot_OA(ttest, 'hm2h 1-pvalue', 'Hemorrhagic to Healthy DE (Healthy Under expressed)')
# plot_OA(ttest, 'h2hm 2-pvalue', 'Healthy to Hemorrhagic DE')

# plot_OA(ttest, 'h2f 1-pvalue', 'Healthy to Fever DE (Healthy Over expressed)')
# plot_OA(ttest, 'f2h 1-pvalue', 'Fever to Healthy DE (Healthy Under expressed)')
# plot_OA(ttest, 'h2f 2-pvalue', 'Healthy to Fever DE')

dl = pd.read_csv('labels3.csv')
ids_to_state = dl.set_index('id').disease_state.to_dict()
h = [id for id, state in ids_to_state.items() if state == 'H']
m = [id for id, state in ids_to_state.items() if state == 'M']
# hemorrhagic = [id for id, state in ids_to_state.items() if state == 'Dengue Hemorrhagic Fever']
# fever = [id for id, state in ids_to_state.items() if state == 'Dengue Fever']

states = {
    'H': h,
    'M': m,
    # 'Fever': fever,
}

expression_data = pd.read_csv('AMI_GSE66360_series_matrix.csv').set_index('ID_REF')

def significant_genes(by_col, for_class):
    genes = ttest.sort_values(by_col).head(3)['ID_REF']

    colors = {}
    colors[list(genes)[0]] = ('r', 'x')
    colors[list(genes)[1]] = ('g', 'o')
    colors[list(genes)[2]] = ('b', '+')

    print('significant genes for %s: %s' % (for_class, list(genes)))
    for gene in genes:
        gene_expression = expression_data.ix[gene]
        healthy_expression = gene_expression.ix[h]
        class_expression = gene_expression.ix[states[for_class]]
        color, marker = colors[gene]
        plt.scatter(list(range(healthy_expression.count())), list(healthy_expression), c=color, marker=marker, label=gene)
        plt.scatter(list(range(healthy_expression.count() + 1, 1 + healthy_expression.count() + class_expression.count())), list(class_expression), c=color, marker=marker)

    plt.axvline(healthy_expression.count() + 0.5)
    ymin, ymax = plt.gca().get_ylim()
    texty = ymin + (ymax - ymin) * 0.08
    plt.text(healthy_expression.count() + 2, texty, for_class)
    plt.text(1, texty, 'Healthy')
    plt.xlabel('Patient')
    plt.ylabel('Expression level')
    plt.title('Selected Expression Levels for Healthy vs %s' % for_class)
    plt.legend(loc='best')
    plt.savefig('significant-%s.png' % by_col.replace(' ', '_'))
    plt.clf()

significant_genes('2-fdr h2m', 'H')
#significant_genes('2-fdr h2hm', 'Hemorrhagic')
#significant_genes('2-fdr h2f', 'Fever')
