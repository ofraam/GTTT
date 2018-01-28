import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



if __name__== "__main__":
    data = pd.read_csv("stats/cogSci.csv")
    print 'hello'
    sns.set_style("darkgrid")
    # ax = sns.factorplot(x="size_type", y="actionsSolution",col="condition", hue="solutionAndValidationCorrect", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    # ax = sns.factorplot(x="solutionAndValidationCorrect", y="actionsSolution", data=data, n_boot=1000)
    correct = data.loc[data['solutionAndValidationCorrect'] == 1]
    wrong = data.loc[data['solutionAndValidationCorrect'] == 0]
    correct = correct['actionsSolution']
    wrong = wrong['actionsSolution']

    test = np.random.binomial(100, p=0.2 * 1.1, size=100) * 1.0
    ctrl = np.random.binomial(100, p=0.2, size=500) * 1.0

    print bs_compare.difference(correct.mean(), wrong.mean())
    print bs.bootstrap_ab(correct.as_matrix(), wrong.as_matrix(), bs_stats.mean, bs_compare.difference)
    # ax = sns.pointplot(x="size_type", y="actionsSolution",hue="condition", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    # ax.show()
    # plt.show()
    # print 'boo'