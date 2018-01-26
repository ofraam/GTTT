import numpy as np
from bootstrapped import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



if __name__== "__main__":
    data = pd.read_csv("stats/cogSci.csv")
    print 'hello'
    sns.set_style("darkgrid")
    # ax = sns.factorplot(x="size_type", y="actionsSolution",col="condition", hue="solutionAndValidationCorrect", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    ax = sns.factorplot(x="solutionAndValidationCorrect", y="actionsSolution", data=data, n_boot=1000)
    correct = data.loc[data['solutionAndValidationCorrect'] == 1]
    wrong = data.loc[data['solutionAndValidationCorrect'] == 0]
    correct = correct['actionsSolution']
    wrong = wrong['actionsSolution']
    # print bs_compare(correct,wrong)
    # print(bs.bootstrap_ab(correct, wrong))
    # ax = sns.pointplot(x="size_type", y="actionsSolution",hue="condition", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    # ax.show()
    plt.show()
    print 'boo'