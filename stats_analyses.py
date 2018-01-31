import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def boot_matrix(z, B):
    """Bootstrap sample

    Returns all bootstrap samples in a matrix"""

    n = len(z)  # sample size
    idz = np.random.randint(0, n, size=(B, n))  # indices to pick for all boostrap samples
    return z[idz]

def bootstrap_mean(x, B=10000, alpha=0.05, plot=False):
    """Bootstrap standard error and (1-alpha)*100% c.i. for the population mean

    Returns bootstrapped standard error and different types of confidence intervals"""

    # Deterministic things
    n = len(x)  # sample size
    orig = x.mean()  # sample mean
    se_mean = x.std()/np.sqrt(n) # standard error of the mean
    qt = stats.t.ppf(q=1 - alpha/2, df=n - 1) # Student quantile

    # Generate boostrap distribution of sample mean
    xboot = boot_matrix(x, B=B)
    sampling_distribution = xboot.mean(axis=1)

   # Standard error and sample quantiles
    se_mean_boot = sampling_distribution.std()
    quantile_boot = np.percentile(sampling_distribution, q=(100*alpha/2, 100*(1-alpha/2)))

    # RESULTS
    print("Estimated mean:", orig)
    print("Classic standard error:", se_mean)
    print("Classic student c.i.:", orig + np.array([-qt, qt])*se_mean)
    print("\nBootstrap results:")
    print("Standard error:", se_mean_boot)
    print("t-type c.i.:", orig + np.array([-qt, qt])*se_mean_boot)
    print("Percentile c.i.:", quantile_boot)
    print("Basic c.i.:", 2*orig - quantile_boot[::-1])

    if plot:
        plt.hist(sampling_distribution, bins="fd")



def bootstrap_t_pvalue(x, y, equal_var=False, B=10000, plot=False):
    """Bootstrap p values for two-sample t test

    Returns boostrap p value, test statistics and parametric p value"""

    # Original t test statistic
    orig = stats.ttest_ind(x, y, equal_var=equal_var)

    # Generate boostrap distribution of t statistic
    xboot = boot_matrix(x - x.mean(), B=B) # important centering step to get sampling distribution under the null
    yboot = boot_matrix(y - y.mean(), B=B)
    sampling_distribution = stats.ttest_ind(xboot, yboot, axis=1, equal_var=equal_var)[0]

    # Calculate proportion of bootstrap samples with at least as strong evidence against null
    p = np.mean(sampling_distribution >= orig[0])

    # RESULTS
    print("p value for null hypothesis of equal population means:")
    # print("Parametric:", orig[1])
    print("Bootstrap:", 2*min(p, 1-p))

    # Plot bootstrap distribution
    if plot:
        plt.figure()
        plt.hist(sampling_distribution, bins="fd")


if __name__== "__main__":
    data = pd.read_csv("stats/cogSci.csv")
    mctsData = pd.read_csv("stats/mctsRuns.csv")
    dataEntropy = pd.read_csv("stats/cogSciEntropy.csv")
    print 'hello'
    # sns.set_style("darkgrid")
    # ax = sns.factorplot(x="size_type", y="actionsSolution",col="condition", hue="solutionAndValidationCorrect", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    # ax = sns.factorplot(x="board", y="solutionAndValidationCorrect", hue="condition", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'], markers=['o','^'], linestyles=["-", "--"], legend=False)
    sns.set(style="whitegrid")

    # participant actions figure-----
    ax = sns.factorplot(x="board", y="actionsSolution", scale= 0.5, hue="condition", data=data, n_boot=1000, order=['6 medium', '10 medium', '6 hard', '10 hard', '10 CV'],  markers=['o','^'], legend_out=False, legend=False)

    ax.set(xlabel='Board', ylabel='Number of Moves')
    lw = ax.ax.lines[0].get_linewidth()
    plt.setp(ax.ax.lines,linewidth=lw)
    plt.legend(loc='best')

    # participant success rate figure-----
    # ax = sns.factorplot(x="board", y="solutionAndValidationCorrectPercent", scale= 0.5, hue="condition", data=data, n_boot=1000, order=['6 medium', '10 medium', '6 hard', '10 hard', '10 CV'],  markers=['o','^'], legend_out=False, legend=False)
    #
    # ax.set(xlabel='Board', ylabel='Percent Correct')
    # lw = ax.ax.lines[0].get_linewidth()
    # plt.setp(ax.ax.lines,linewidth=lw)
    # plt.legend(loc='best')
    # participant success rate figure end-----

    # # alpha-beta 50 moves success rate-----
    # ax = sns.factorplot(x="board", y="solutionAndValidationCorrectPercent", scale= 0.5, hue="condition", data=data, n_boot=1000, order=['6 medium', '10 medium', '6 hard', '10 hard', '10 CV'],  markers=['o','^'], legend_out=False, legend=False)
    #
    # ax.set(xlabel='Board', ylabel='Percent Correct')
    # lw = ax.ax.lines[0].get_linewidth()
    # plt.setp(ax.ax.lines,linewidth=lw)
    # plt.legend(loc='best')

    plt.show()
    # alpha-beta 50 moves success rate-----

    # mcts num Nodes CI


    # ax.axes[0][0].legend(loc=1)
    # ax = sns.factorplot(x="size_type", y="entropyNormalized",col="condition", hue="solutionAndValidationCorrect", data=dataEntropy, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    # ax = sns.factorplot(x="solutionAndValidationCorrect", y="actionsSolution", data=data, n_boot=1000)
    correct = data.loc[data['solutionAndValidationCorrect'] == 1]
    wrong = data.loc[data['solutionAndValidationCorrect'] == 0]

    # correct = correct['actionsSolution']
    # wrong = wrong['actionsSolution']
    #
    easy_full_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'medium') & (data['condition'] == 'full')]
    easy_pruned_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'medium') & (data['condition'] == 'truncated')]
    hard_full_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'hard') & (data['condition'] == 'full')]
    hard_pruned_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'hard') & (data['condition'] == 'truncated')]
    easy_full_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'medium') & (data['condition'] == 'full')]
    easy_pruned_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'medium') & (data['condition'] == 'truncated')]
    hard_full_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'hard') & (data['condition'] == 'full')]
    hard_pruned_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'hard') & (data['condition'] == 'truncated')]
    medium_full_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'CV') & (data['condition'] == 'full')]
    medium_pruned_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'CV') & (data['condition'] == 'truncated')]
    #
    easy_full = data.loc[(data['board_type'] == 'medium') & (data['condition'] == 'full')]
    hard_full = data.loc[(data['board_type'] == 'hard') & (data['condition'] == 'full')]

    easy_6 = data.loc[(data['board_size'] == 6) & (data['board_type'] == 'medium')]
    hard_6 = data.loc[(data['board_size'] == 6) & (data['board_type'] == 'hard')]

    easy_10 = data.loc[(data['board_size'] == 10) & (data['board_type'] == 'medium')]
    hard_10 = data.loc[(data['board_size'] == 10) & (data['board_type'] == 'hard')]


    # --- full vs truncated accuracy and num of actions
    # print '6 medium full vs truncated'
    # print bs.bootstrap_ab(easy_pruned_6['solutionAndValidationCorrect'].values,easy_full_6['solutionAndValidationCorrect'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(easy_full_6['solutionAndValidationCorrect'].values, easy_pruned_6['solutionAndValidationCorrect'].values)
    #
    # print '10 medium full vs truncated'
    # print bs.bootstrap_ab(easy_pruned_10['solutionAndValidationCorrect'].values,easy_full_10['solutionAndValidationCorrect'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(easy_full_10['solutionAndValidationCorrect'].values, easy_pruned_10['solutionAndValidationCorrect'].values)
    #
    # print '6 hard full vs truncated'
    # print bs.bootstrap_ab( hard_pruned_6['solutionAndValidationCorrect'].values, hard_full_6['solutionAndValidationCorrect'].values,bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(hard_full_6['solutionAndValidationCorrect'].values, hard_pruned_6['solutionAndValidationCorrect'].values)
    #
    # print '10 hard full vs truncated'
    # print bs.bootstrap_ab( hard_pruned_10['solutionAndValidationCorrect'].values,hard_full_10['solutionAndValidationCorrect'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(hard_full_10['solutionAndValidationCorrect'].values, hard_pruned_10['solutionAndValidationCorrect'].values)
    #
    # print '10 CV full vs truncated'
    # print bs.bootstrap_ab( medium_pruned_10['solutionAndValidationCorrect'].values,medium_full_10['solutionAndValidationCorrect'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(medium_full_10['solutionAndValidationCorrect'].values, medium_pruned_10['solutionAndValidationCorrect'].values)
    #
    #
    # print '6 medium full vs truncated'
    # print bs.bootstrap_ab( easy_pruned_6['actionsSolution'].values,easy_full_6['actionsSolution'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(easy_full_6['actionsSolution'].values, easy_pruned_6['actionsSolution'].values)
    #
    # print '10 medium full vs truncated'
    # print bs.bootstrap_ab( easy_pruned_10['actionsSolution'].values,easy_full_10['actionsSolution'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(easy_full_10['actionsSolution'].values, easy_pruned_10['actionsSolution'].values)
    #
    # print '6 hard full vs truncated'
    # print bs.bootstrap_ab( hard_pruned_6['actionsSolution'].values, hard_full_6['actionsSolution'].values,bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(hard_full_6['actionsSolution'].values, hard_pruned_6['actionsSolution'].values)
    #
    # print '10 hard full vs truncated'
    # print bs.bootstrap_ab( hard_pruned_10['actionsSolution'].values,hard_full_10['actionsSolution'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(hard_full_10['actionsSolution'].values, hard_pruned_10['actionsSolution'].values)
    #
    # print '10 CV full vs truncated'
    # print bs.bootstrap_ab( medium_pruned_10['actionsSolution'].values,medium_full_10['actionsSolution'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(medium_full_10['actionsSolution'].values, medium_pruned_10['actionsSolution'].values)

    # --- full vs truncated accuracy and num of actions end

    # success rates and number of moves participants
    # # print bs_compare.difference(correct.mean(), wrong.mean())
    # # print bs.bootstrap_ab(correct.as_matrix(), wrong.as_matrix(), bs_stats.mean, bs_compare.difference)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print bs.bootstrap(easy_full_6['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full_6['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)

    # print bs.bootstrap(easy_pruned_6['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_pruned_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_pruned_6['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_pruned_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)

    # print bs.bootstrap_ab(easy_full_6['solutionAndValidationCorrect'].values, easy_pruned_6['solutionAndValidationCorrect'].values, bs_stats.mean, bs_compare.difference)

    # print bs.bootstrap(data['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)


    # print bs.bootstrap(hard_full['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_pruned_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_pruned_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_pruned_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_pruned_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)

    # print bs.bootstrap_ab(correct.as_matrix(), wrong.as_matrix(), bs_stats.mean, bs_compare.difference)

    # ax = sns.pointplot(x="size_type", y="actionsSolution",hue="condition", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    # ax.show()

    # print 'boo'