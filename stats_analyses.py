import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



if __name__== "__main__":
    data = pd.read_csv("stats/cogSci.csv")
    mctsData = pd.read_csv("stats/mctsRuns.csv")
    dataEntropy = pd.read_csv("stats/cogSciEntropy.csv")
    print 'hello'
    # sns.set_style("darkgrid")
    # # ax = sns.factorplot(x="size_type", y="actionsSolution",col="condition", hue="solutionAndValidationCorrect", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    # ax = sns.factorplot(x="board", y="solutionAndValidationCorrect", hue="condition", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'], markers=['o','^'], linestyles=["-", "--"], legend=False)
    sns.set(style="whitegrid")

    # participant success rate figure-----
    ax = sns.factorplot(x="board", y="solutionAndValidationCorrectPercent", scale= 0.5, hue="condition", data=data, n_boot=1000, order=['6 medium', '10 medium', '6 hard', '10 hard', '10 CV'],  markers=['o','^'], legend_out=False, legend=False)

    ax.set(xlabel='Board', ylabel='Percent Correct')
    lw = ax.ax.lines[0].get_linewidth()
    plt.setp(ax.ax.lines,linewidth=lw)
    plt.legend(loc='best')
    # participant success rate figure end-----

    # alpha-beta 50 moves success rate-----
    ax = sns.factorplot(x="board", y="solutionAndValidationCorrectPercent", scale= 0.5, hue="condition", data=data, n_boot=1000, order=['6 medium', '10 medium', '6 hard', '10 hard', '10 CV'],  markers=['o','^'], legend_out=False, legend=False)

    ax.set(xlabel='Board', ylabel='Percent Correct')
    lw = ax.ax.lines[0].get_linewidth()
    plt.setp(ax.ax.lines,linewidth=lw)
    plt.legend(loc='best')
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
    easy_full_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'easy') & (data['condition'] == 'full')]
    easy_pruned_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'easy') & (data['condition'] == 'truncated')]
    hard_full_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'hard') & (data['condition'] == 'full')]
    hard_pruned_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'hard') & (data['condition'] == 'truncated')]
    easy_full_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'easy') & (data['condition'] == 'full')]
    easy_pruned_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'easy') & (data['condition'] == 'truncated')]
    hard_full_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'hard') & (data['condition'] == 'full')]
    hard_pruned_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'hard') & (data['condition'] == 'truncated')]
    medium_full_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'medium') & (data['condition'] == 'full')]
    medium_pruned_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'medium') & (data['condition'] == 'truncated')]
    #
    easy_full = data.loc[(data['board_type'] == 'easy') & (data['condition'] == 'full')]
    hard_full = data.loc[(data['board_type'] == 'hard') & (data['condition'] == 'full')]

    easy_6 = data.loc[(data['board_size'] == 6) & (data['board_type'] == 'easy')]
    hard_6 = data.loc[(data['board_size'] == 6) & (data['board_type'] == 'hard')]

    easy_10 = data.loc[(data['board_size'] == 10) & (data['board_type'] == 'easy')]
    hard_10 = data.loc[(data['board_size'] == 10) & (data['board_type'] == 'hard')]


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
    plt.show()
    # print 'boo'