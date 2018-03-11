import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
# import Tkinter



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
    print("Parametric:", orig[1])
    print("Bootstrap:", 2*min(p, 1-p))

    # Plot bootstrap distribution
    if plot:
        plt.figure()
        plt.hist(sampling_distribution, bins="fd")


def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_y(patch.get_y() + diff * .5)

if __name__== "__main__":
    data = pd.read_csv("stats/cogSci.csv")
    mctsData = pd.read_csv("stats/mctsRuns.csv")
    dataEntropy = pd.read_csv("stats/cogSciEntropy.csv")
    alphaBetaFull = pd.read_csv("stats/cogsciAlphaBeta100000.csv")
    alphaBeta50 = pd.read_csv("stats/alphaBest50_success.csv")
    distances = pd.read_csv("stats/distanceFirstMoves.csv")
    population = pd.read_csv("stats/cogsciPopulation1.csv")
    likelihood = pd.read_csv("stats/logLikelihood.csv")
    dynamics = pd.read_csv("stats/dynamics.csv")
    # exploreExploit = pd.read_csv("stats/exploreExploitData.csv")
    # exploreExploit2 = pd.read_csv("stats/exploreExploitData2.csv")
    exploreExploit2 = pd.read_csv("stats/exploreExploitDataNoUndo.csv")
    timeResets = pd.read_csv("stats/timeBeforeReset.csv")
    timeUndos = pd.read_csv("stats/timeBeforeUndo.csv")

    # log-likelhood
    # density = likelihood.loc[likelihood['heuristic'] == 'density']
    # linear = likelihood.loc[likelihood['heuristic'] == 'linear']
    # nonLinear = likelihood.loc[likelihood['heuristic'] == 'non-linear']
    # nonLinearInteraction  = likelihood.loc[likelihood['heuristic'] == 'non-linear_interaction']
    # blocking = likelihood.loc[likelihood['heuristic'] == 'blocking']
    # chance = likelihood.loc[likelihood['heuristic'] == 'chance']
    #
    # densityCorrect = density.loc[density['participants'] == 'correct']
    # densityWrong = density.loc[density['participants'] == 'wrong']
    # linearCorrect = linear.loc[linear['participants'] == 'correct']
    # linearWrong = linear.loc[linear['participants'] == 'wrong']
    # nonLinearCorrect = nonLinear.loc[nonLinear['participants'] == 'correct']
    # nonLinearWrong = nonLinear.loc[nonLinear['participants'] == 'wrong']
    # nonLinearInteractionCorrect = nonLinearInteraction.loc[nonLinearInteraction['participants'] == 'correct']
    # nonLinearInteractionWrong = nonLinearInteraction.loc[nonLinearInteraction['participants'] == 'wrong']
    # blockingCorrect = blocking.loc[blocking['participants'] == 'correct']
    # blockingWrong = blocking.loc[blocking['participants'] == 'wrong']
    #
    # print bs.bootstrap(data['timeMinutes'].values, stat_func=bs_stats.mean, is_pivotal=False)


    # print bootstrap_t_pvalue(wrong['actionsSolution'].values, correct['actionsSolution'].values)
    # print bs.bootstrap(density['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(linear['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinear['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearInteraction['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(blocking['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(chance['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print 'correct'
    # print bs.bootstrap(densityCorrect['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(linearCorrect['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearCorrect['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearInteractionCorrect['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(blockingCorrect['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(chance['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print 'wrong'
    # print bs.bootstrap(densityWrong['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(linearWrong['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearWrong['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearInteractionWrong['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(blockingWrong['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(chance['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    #
    # print 'p-values'
    # print bootstrap_t_pvalue(densityCorrect['value'].values, densityWrong['value'].values)
    # print bootstrap_t_pvalue(linearCorrect['value'].values, linearWrong['value'].values)
    # print bootstrap_t_pvalue(nonLinearCorrect['value'].values, nonLinearWrong['value'].values)
    # print bootstrap_t_pvalue(nonLinearInteractionCorrect['value'].values, nonLinearInteractionWrong['value'].values)
    # print bootstrap_t_pvalue(blockingCorrect['value'].values, blockingWrong['value'].values)
    # print 'hello'
    # sns.set_style("darkgrid")
    # ax = sns.factorplot(x="size_type", y="actionsSolution",col="condition", hue="solutionAndValidationCorrect", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    # ax = sns.factorplot(x="board", y="solutionAndValidationCorrect", hue="condition", data=data, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'], markers=['o','^'], linestyles=["-", "--"], legend=False)
    sns.set(style="whitegrid")

    # --------------dynamics analysis----------------
    # print stats.spearmanr(exploreExploit2['explore_time'], exploreExploit2['exploit_time'])
    # exploreExploit_filtered1 = exploreExploit2.loc[(exploreExploit2['explore_time'] < 100) & (exploreExploit2['exploit_time'] < 100) & (exploreExploit2['solved']=='validatedCorrect') & (exploreExploit2['board_name']=='6_hard_full')]
    # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    # print stats.spearmanr(exploreExploit_filtered1['explore_time'], exploreExploit_filtered1['exploit_time'])
    # ax = sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered1, n_boot=1000, marker='+', color='green')
    #
    # # plt.xlim(0,100)
    # # plt.ylim(0,100)
    # # plt.show()
    #
    # exploreExploit_filtered2 = exploreExploit2.loc[(exploreExploit2['explore_time'] < 100) & (exploreExploit2['exploit_time'] < 100) & (exploreExploit2['solved']=='wrong') & (exploreExploit2['board_name']=='6_hard_full')]
    # print stats.spearmanr(exploreExploit_filtered2['explore_time'], exploreExploit_filtered2['exploit_time'])
    # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    # ax = sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered2, n_boot=1000, color='red')
    # plt.xlim(0,100)
    # plt.ylim(0,100)
    # plt.show()

    # reset and undo distributions
    # ax = sns.distplot(timeResets['time_before_sec'])
    # timeUndos_filtered = timeUndos.loc[(timeUndos['time_before_sec'] < 10)]
    # timeResets_filtered = timeResets.loc[(timeResets['time_before_sec'] < 10)]
    # # ax = sns.distplot(timeUndos_filtered['time_before_sec'])
    # ax = sns.distplot(timeResets_filtered['time_before_sec'])
    # plt.show()


    # dynamics_filtered = dynamics.loc[(dynamics['move_number_in_path'] < 11) & (dynamics['move_number_in_path'] > 1) & (dynamics['player'] == 2)]
    userids = dynamics['userid'].unique()

    for user in userids:
        # print user
        f, (ax1, ax2, ax3) = plt.subplots(3, figsize = (20,10))
        clicks_filtered = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='click')]
        clicks_filtered_p1 = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='click') & (dynamics['player']==1)]
        clicks_filtered_p2 = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='click') & (dynamics['player']==2)]

        # ax = sns.FacetGrid(dynamics_filtered, row="userid")
        # ax = ax.map_dataframe(sns.tsplot, time='time_rel_sec', value='time_between', unit='userid', data=clicks_filtered, interpolate=False)
        if (len(clicks_filtered_p1) < 2) | (len(clicks_filtered_p2) < 2):
            continue
        sns.tsplot(time='time_rel_sec', value='time_from_click', unit='userid', data=clicks_filtered, interpolate=False, ax=ax1)
        sns.tsplot(time='time_rel_sec', value='score_move', unit='userid', data=clicks_filtered_p1, interpolate=False, color='blue', ax=ax2)
        sns.tsplot(time='time_rel_sec', value='top_possible_score', unit='userid', data=clicks_filtered_p1, interpolate=False, color='orange',  ax=ax2)

        sns.tsplot(time='time_rel_sec', value='score_move', unit='userid', data=clicks_filtered_p2, interpolate=False, color='blue', ax=ax3)
        sns.tsplot(time='time_rel_sec', value='top_possible_score', unit='userid', data=clicks_filtered_p2, interpolate=False, color='orange',  ax=ax3)

        resets = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='reset')]

        for index, event in resets.iterrows():
            time_reset = int(event['time_rel_sec'])
            # print time_reset
            ax1.axvline(time_reset, color="red", linestyle="--");
            ax2.axvline(time_reset, color="red", linestyle="--");
            ax3.axvline(time_reset, color="red", linestyle="--");

            solved = event['solved']
            board_name = event['board_name']

        undos = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='undo')]

        for index, event in undos.iterrows():
            time_undo = int(event['time_rel_sec'])
            # print time_undo
            ax1.axvline(time_undo, color="purple", linestyle="--");
            ax2.axvline(time_undo, color="purple", linestyle="--");
            ax3.axvline(time_undo, color="purple", linestyle="--");



        ax2.set(yscale="symlog")
        ax2.set_ylim(-100000,100000)

        ax3.set(yscale="symlog")
        ax3.set_ylim(-100000,100000)
        # plt.show()

        ax2.set_ylabel('x score vs. best')
        ax3.set_ylabel('o score vs. best')

        title = user + '_' + solved + '_' + board_name
        ax1.set_title(title)
        # plt.show()
        plt.savefig("dynamics/time_series2/timeSeries_"+ title +".png", format='png')

        plt.clf()
        plt.close()

    # participant actions figure-----
    # ax = sns.factorplot(x="board", y="actionsSolution", scale= 0.5, hue="condition", data=data, n_boot=1000, order=['6 medium', '10 medium', '6 hard', '10 hard', '10 CV'],  markers=['o','^'], legend_out=False, legend=False)
    #
    # ax.set(xlabel='Board', ylabel='Number of Moves')
    # lw = ax.ax.lines[0].get_linewidth()
    # plt.setp(ax.ax.lines,linewidth=lw)
    # plt.legend(loc='best')

    # participant actions figure-----
    # alphaBetaFull['heuristic_name'] = alphaBetaFull['heuristic_name'].map({'density':'density', 'linear':  'linear','non-linear':'non-linear', 'non-linear-interaction': 'interaction','blocking':'blocking', 'participants':'participants'})
    # # alpha beta and participant actions figure-----
    # # ax = sns.factorplot(x="board", y="moves",, hue="heuristic_name", data=alphaBetaFull, n_boot=1000, order=['6 MC', '10 MC', '6 HC', '10 HC', '10 DC'],  markers=["1","2","3","4","8","o"], legend_out=False, legend=False)
    # # data['board'] = data['board'].map({'6 MC full': 'MC6 full','10 MC': 'MC10','6 HC': 'HC6','10 HC': 'HC10','10 DC': 'DC10'})
    # alphaBetaFull['board'] = alphaBetaFull['board'].map({'6 MC full': 'MC6 full', '6 MC truncated': 'MC6 truncated','10 MC full': 'MC10 full','10 MC truncated':'MC10 truncated','6 HC full': 'HC6 full', '6 HC truncated':'HC6 truncated','10 HC full':'HC10 full','10 HC truncated':'HC10 truncated', '10 DC full': 'DC10 full','10 DC truncated':'DC10 truncated'})
    # ax = sns.factorplot(x="board", y="moves",  scale= 0.5, data=alphaBetaFull, hue="heuristic_name", n_boot=1000, order=['MC6 full', 'MC6 truncated','MC10 full','MC10 truncated','HC6 full', 'HC6 truncated','HC10 full','HC10 truncated', 'DC10 full','DC10 truncated'],  markers=["<","1","2","3","4","*"],linestyles=["-","-","-","-","-", "--"], legend_out=False, legend=False)
    # ax.fig.get_axes()[0].set_yscale('log')
    # # print alphaBetaFull['moves']
    # # ax.ax.show()
    # plt.ylim(0, 200000)
    # sns.plt.xlim(0, None)


    # ax.set(xlabel='Board', ylabel='Number of Moves')
    # lw = ax.ax.lines[0].get_linewidth()
    # plt.setp(ax.ax.lines,linewidth=lw)
    # plt.legend(loc='best')
    # plt.show()
    #heatmap distance----
    # distances['scoring'] = distances['scoring'].map({'mcts':'mcts','density':'density', 'linear':  'linear','non-linear':'non-linear', 'non-linear-interaction': 'non-linear'+'\n'+'interaction','blocking':'blocking'})
    # ax = sns.barplot(x="scoring", y="distance", data=distances, order=['mcts','density', 'linear', 'non-linear','non-linear'+'\n'+'interaction','blocking'])
    # # ax = sns.factorplot(x="board", y="moves",  data=alphaBetaFull, hue="heuristic_name", n_boot=1000, order=['6 MC full', '6 MC truncated','10 MC full','10 MC truncated','6 HC full', '6 HC truncated','10 HC full','10 HC truncated', '10 DC full','10 DC truncated'],  markers=["<","1","2","3","4","*"],linestyles=["-","-","-","-","-", "--"], legend_out=False, legend=False)
    # # ax.fig.get_axes()[0].set_yscale('log')
    # # # print alphaBetaFull['moves']
    # # # ax.ax.show()
    # # plt.ylim(0, 200000)
    # # sns.plt.xlim(0, None)
    #
    #
    # ax.set(xlabel='Board', ylabel='Distance From Participants First Moves')
    # # lw = ax.ax.lines[0].get_linewidth()
    # # plt.setp(ax.ax.lines,linewidth=lw)
    # # plt.legend(loc='best')
    # # lw = ax.ax.lines[0].get_linewidth()
    # # plt.setp(ax.ax.lines,linewidth=lw)
    # # # plt.legend(loc='best')
    # # lw = ax.ax.lines[0].get_linewidth()
    # # plt.setp(ax.ax.lines,linewidth=lw)
    # # change_width(ax, .35)
    # plt.show()

    # mcts = distances.loc[distances['scoring'] == 'mcts']
    # density = distances.loc[distances['scoring'] == 'density']
    # linear = distances.loc[distances['scoring'] == 'linear']
    # nonlinear = distances.loc[distances['scoring'] == 'non-linear']
    # nonlinearInteraction = distances.loc[distances['scoring'] == 'non-linear-interaction']
    # blocking = distances.loc[distances['scoring'] == 'blocking']
    # density = distances.loc[distances['scoring'] == 'density']
    # print bootstrap_t_pvalue(blocking['distance'].values, mcts['distance'].values)
    # print bootstrap_t_pvalue(blocking['distance'].values, density['distance'].values)
    # print bootstrap_t_pvalue(blocking['distance'].values, linear['distance'].values)
    # print bootstrap_t_pvalue(blocking['distance'].values, nonlinear['distance'].values)
    # print bootstrap_t_pvalue(blocking['distance'].values, nonlinearInteraction['distance'].values)

    # print bootstrap_t_pvalue(nonlinearInteraction['distance'].values, mcts['distance'].values)
    # print bootstrap_t_pvalue(nonlinearInteraction['distance'].values, density['distance'].values)
    # print bootstrap_t_pvalue(nonlinearInteraction['distance'].values, linear['distance'].values)
    # print bootstrap_t_pvalue(nonlinearInteraction['distance'].values, nonlinear['distance'].values)

    # print bootstrap_t_pvalue(nonlinear['distance'].values, mcts['distance'].values)
    # print bootstrap_t_pvalue(nonlinear['distance'].values, density['distance'].values)
    # print bootstrap_t_pvalue(nonlinear['distance'].values, linear['distance'].values)
    # # #
    # print bootstrap_t_pvalue(linear['distance'].values, mcts['distance'].values)
    # print bootstrap_t_pvalue(linear['distance'].values, density['distance'].values)

    #heatmap distance----

    # participant success rate figure-----
    # ax = sns.factorplot(x="board", y="solutionAndValidationCorrectPercent", scale= 0.5, hue="condition", data=data, n_boot=1000, order=['6 MC', '10 MC', '6 HC', '10 HC', '10 DC'],  markers=['o','^'], legend_out=False, legend=False)
    # data['board'] = data['board'].map({'6 MC': 'MC6','10 MC': 'MC10','6 HC': 'HC6','10 HC': 'HC10','10 DC': 'DC10'})
    # ax = sns.barplot(x="board", y="solutionAndValidationCorrectPercent",  hue="condition", data=data, n_boot=1000, order=['MC6', 'MC10', 'HC6', 'HC10', 'DC10'])
    #
    # ax.set(xlabel='Board', ylabel='Percent Correct')
    # # lw = ax.ax.lines[0].get_linewidth()
    # # plt.setp(ax.ax.lines,linewidth=lw)
    #
    # plt.legend(loc='best')
    # plt.show()
    # participant success rate figure end-----

    # # alpha-beta 50 moves success rate-----
    # ax = sns.factorplot(x="board", y="solutionAndValidationCorrectPercent", scale= 0.5, hue="condition", data=data, n_boot=1000, order=['6 medium', '10 medium', '6 hard', '10 hard', '10 CV'],  markers=['o','^'], legend_out=False, legend=False)
    #
    # alphaBeta50['scoring'].rename_categories(['density','linear','non-linear' + '\n'+ 'interaction','non-linear','blocking'],inplace=True)
    # ax = sns.barplot(x="percent correct", y="scoring", n_boot=1000, data=alphaBeta50)
    # # ax.set(axis_labels=["a","b","c","d","e"])
    # # set_axis_labels("a","b","c","d","e")
    # # ax.set_axis_labels("a","b","c","d","e")
    # # ax.set(xlabel='Board', ylabel='Percent Correct')
    # # lw = ax.ax.lines[0].get_linewidth()
    # # plt.setp(ax.ax.lines,linewidth=lw)
    # # plt.legend(loc='best')
    # # change_width(ax, .30)
    # plt.xlim(0, 100)
    # plt.show()
    # alpha-beta 50 moves success rate-----

    # mcts num Nodes CI


    # ax.axes[0][0].legend(loc=1)
    # ax = sns.factorplot(x="size_type", y="entropyNormalized",col="condition", hue="solutionAndValidationCorrect", data=dataEntropy, n_boot=1000, order=['6_easy', '10_easy', '6_hard', '10_hard', '10_medium'])
    # ax = sns.factorplot(x="solutionAndValidationCorrect", y="actionsSolution", data=data, n_boot=1000)
    # correct = data.loc[data['solutionAndValidationCorrect'] == 1]
    # wrong = data.loc[data['solutionAndValidationCorrect'] == 0]
    # print bs.bootstrap(correct['timeMinutes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(wrong['timeMinutes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bootstrap_t_pvalue(wrong['timeMinutes'].values, correct['timeMinutes'].values)

    # print bs.bootstrap(correct['timePerMove'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(wrong['timePerMove'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bootstrap_t_pvalue(wrong['timePerMove'].values, correct['timePerMove'].values)

    # densityPop = population.loc[population['heuristic'] == 'density']
    # linearPop = population.loc[population['heuristic'] == 'linear']
    # nonLinearPop = population.loc[population['heuristic'] == 'non-linear']
    # nonLinearInteractionPop = population.loc[population['heuristic'] == 'non-linear-interaction']
    # blockingPop = population.loc[population['heuristic'] == 'blocking']
    # # print bootstrap_t_pvalue(wrong['actionsSolution'].values, correct['actionsSolution'].values)
    # print bs.bootstrap(densityPop['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(linearPop['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearPop['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(nonLinearInteractionPop['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # print bs.bootstrap(blockingPop['value'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # # print bs.bootstrap(wrong['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # correct = correct['actionsSolution']
    # wrong = wrong['actionsSolution']
    #
    #
    # #
    # easy_full_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'medium') & (data['condition'] == 'full')]
    # easy_pruned_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'medium') & (data['condition'] == 'truncated')]
    # hard_full_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'hard') & (data['condition'] == 'full')]
    # hard_pruned_6 = data.loc[(data['board_size'] == 6) &(data['board_type'] == 'hard') & (data['condition'] == 'truncated')]
    # easy_full_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'medium') & (data['condition'] == 'full')]
    # easy_pruned_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'medium') & (data['condition'] == 'truncated')]
    # hard_full_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'hard') & (data['condition'] == 'full')]
    # hard_pruned_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'hard') & (data['condition'] == 'truncated')]
    # medium_full_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'CV') & (data['condition'] == 'full')]
    # medium_pruned_10 = data.loc[(data['board_size'] == 10) &(data['board_type'] == 'CV') & (data['condition'] == 'truncated')]
    # #
    # easy_full = data.loc[(data['board_type'] == 'medium') & (data['condition'] == 'full')]
    # hard_full = data.loc[(data['board_type'] == 'hard') & (data['condition'] == 'full')]
    #
    # full_boards1 =  data.loc[data['condition'] == 'full']
    # # print bs.bootstrap(full_boards1['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # easy_6 = data.loc[(data['board_size'] == 6) & (data['board_type'] == 'medium')]
    # hard_6 = data.loc[(data['board_size'] == 6) & (data['board_type'] == 'hard')]
    #
    # easy_10 = data.loc[(data['board_size'] == 10) & (data['board_type'] == 'medium')]
    # hard_10 = data.loc[(data['board_size'] == 10) & (data['board_type'] == 'hard')]


    #mcts----
    #
    # mcts_10_easy_full = mctsData.loc[mctsData['board'] == '10_easy_full']
    # mcts_10_easy_pruned = mctsData.loc[mctsData['board'] == '10_easy_pruned']
    #
    # mcts_6_easy_full = mctsData.loc[mctsData['board'] == '6_easy_full']
    # mcts_6_easy_pruned = mctsData.loc[mctsData['board'] == '6_easy_pruned']
    #
    #
    # mcts_6_hard_full = mctsData.loc[mctsData['board'] == '6_hard_full']
    # mcts_6_hard_pruned = mctsData.loc[mctsData['board'] == '6_hard_pruned']
    #
    # mcts_10_hard_full = mctsData.loc[mctsData['board'] == '10_hard_full']
    # mcts_10_hard_pruned = mctsData.loc[mctsData['board'] == '10_hard_pruned']
    #
    # print bs.bootstrap(mcts_6_easy_full['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_6_easy_pruned['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_10_easy_full['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_10_easy_pruned['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_6_hard_full['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_6_hard_pruned['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_10_hard_full['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(mcts_10_hard_pruned['nodes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap_ab(mcts_6_easy_full['nodes'].values,easy_full_6['solutionAndValidationCorrect'].values, bs_stats.mean, bs_compare.difference)
    #mcts----

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
    # print bs.bootstrap(medium_full_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(medium_pruned_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
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
    #
    # print bs.bootstrap(medium_full_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(medium_pruned_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print '10 CV full vs truncated'
    # print bs.bootstrap_ab( medium_pruned_10['actionsSolution'].values,medium_full_10['actionsSolution'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(medium_full_10['actionsSolution'].values, medium_pruned_10['actionsSolution'].values)


    # print '6 medium full vs truncated'
    # print bs.bootstrap_ab( easy_pruned_6['timeMinutes'].values,easy_full_6['timeMinutes'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(easy_full_6['timeMinutes'].values, easy_pruned_6['timeMinutes'].values)
    #
    # print '10 medium full vs truncated'
    # print bs.bootstrap_ab( easy_pruned_10['timeMinutes'].values,easy_full_10['timeMinutes'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(easy_full_10['timeMinutes'].values, easy_pruned_10['timeMinutes'].values)
    #
    # print '6 hard full vs truncated'
    # print bs.bootstrap_ab( hard_pruned_6['timeMinutes'].values, hard_full_6['timeMinutes'].values,bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(hard_full_6['timeMinutes'].values, hard_pruned_6['timeMinutes'].values)
    #
    # print '10 hard full vs truncated'
    # print bs.bootstrap_ab( hard_pruned_10['timeMinutes'].values,hard_full_10['timeMinutes'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(hard_full_10['timeMinutes'].values, hard_pruned_10['timeMinutes'].values)
    #
    #
    # print bs.bootstrap(medium_full_10['timeMinutes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(medium_pruned_10['timeMinutes'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print '10 CV full vs truncated'
    # print bs.bootstrap_ab( medium_pruned_10['timeMinutes'].values,medium_full_10['timeMinutes'].values, bs_stats.mean, bs_compare.difference)
    # bootstrap_t_pvalue(medium_full_10['timeMinutes'].values, medium_pruned_10['timeMinutes'].values)

    # --- full vs truncated accuracy and num of actions end

    # success rates and number of moves participants
    # print bs_compare.difference(wrong.mean(), correct.mean())
    # print bs.bootstrap_ab(wrong.as_matrix(), correct.as_matrix(), bs_stats.mean, bs_compare.difference)
    #
    # bootstrap_t_pvalue(easy_full_6['solutionAndValidationCorrect'].values, easy_pruned_6['solutionAndValidationCorrect'].values)

    # print bs.bootstrap(easy_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full_6['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full_10['solutionAndValidationCorrect'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
    # print bs.bootstrap(easy_full_6['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(easy_full_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full_6['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    # print bs.bootstrap(hard_full_10['actionsSolution'].values, stat_func=bs_stats.mean, is_pivotal=False)
    #
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