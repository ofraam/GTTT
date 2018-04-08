import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import copy
import csv
from sklearn.covariance import EllipticEnvelope
from  sklearn.neighbors import LocalOutlierFactor
from altair import Chart
from IPython.display import display
from sklearn.linear_model import LinearRegression
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from replay import *
# import Tkinter
import ast


# these are the files with user data foeach of the board
LOGFILE = ['6_hard_full','6_hard_pruned','10_hard_full','10_hard_pruned', '6_easy_full','6_easy_pruned','10_easy_full','10_easy_pruned', '10_medium_full','10_medium_pruned']
# these are the boards starting positions (1 = X, 2 = O)
START_POSITION = [[[0,2,0,0,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[0,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
                  [[0,2,0,1,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[2,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
                  [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,0,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[0,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
                 [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,1,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[2,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
                  [[0,1,0,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,0],[2,0,1,1,2,0],[1,0,2,2,0,0],[0,0,0,0,0,0]],
                  [[0,1,2,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,0],[2,0,1,1,2,1],[1,0,2,2,0,0],[0,0,0,0,0,0]],
                [[0,0,0,0,1,0,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
                  [[0,0,0,0,1,2,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,1],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
                [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]],
                 [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,1,1,1,2,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]]
                  ]


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


def get_user_stats():
    dataset = pd.read_csv("stats/dynamics.csv")
    exploreExploitData = pd.read_csv("stats/exploreExploit2603_avg.csv")
    all_data = []
    curr_data = {}
    for userid in dataset['userid'].unique():
        user_data = dataset.loc[dataset['userid'] == userid]
        exploreExploitData_user = exploreExploitData.loc[exploreExploitData['userid'] == userid]
        curr_data['userid'] = userid

        curr_data['num_resets'] = user_data[user_data['action'] == 'reset'].shape[0]
        curr_data['num_restarts'] = user_data[(user_data['prev_action'] != '') & (user_data['move_number_in_path'] == 1)].shape[0]
        curr_data['mean_score'] = user_data['score_move'].mean()
        curr_data['median_score'] = user_data['score_move'].median()
        curr_data['num_moves_win_score'] = user_data[(user_data['score_move'] == 10000) & (user_data['player'] == 1)].shape[0]
        first_moves = user_data.loc[user_data['move_number_in_path'] == 1]
        curr_data['num_unique_first_moves'] = len(first_moves['position'].unique())
        curr_data['solved'] = user_data.iloc[0]['solved']
        curr_data['board'] = user_data.iloc[0]['board_name']
        curr_data['number_of_moves'] = user_data['move_number'].max()
        curr_data['solution_time'] = user_data['time_rel_sec'].max()
        curr_data['explore_time'] = None
        curr_data['exploit_time'] = None
        curr_data['avg_first_move_score'] = first_moves['score_move'].mean()
        curr_data['median_first_move_score'] = first_moves['score_move'].median()
        if exploreExploitData_user.shape[0]>0:
            curr_data['explore_time'] = exploreExploitData_user.iloc[0]['explore_time']
            curr_data['exploit_time'] = exploreExploitData_user.iloc[0]['exploit_time']
        all_data.append(copy.deepcopy(curr_data))
    dataFile = open('stats\user_stats2603_3.csv', 'wb')
    dataWriter = csv.DictWriter(dataFile, fieldnames=curr_data.keys(), delimiter=',')
    dataWriter.writeheader()
    for record in all_data:
        dataWriter.writerow(record)

def compare_start_move(dynamics):
    first_move = {'6hard':'0_3', '6easy':'3_5', '10hard':'2_4', '10easy':'4_9', '10medium':'3_2'}
    second_move = {'6hard':'3_0', '6easy':'0_2', '10hard':'6_0', '10easy':'0_5', '10medium':'3_5'}
    # hard6pruned = dynamics.loc[(dynamics['board_name']=='6_hard_pruned') & (dynamics['move_number_in_path']==1)]
    # hard6full = dynamics.loc[dynamics['board_name']=='6_hard_full' ]

    # hard6full['prev_pos'] = hard6full['position'].shift()
    # hard6full.ix[0,'prev_pos'] =
    dynamics['first_pruned'] = False
    for index, row in dynamics.iterrows():
        if (row['move_number_in_path'] == 3) & (row['condition'] == 'full'):
            board = row['sizeType']
            # print board
            prev_row = dynamics.iloc[[index-1]]
            prev_prev_row = dynamics.iloc[[index-2]]
            # print prev_row['position'].values[0]
            # print prev_prev_row['position'].values[0]
            if (prev_row['position'].values[0] == second_move[board]) & (prev_prev_row['position'].values[0] == first_move[board]):
                # print 'boo'
                dynamics.loc[index,'first_pruned'] = True

    dynamics.to_csv('stats/dynamicsFirstMoves.csv')

        # print row

def probs_clicks(dynamics):
    data_matrics = {}
    full = dynamics.loc[(dynamics['condition'] == 'full') & (dynamics['first_pruned'] == True) & (dynamics['action'] == 'click')]
    pruned = dynamics.loc[(dynamics['condition'] == 'pruned') & (dynamics['move_number_in_path'] == 1) & (dynamics['action'] == 'click')]
    boards_full = full.board_name.unique()
    for i in range(len(LOGFILE)):
        board = LOGFILE[i]
        board_matrix = copy.deepcopy(START_POSITION[i])

        board_data = full
        if board.endswith('pruned'):
            board_data = pruned

        first_moves = board_data.loc[board_data['board_name'] == board]
        total_first_moves = first_moves.shape[0]

        print total_first_moves
        for row in range(len(board_matrix)):
            for col in range(len(board_matrix[row])):
                if board_matrix[row][col] == 1:
                    board_matrix[row][col] = -0.00001
                elif board_matrix[row][col] == 2:
                    board_matrix[row][col] = -0.00002
                elif total_first_moves > 0:
                    num = first_moves[(first_moves['row'] == row) & (first_moves['col'] == col)].shape[0]
                    board_matrix[row][col] = float(num)/float(total_first_moves)

        data_matrics[board] = copy.deepcopy(board_matrix)

    write_matrices_to_file(data_matrics, 'data_matrices/cogsci/first_pruned.json')


def fit_heuristic_user_moves(transitions,dynamics):
    epsilon = 0.0001
    userids = []
    likelihoods_block = []
    likelihoods_int = []
    likelihoods_dens = []
    heuristic = []
    boards = []
    move_numbers = []

    for userid in dynamics['userid'].unique():
        user_data = dynamics.loc[(dynamics['userid'] == userid) & (dynamics['action'] == 'click')]
        if user_data.shape[0] > 0:

            log_likelihoods_block = 0.0
            log_likelihoods_interaction = 0.0
            log_likelihoods_density = 0.0
            prob_user_block = 1.0
            prob_user_interaction = 1.0
            prob_user_density = 1.0
            paths = []
            curr_path = []
            counter = 0.0
            for index, row in user_data.iterrows():
                transitions_board = transitions.loc[transitions['sizeType'] == row['sizeType']]

                move = row['position'].split('_')
                row_pos = move[0]
                col_pos = move[1]
                board_state = row['board_state']
                state = np.array(ast.literal_eval(board_state))
                probs_data = transitions_board.loc[transitions_board['board_state'] == board_state]
                probs_block = np.array(ast.literal_eval(probs_data['probs_blocking'].iloc[0]))
                probs_interaction = np.array(ast.literal_eval(probs_data['probs_interaction'].iloc[0]))
                probs_density = np.array(ast.literal_eval(probs_data['probs_density'].iloc[0]))

                last_move = row['position'].split('_')
                row_pos = int(last_move[0])
                col_pos = int(last_move[1])
                prob_block = probs_block[row_pos][col_pos]
                if prob_block == 0:
                    prob_block = epsilon
                prob_interaction = probs_interaction[row_pos][col_pos]
                if prob_interaction == 0:
                    prob_interaction = epsilon
                prob_density = probs_density[row_pos][col_pos]
                if prob_density == 0:
                    prob_density = epsilon
                prob_user_block = prob_user_block*prob_block
                prob_user_interaction = prob_user_interaction*prob_interaction
                prob_user_density = prob_user_density*prob_density
                # sum_user_likelihoods +/= comm_prob
                # print prob_block
                likelihoods_block.append(math.log(prob_block))
                boards.append(user_data['board_name'].iloc[0])
                userids.append(userid)
                move_numbers.append(row['move_number_in_path'])
                likelihoods_int.append(math.log(prob_interaction))
                # boards.append(user_data['board_name'].iloc[0])
                # userids.append(userid)
                # move_numbers.append(row['move_number_in_path'])
                likelihoods_dens.append(math.log(prob_density))
                # boards.append(user_data['board_name'].iloc[0])
                # userids.append(userid)
                # move_numbers.append(row['move_number_in_path'])
                counter += 1.0
                # break

            likelihood_vals = []


            # print log_likelihoods_block/counter
            # print log_likelihoods_interaction/counter
            # print log_likelihoods_density/counter
            # if max([log_likelihoods_block,log_likelihoods_interaction,log_likelihoods_density]) == log_likelihoods_block:
            #     heuristic.append('blocking')
            # elif max([log_likelihoods_block,log_likelihoods_interaction,log_likelihoods_density]) == log_likelihoods_interaction:
            #     heuristic.append('interaction')
            # else:
            #     heuristic.append('density')
    heuristic_vals = {'board':boards, 'userid':userids,'likelihoods_block':likelihoods_block,'likelihoods_interaction':likelihoods_int,'likelihoods_density':likelihoods_dens, 'move_number_in_path':move_numbers}
    heuristics_df = pd.DataFrame(heuristic_vals)
    heuristics_df.to_csv('stats/heuristics_fitted_by_move_oWeight=0.csv')


def fit_heuristic_user(transitions,dynamics):
    epsilon = 0.0001
    userids = []
    likelihoods_block = []
    likelihoods_int = []
    likelihoods_dens = []
    heuristic = []
    boards = []

    for userid in dynamics['userid'].unique():
        user_data = dynamics.loc[(dynamics['userid'] == userid) & (dynamics['action'] == 'click')]
        if user_data.shape[0] > 0:
            boards.append(user_data['board_name'].iloc[0])
            log_likelihoods_block = 0.0
            log_likelihoods_interaction = 0.0
            log_likelihoods_density = 0.0
            prob_user_block = 1.0
            prob_user_interaction = 1.0
            prob_user_density = 1.0
            paths = []
            curr_path = []
            counter = 0.0
            for index, row in user_data.iterrows():
                transitions_board = transitions.loc[transitions['size_type'] == row['size_type']]

                move = row['position'].split('_')
                row_pos = move[0]
                col_pos = move[1]
                board_state = row['board_state']
                state = np.array(ast.literal_eval(board_state))
                probs_data = transitions_board.loc[transitions_board['board_state'] == board_state]
                probs_block = np.array(ast.literal_eval(probs_data['probs_blocking'].iloc[0]))
                probs_interaction = np.array(ast.literal_eval(probs_data['probs_interaction'].iloc[0]))
                probs_density = np.array(ast.literal_eval(probs_data['probs_density'].iloc[0]))

                last_move = row['position'].split('_')
                row_pos = int(last_move[0])
                col_pos = int(last_move[1])
                prob_block = probs_block[row_pos][col_pos]
                if prob_block == 0:
                    prob_block = epsilon
                prob_interaction = probs_interaction[row_pos][col_pos]
                if prob_interaction == 0:
                    prob_interaction = epsilon
                prob_density = probs_density[row_pos][col_pos]
                if prob_density == 0:
                    prob_density = epsilon
                prob_user_block = prob_user_block*prob_block
                prob_user_interaction = prob_user_interaction*prob_interaction
                prob_user_density = prob_user_density*prob_density
                # sum_user_likelihoods +/= comm_prob
                # print prob_block
                log_likelihoods_block += math.log(prob_block)
                log_likelihoods_interaction += math.log(prob_interaction)
                log_likelihoods_density += math.log(prob_density)
                counter += 1.0
                # break

            userids.append(userid)
            likelihood_vals = []

            likelihoods_block.append(log_likelihoods_block/counter)
            likelihoods_int.append(log_likelihoods_interaction/counter)
            likelihoods_dens.append(log_likelihoods_density/counter)
            # print log_likelihoods_block/counter
            # print log_likelihoods_interaction/counter
            # print log_likelihoods_density/counter
            if max([log_likelihoods_block,log_likelihoods_interaction,log_likelihoods_density]) == log_likelihoods_block:
                heuristic.append('blocking')
            elif max([log_likelihoods_block,log_likelihoods_interaction,log_likelihoods_density]) == log_likelihoods_interaction:
                heuristic.append('interaction')
            else:
                heuristic.append('density')
    heuristic_vals = {'board':boards, 'userid':userids,'likelihoods_block':likelihoods_block,'likelihoods_interaction':likelihoods_int,'likelihoods_density':likelihoods_dens,'heuristic':heuristic}
    heuristics_df = pd.DataFrame(heuristic_vals)
    heuristics_df.to_csv('stats/heuristics_fitted.csv')


def fit_heuristic_user_path(transitions,dynamics):
    epsilon = 0.0001
    for userid in dynamics['userid'].unique():
        user_data = dynamics.loc[(dynamics['userid'] == userid) & (dynamics['action'] == 'click')]
        log_likelihoods_block = 0.0
        log_likelihoods_interaction = 0.0
        log_likelihoods_density = 0.0
        paths = []
        curr_path = []
        for index, row in user_data.iterrows():
            transitions_board = transitions.loc[transitions['size_type'] == row['size_type']]
            move = row['position'].split('_')
            row_pos = move[0]
            col_pos = move[1]
            board_state = row['board_state']
            state = np.array(ast.literal_eval(board_state))


            path_data = np.array(ast.literal_eval(row['path']))
            for pos in path_data:
                row_pos = pos[0]
                col_pos = pos[1]
                state[row_pos][col_pos] = 0

            probs_data = transitions_board.loc[transitions_board['board_state'] == str(state)]
            probs_block = np.array(ast.literal_eval(probs_data['probs_blocking'].iloc[0]))
            probs_interaction = np.array(ast.literal_eval(probs_data['probs_interaction'].iloc[0]))
            probs_density = np.array(ast.literal_eval(probs_data['probs_density'].iloc[0]))


            prob_path_block = 1.0
            prob_path_interaction = 1.0
            prob_path_density = 1.0
            for pos in path_data:
                row_pos = pos[0]
                col_pos = pos[1]
                prob_block = probs_block[row_pos][col_pos]
                if prob_block == 0:
                    prob_block = epsilon
                prob_interaction = probs_interaction[row_pos][col_pos]
                if prob_interaction == 0:
                    prob_interaction = epsilon
                prob_density = probs_density[row_pos][col_pos]
                if prob_density == 0:
                    prob_density = epsilon
                prob_path_block = prob_path_block*prob_block
                prob_path_interaction = prob_path_interaction*prob_interaction
                prob_path_density = prob_path_density*prob_density

            last_move = row['position'].split('_')
            row_pos = int(last_move[0])
            col_pos = int(last_move[0])
            prob_block = probs_block[row_pos][col_pos]
            if prob_block == 0:
                prob_block = epsilon
            prob_interaction = probs_interaction[row_pos][col_pos]
            if prob_interaction == 0:
                prob_interaction = epsilon
            prob_density = probs_density[row_pos][col_pos]
            if prob_density == 0:
                prob_density = epsilon
            prob_path_block = prob_path_block*prob_block
            prob_path_interaction = prob_path_interaction*prob_interaction
            prob_path_density = prob_path_density*prob_density
            # sum_user_likelihoods +/= comm_prob
            print prob_path_block
            log_likelihoods_block += math.log(prob_path_block)
            log_likelihoods_interaction += math.log(prob_path_interaction)
            log_likelihoods_density += math.log(prob_path_density)

        print userid
        print log_likelihoods_block
        print log_likelihoods_interaction
        print log_likelihoods_density


def compute_path_likelihood_mc():
    simulation_data = pd.read_csv("stats/paths_simulations2000.csv")
    participant_data = pd.read_csv("dynamics06042018")
    userids = []
    path_nums = []
    probs = []
    likelihoods = []
    sim_heuristics = []
    paths = []
    path_lengths = []
    boards = []

    for userid in participant_data['userid'].unique():
        user_data = participant_data.loc[(participant_data['userid'] == userid) & (participant_data['action'] == 'click')]
        if user_data.shape[0] == 0:  # no user clicks
            continue
        for path_num in user_data['path_number'].unique():
            path_data = user_data.loc[(user_data['move_number_in_path'] == user_data['move_number_in_path'].max())]
            path_str = path_data['path_after'].iloc[0]
            path = np.array(ast.literal_eval(path_str))
            board_sim_data = simulation_data.loc[simulation_data['board_name'] == path_data['board_name'].iloc[0]]
            prob_path = 0.0
            path_sim_data = board_sim_data.loc[board_sim_data['path'] == path_str]
            if path_sim_data.shape[0] > 0:
                prob_path = path_sim_data['probability']
            userids.append(userid)
            path_nums.append(path_num)
            paths.append(path)
            path_lengths.append(len(path))
            probs.append(prob_path)
            likelihoods.append(math.log(prob_path))
            boards.append(path_data['board_name'].iloc[0])

    data_dict = {'board':boards, 'userid':userids, 'path_number': path_nums, 'path':paths, 'path_length':path_lengths, 'probability_block':probs, 'likelihood_block':likelihoods}
    paths_probs_df = pd.DataFrame(data_dict)
    paths_probs_df.to_csv('stats/participants_path_probabilities_simulation.csv')


def compute_path_probabilities_participants():
    participant_data = pd.read_csv("stats/dynamics06042018.csv")
    board_names = ['6_easy_full','6_easy_pruned', '10_easy_full', '10_easy_pruned','6_hard_full','6_hard_pruned', '10_hard_full', '10_hard_pruned',  '10_medium_full', '10_medium_pruned']
    boards = []
    path_lengths = []
    probs = []
    counts = []
    paths = []
    # path_numbers = []

    for board_name in board_names:
        # for path_number in participant_data['path_number'].unique():
        # path_data_participants = participant_data.loc[(participant_data['action'] == 'click') & (participant_data['board_name'] == board_name) & (participant_data['path_number'] == path_number)]
        path_data_participants = participant_data.loc[(participant_data['action'] == 'click') & (participant_data['board_name'] == board_name)]
        for path_length in participant_data['move_number_in_path'].unique():
            # print path_length

            paths_counts = {}
            num_paths = 0.0
            paths_data = path_data_participants.loc[path_data_participants['move_number_in_path'] == path_length]

            max_vals = path_data_participants.groupby(['userid','path_number'], as_index=False)['move_number_in_path'].max()
            for index, row in paths_data.iterrows():
                max_val = max_vals.loc[(max_vals['userid'] == row['userid']) & (max_vals['path_number'] == row['path_number'])]
                if row['move_number_in_path'] != max_val['move_number_in_path'].iloc[0]:
                    continue
                path_str = row['path_after']

                if path_str in paths_counts:
                    paths_counts[path_str] += 1.0
                else:
                    paths_counts[path_str] = 1.0

                num_paths += 1.0

            for p, count in paths_counts.iteritems():
                path = np.array(ast.literal_eval(p))
                paths.append(p)
                path_lengths.append(len(path))
                counts.append(count)
                probs.append(count/num_paths)
                # path_numbers.append(path_number)
                boards.append(board_name)


    # data_dict = {'board':boards, 'path_length': path_lengths, 'path':paths, 'probability': probs, 'counts': counts, 'path_number':path_numbers}
    data_dict = {'board':boards, 'path_length': path_lengths, 'path':paths, 'probability': probs, 'counts': counts,}

    paths_probs_df = pd.DataFrame(data_dict)
    paths_probs_df.to_csv('stats/participants_path_probabilities_subpaths2.csv')


def compare_distributions_simulation_population():
    simulation_data = pd.read_csv("stats/paths_simulations_interaction_density.csv")
    participant_data = pd.read_csv("stats/participants_path_probabilities_subpaths2.csv")

    board_names = ['6_easy_full','6_easy_pruned', '10_easy_full', '10_easy_pruned','6_hard_full','6_hard_pruned', '10_hard_full', '10_hard_pruned',  '10_medium_full', '10_medium_pruned']

    boards = []
    path_lengths = []
    wass_dist = []

    for board in board_names:
        print board
        simulation_data_filtered = simulation_data.loc[(simulation_data['board_name'] == board)]
        for path_length in simulation_data_filtered['path_length'].unique():
            print path_length
            paths_dict = {}
            simulation_data_board = simulation_data.loc[(simulation_data['board_name'] == board) & (simulation_data['path_length'] == path_length)]
            participant_data_board = participant_data.loc[(participant_data['board_name'] == board) & (participant_data['path_length'] == path_length)]
            data_size = participant_data_board['counts'].sum()
            probs_participants = []
            probs_sim = []
            for index, row in participant_data_board.iterrows():
                p = row['path']
                if str(p) not in paths_dict.keys():
                    paths_dict[str(p)] = 1
                prob_participant = row['probability']
                prob_sim = 0.0

                path_sim = simulation_data_board.loc[simulation_data_board['path'] == str(p)]
                if path_sim.shape[0] > 0:
                    prob_sim = path_sim['probability'].iloc[0]
                # probs_participants.append(prob_participant*data_size)
                probs_participants.append(prob_participant)
                probs_sim.append(prob_sim)
                # probs_sim.append(prob_sim*data_size)

            for index, row in simulation_data_board.iterrows():
                p = row['path']
                if str(p) not in paths_dict.keys():
                    probs_participants.append(0.0)
                    # probs_sim.append(row['probability']*data_size)
                    probs_sim.append(row['probability'])

            boards.append(board)
            path_lengths.append(path_length)
            wass_dist.append(stats.wasserstein_distance(probs_participants, probs_sim))
            print stats.wasserstein_distance(probs_participants, probs_sim)

    data_dict = {'board':boards, 'path_length': path_lengths, 'wasserstein': wass_dist}
    paths_probs_df = pd.DataFrame(data_dict)
    paths_probs_df.to_csv('stats/wasserstein_interactionDensityVsParticipants.csv')



# def compare_distributions_simulation_population():
#     simulation_data = pd.read_csv("stats/paths_simulations2000.csv")
#     participant_data = pd.read_csv("dynamics06042018")
#
#     probs_sim = []
#     probs_participants = []
#     paths_dict = {}
#     board_names = []
#
#     for board_name in board_names:
#         path_data_participants = participant_data.loc[(participant_data['move_number_in_path'] == participant_data['move_number_in_path'].max()) & (participant_data['action'] == 'click') & (participant_data['board_name'] == board_name)]
#         path_data_simulation = simulation_data.loc[simulation_data['board_name'] == board_name]
#
#         for index, row in path_data_participants.iterrows():
#
#
#
#         if user_data.shape[0] == 0:  # no user clicks
#             continue
#         for path_num in user_data['path_number'].unique():
#             path_data = user_data.loc[(user_data['move_number_in_path'] == user_data['move_number_in_path'].max())]
#             path_str = path_data['path_after'].iloc[0]
#             path = np.array(ast.literal_eval(path_str))
#             board_sim_data = simulation_data.loc[simulation_data['board_name'] == path_data['board_name'].iloc[0]]
#             prob_path = 0.0
#             path_sim_data = board_sim_data.loc[board_sim_data['path'] == path_str]
#             if path_sim_data.shape[0] > 0:
#                 prob_path = path_sim_data['probability']
#             userids.append(userid)
#             path_nums.append(path_num)
#             paths.append(path)
#             path_lengths.append(len(path))
#             probs.append(prob_path)
#             likelihoods.append(math.log(prob_path))
#             boards.append(path_data['board_name'].iloc[0])
#
#     data_dict = {'board':boards, 'userid':userids, 'path_number': path_nums, 'path':paths, 'path_length':path_lengths, 'probability_block':probs, 'likelihood_block':likelihoods,}
#     paths_probs_df = pd.DataFrame(data_dict)
#     paths_probs_df.to_csv('stats/participants_path_probabilities_simulation.csv')



if __name__== "__main__":
    # compute_path_probabilities_participants();
    # # get_user_stats()
    # print stats.wasserstein_distance([1,2,7], [3,1,6])
    # print stats.wasserstein_distance([0.1,0.2,0.7], [0.3,0.1,0.6])
    compare_distributions_simulation_population();
    print 1/0

    data = pd.read_csv("stats/cogSci.csv")
    mctsData = pd.read_csv("stats/mctsRuns.csv")
    dataEntropy = pd.read_csv("stats/cogSciEntropy.csv")
    alphaBetaFull = pd.read_csv("stats/cogsciAlphaBeta100000.csv")
    alphaBeta50 = pd.read_csv("stats/alphaBest50_success.csv")
    distances = pd.read_csv("stats/distanceFirstMoves.csv")
    population = pd.read_csv("stats/cogsciPopulation1.csv")
    likelihood = pd.read_csv("stats/logLikelihood.csv")
    # dynamics = pd.read_csv("stats/dynamics.csv")
    dynamics = pd.read_csv("stats/dynamics06042018.csv")
    transitions = pd.read_csv("stats/transitions=0.csv",dtype = {'board_state':str})

    # fit_heuristic_user_moves(transitions,dynamics)
    # print 1/0

    states = pd.read_csv("stats/states.csv")
    # dynamics = pd.read_csv("stats/dynamicsFirstMoves1.csv")
    # compare_start_move(dynamics)
    # probs_clicks(dynamics)
    # print 1/0

    # exploreExploit = pd.read_csv("stats/exploreExploit0311_avg.csv")
    # exploreExploit = pd.read_csv("stats/exploreExploitPath_avg.csv")
    exploreExploit = pd.read_csv("stats/exploreExploitCombined2.csv")
    # exploreExploit2 = pd.read_csv("stats/exploreExploitData2.csv")
    # exploreExploit2 = pd.read_csv("stats/exploreExploitDataNoUndo.csv")
    timeResets = pd.read_csv("stats/timeBeforeReset.csv")
    timeUndos = pd.read_csv("stats/timeBeforeUndo.csv")
    resetsData = pd.read_csv("stats/resetsData.csv")
    # resetsDelta = pd.read_csv("stats/resetsDeltaData.csv")
    # resetsDelta = pd.read_csv("stats/actionsLogDelta_blocking_abs.csv")
    resetsDelta = pd.read_csv("stats/resetsFiltered2.csv")

    # ten_to_win = ['6_hard_full', '10_hard_full', '10_medium_full']
    # eight_to_win = ['6_hard_pruned', '10_hard_pruned', '10_medium_pruned', '10_easy_full', '6_easy_full']
    # six_to_win = ['6_easy_pruned', '10_easy_pruned']
    # dynamics_filtered = dynamics.loc[dynamics['board_name'] in ten_to_win]
    # max_vals = dynamics.groupby(['userid','path_number', 'board_name', 'sizeType','condition','moves_to_win'], as_index=False)['move_number_in_path'].max()
    max_vals = dynamics.groupby(['userid','path_number'], as_index=False)['move_number_in_path'].max()
    resets = dynamics.loc[(dynamics['action'] == 'reset')]
    # g = sns.FacetGrid(max_vals, row="sizeType", col="condition", legend_out=False)
    # g = sns.FacetGrid(dynamics, row="move_number_in_path", legend_out=False)
    g = sns.FacetGrid(dynamics, col="moves_to_win", legend_out=False)
    # g = g.map(sns.distplot, "move_number_in_path")
    # g = g.map(sns.distplot, "score_move_x")
    bins = np.linspace(0, 15, 15)
    g.map(plt.hist, "move_number_in_path", color="steelblue", bins=bins, lw=0)
    # bins = np.linspace(-100, 100, 200)
    # g.map(plt.hist, "score_move_x", color="steelblue", bins=bins, lw=0)

    plt.show()
    print 1/0

    # -- explore-exploit correlation line
    # user_stats_exploration = pd.read_csv("stats/user_stats2603_3.csv")
    # df = user_stats_exploration[['explore_time','exploit_time']]
    # lof = LocalOutlierFactor()
    # outliers =  lof.fit_predict(df)
    #
    #
    # # ev = EllipticEnvelope(contamination=0.05)
    # # print ev.fit(df)
    # # outliers = ev.predict(df)
    # print len(outliers)
    # user_stats_exploration['outliers'] = outliers
    # # print user_stats_exploration['outliers']
    # user_stats_exploration_filtered = user_stats_exploration.loc[user_stats_exploration['outliers']!=-1]
    # # print user_stats_exploration_filtered['explore_time']
    # ax = sns.regplot(x="explore_time", y="exploit_time", data=user_stats_exploration_filtered, n_boot=1000)
    # plt.show()
    # l = ax.get_lines()
    # x1 = l[0]._path._vertices[0][0]
    # y1 = l[0]._path._vertices[0][1]
    #
    # x2 = l[0]._path._vertices[len(l[0]._path._vertices)-1][0]
    # y2 = l[0]._path._vertices[len(l[0]._path._vertices)-1][1]
    # new_x = []
    # new_y = []
    # for index, row in user_stats_exploration_filtered.iterrows():
    #     x3 = row['explore_time']
    #     y3 = row['exploit_time']
    #     dx = x2 - x1
    #     dy = y2 - y1
    #     d2 = dx*dx + dy*dy
    #     nx = ((x3-x1)*dx + (y3-y1)*dy) / d2
    #     point = (dx*nx + x1, dy*nx + y1)
    #     new_x.append(point[0])
    #     new_y.append(point[1])
    #     # print point
    # user_stats_exploration_filtered['new_x'] = new_x
    # user_stats_exploration_filtered['new_y'] = new_y
    # # ax = sns.regplot(x="new_x", y="new_y", data=user_stats_exploration_filtered, n_boot=1000)
    # min_x_val = user_stats_exploration_filtered['new_x'].min()
    # min_y_val = user_stats_exploration_filtered['new_y'].min()
    # print min_x_val
    # min_explore = math.sqrt((math.pow(min_x_val,2) + math.pow(min_y_val,2)))
    # print min_explore
    # exploration = []
    # for index, row in user_stats_exploration_filtered.iterrows():
    #     distance = math.sqrt((math.pow(row['explore_time']-min_x_val,2) + math.pow(row['exploit_time']-min_y_val,2)))
    #     exploration.append(distance+min_explore)
    # user_stats_exploration_filtered['exploration'] = exploration
    # user_stats_exploration_filtered.to_csv('stats/exploreExploitCombined2.csv')
    # print 1/0
    #
    # lr = LinearRegression()
    # y = user_stats_exploration_filtered.exploration
    # df = user_stats_exploration_filtered[['median_score']]
    # predicted = cross_val_predict(lr, df, y, cv=10)
    # #
    # fig, ax = plt.subplots()
    # ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    # ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    # ax.set_xlabel('Measured')
    # ax.set_ylabel('Predicted')
    # plt.show()

    # plt.clf()
    # ax = sns.distplot(user_stats_exploration_filtered['exploration'], )
    # g = sns.FacetGrid(user_stats_exploration_filtered, hue="condition", legend_out=False)
    # g = g.map(sns.distplot, "exploration")
    #
    # X = user_stats_exploration_filtered
    # y = # Some classes
    #
    # clf = linear_model.Lasso()
    # scores = cross_val_score(clf, X, y, cv=10)
    # g = sns.FacetGrid(user_stats_exploration_filtered, col="correctness", margin_titles=True)
    user_stats_exploration = pd.read_csv("stats/exploreExploitCombined2.csv")
    # user_stats_exploration_filtered =user_stats_exploration.loc[user_stats_exploration['exploration']]
    # user_stats_exploration_correct = user_stats_exploration.loc[user_stats_exploration['solved']=='validatedCorrect']
    # user_stats_exploration_6hard = user_stats_exploration.loc[user_stats_exploration['typeSize']=='6hard']
    g = sns.FacetGrid(user_stats_exploration, col="correct", margin_titles=True)
    # # #
    bins = np.linspace(0, 120, 20)
    # g.map(plt.hist, "exploration", color="steelblue", bins=bins, lw=0)
    g.map(sns.distplot, "exploration", bins=bins)
    test_char = "avg_first_move_score"
    # g.map(sns.regplot,"exploration", test_char);
    # print stats.spearmanr(user_stats_exploration_6hard['exploration'], user_stats_exploration_6hard[test_char])
    #
    # ax = sns.regplot(x="exploration", y="num_resets",data=user_stats_exploration, n_boot=1000)
    plt.show()

    # feature_names = ["num_moves", "solved"]
    # df = pd.DataFrame(user_stats_exploration_filtered, columns=feature_names)
    # print df
    # target = pd.DataFrame(user_stats_exploration_filtered, columns=["exploration"])
    # print target


    # print reg.get_params()
    # print m
    # print b
    # plt.ylim(0,80)
    # plt.show()
    # get_user_stats()

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
    # user_stats_exploration = exploreExploit
    # print stats.spearmanr(user_stats_exploration['explore_time'], user_stats_exploration['exploit_time'])
    # exploreExploit_filtered1 = user_stats_exploration.loc[(user_stats_exploration['explore_time'] < 1000) & (user_stats_exploration['exploit_time'] < 1000) & (user_stats_exploration['solved']=='validatedCorrect')]
    # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    # print exploreExploit_filtered1.shape[0]
    # print stats.spearmanr(exploreExploit_filtered1['explore_time'], exploreExploit_filtered1['exploit_time'])
    # ax = sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered1, n_boot=1000, marker='+', color='green')
    #
    # # plt.xlim(0,100)
    # # plt.ylim(0,100)
    # # plt.show()
    #
    # exploreExploit_filtered2 = user_stats_exploration.loc[(user_stats_exploration['explore_time'] < 1000) & (user_stats_exploration['exploit_time'] < 1000) & ((user_stats_exploration['solved']=='wrong')  | (user_stats_exploration['solved']=='solvedCorrect'))]
    # print stats.spearmanr(exploreExploit_filtered2['explore_time'], exploreExploit_filtered2['exploit_time'])
    # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    # ax = sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered2, n_boot=1000, color='red')
    # # plt.xlim(0,100)
    # # plt.ylim(0,100)
    # plt.show()

    # boards = ['6_easy_full','6_easy_pruned', '10_easy_full', '10_easy_pruned','6_hard_full','6_hard_pruned', '10_hard_full', '10_hard_pruned',  '10_medium_full', '10_medium_pruned']
    # for board in boards:
    #     print board
    #     # sns.load_dataset
    #     exploreExploit_filtered1 = user_stats_exploration.loc[(user_stats_exploration['board']==board)]
    #     colors = {'solvedCorrect':'blue','validatedCorrect':'green', 'wrong':'red' }
    #     plt.scatter(exploreExploit_filtered1.explore_time, exploreExploit_filtered1.exploit_time,
    #                 c = exploreExploit_filtered1.solved_num, s=(exploreExploit_filtered1.num_resets**2), cmap="viridis")
    #     ax = plt.gca()
    #
    #     plt.colorbar(label="solved")
    #     plt.xlabel("explore_time")
    #     plt.ylabel("exploit_time")
    #
    #     #make a legend:
    #     # pws = [0.5, 1, 1.5, 2., 2.5]
    #     # for pw in pws:
    #     #     plt.scatter([], [], s=(pw**2), c="k",label=str(pw))
    #     #
    #     # h, l = plt.gca().get_legend_handles_labels()
    #     # plt.legend(h[1:], l[1:], labelspacing=1.2, title="num_resets", borderpad=1,
    #     #             frameon=True, framealpha=0.6, )
    #
    #     plt.show()
    # for board in boards:
    # exploreExploit_filtered1 = user_stats_exploration.loc[user_stats_exploration['board']==board]
    # exploreExploit_filtered1 = user_stats_exploration
    # exploreExploit_filtered1 = user_stats_exploration.loc[user_stats_exploration['explore_time']<100]
    # ax = sns.lmplot(x="explore_time", y="exploit_time",data=exploreExploit_filtered1,hue='condition', n_boot=1000,fit_reg=False)
    # plt.show()
    # # exploreExploit_filtered1 = user_stats_exploration.loc[user_stats_exploration['board'].str.endswith('full')]
    # colors = {'correct':'green', 'wrong':'red'}
    # cols = ['num_resets','num_unique_first_moves','num_moves_win_score','mean_score','solution_time','median_score','number_of_moves']
    # # mult_vals = [10.0, 50.0,50.0,50.0,5.0,10.0,10.0]
    # for i in range(len(cols)):
    #     # f, (ax1, ax2) = plt.subplots(2)
    #     p = cols[i]
    #     # ax = sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered1, n_boot=1000, scatter_kws={"s": (exploreExploit_filtered1.num_resets**2)})
    #     print p
    #     max_value = exploreExploit_filtered1[p].max()*1.0
    #     filt_min = exploreExploit_filtered1[exploreExploit_filtered1[p]>0.001]
    #     min_value = filt_min[p].min()*1.0
    #     # print exploreExploit_filtered1['norm_val']
    #
    #     mult = ((min_value+0.)**2)/30.0
    #     print mult
    #     ax = sns.lmplot(x="explore_time", y="exploit_time",data=exploreExploit_filtered1,hue='board', n_boot=1000,palette=colors,scatter_kws={"s": ((exploreExploit_filtered1[p])**2 )/mult},fit_reg=False, legend=False)
    #     # plt.scatter(exploreExploit_filtered1.explore_time, exploreExploit_filtered1.exploit_time,
    #     #             c = exploreExploit_filtered1['solved'].apply(lambda x: colors[x]), s=(exploreExploit_filtered1.num_resets**2), cmap="Paired")
    #     # ax = plt.gca()
    #
    #     # plt.colorbar(label="solved")
    #     plt.xlabel("explore_time")
    #     plt.ylabel("exploit_time")
    #
    #     #make a legend:
    #     # pws = [0.5, 1, 1.5, 2., 2.5]
    #     # for pw in pws:
    #     #     plt.scatter([], [], s=(pw**2), c="k",label=str(pw))
    #     #
    #     # h, l = plt.gca().get_legend_handles_labels()
    #     # plt.legend(h[1:], l[1:], labelspacing=1.2, title="num_resets", borderpad=1,
    #     #             frameon=True, framealpha=0.6, )
    #     plt.xlim(0, min(exploreExploit_filtered1['explore_time'].max()+10,100))
    #     plt.ylim(0, min(exploreExploit_filtered1['exploit_time'].max()+10,100))
    #     # plt.xlim(0,60)
    #     # plt.ylim(0,60)
    #     # title = p + '_' +board
    #     title = p + '_full'
    #     plt.title(title)
    #     # plt.show()
    #     # plt.figure(figsize=(20,10))
    #     # ax.set(yscale="symlog")
    #     # ax.set(xscale="symlog")
    #     plt.show()
        # plt.savefig("dynamics/explore_exploit/test/15_explore_exploit_allBoards60_"+ title +".png", format='png')
        # plt.clf()
        # c = Chart(exploreExploit_filtered1)
        # c.mark_circle().encode(
        #     x='explore_time',
        #     y='exploit_time',
        #     color='solved',
        #     size='num_resets',
        # )
        # c.serve()
        # break
        # display(c)
        # print(c.to_json(indent=2))
        # plt.show()
        # exploreExploit_filtered1 = user_stats_exploration.loc[(user_stats_exploration['solved']=='validatedCorrect') & (user_stats_exploration['board']==board)]

        # print stats.spearmanr(exploreExploit_filtered1['explore_time'], exploreExploit_filtered1['exploit_time'])
        # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
        # ax = sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered1, n_boot=1000, color='green')
        # # plt.gca().set_xlim(left=0)
        # # plt.gca().set_ylim(left=0)
        # plt.xlim(0, 130)
        # plt.ylim(0, 180)
        #
        # exploreExploit_filtered2 = user_stats_exploration.loc[((user_stats_exploration['solved']=='wrong') | (user_stats_exploration['solved']=='solvedCorrect')) & (user_stats_exploration['board']==board)]
        # print stats.spearmanr(exploreExploit_filtered2['explore_time'], exploreExploit_filtered2['exploit_time'])
        # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
        # ax = sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered2, n_boot=1000, color='red')
        # plt.xlim(0, 130)
        # plt.ylim(0, 180)
        # # plt.gca().set_xlim(left=0)
        # # plt.gca().set_ylim(left=0)
        # plt.show()

    # print len(users)
    # for user in users:
    #     print user
    # for board in boards:
    #     print board
    #     f, (ax1, ax2) = plt.subplots(2)
    #     exploreExploit_filtered = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100)   & (exploreExploit['board']==board)]
    #     print stats.spearmanr(exploreExploit_filtered['explore_time'], exploreExploit_filtered['exploit_time'])
    #     spear = stats.spearmanr(exploreExploit_filtered['explore_time'], exploreExploit_filtered['exploit_time'])
    #     # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    #     sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered, n_boot=1000, color='blue', ax=ax1)
    #     ax1.set_xlim(0,100)
    #     ax1.set_ylim(0,100)
    #
    #     exploreExploit_filtered1 = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100) & (exploreExploit['solved']=='validatedCorrect') & (exploreExploit['board']==board)]
    #     print stats.spearmanr(exploreExploit_filtered1['explore_time'], exploreExploit_filtered1['exploit_time'])
    #     # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    #     sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered1, n_boot=1000, color='green', ax=ax2)
    #     plt.xlim(0,100)
    #     plt.ylim(0,100)
    #
    #     exploreExploit_filtered2 = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100)  & ((exploreExploit['solved']=='wrong') | (exploreExploit['solved']=='solvedCorrect')) & (exploreExploit['board']==board)]
    #     print stats.spearmanr(exploreExploit_filtered2['explore_time'], exploreExploit_filtered2['exploit_time'])
    #     # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    #     sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered2, n_boot=1000, color='red', ax=ax2)
    #     plt.xlim(0,100)
    #     plt.ylim(0,100)
    #
    #
    #     title = board + '_' + "spearman = " + str(round(spear.correlation,2))
    #     ax1.set_title(title)
    #     # plt.show()
    #     plt.savefig("dynamics/explore_exploit/explore_exploit_paths_"+ title +".png", format='png')

    # f, (ax1, ax2) = plt.subplots(2)
    # exploreExploit_filtered = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100)]
    # print stats.spearmanr(exploreExploit_filtered['explore_time'], exploreExploit_filtered['exploit_time'])
    # spear = stats.spearmanr(exploreExploit_filtered['explore_time'], exploreExploit_filtered['exploit_time'])
    # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    # sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered, n_boot=1000, color='blue', ax=ax1)
    # ax1.set_xlim(0,100)
    # ax1.set_ylim(0,100)
    #
    # exploreExploit_filtered1 = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100) & (exploreExploit['solved']=='validatedCorrect')]
    # # exploreExploit_filtered1 = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100)]
    #
    # print stats.spearmanr(exploreExploit_filtered1['explore_time'], exploreExploit_filtered1['exploit_time'])
    # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    # sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered1, n_boot=1000, color='green', ax=ax2)
    # # ax = sns.regplot(x="explore_exploit_ratio", y="solution_time", data=exploreExploit_filtered1, n_boot=1000, color='green')
    #
    # # plt.xlim(0,100)
    # # plt.ylim(0,100)
    # # plt.show()
    #
    # exploreExploit_filtered2 = exploreExploit.loc[(exploreExploit['explore_time'] < 100) & (exploreExploit['exploit_time'] < 100)  & ((exploreExploit['solved']=='wrong') | (exploreExploit['solved']=='solvedCorrect'))]
    # print stats.spearmanr(exploreExploit_filtered2['explore_time'], exploreExploit_filtered2['exploit_time'])
    # # ax = sns.barplot(x="solved", y="exploit_time", data=exploreExploit)
    # sns.regplot(x="explore_time", y="exploit_time", data=exploreExploit_filtered2, n_boot=1000, color='red', ax=ax2)
    # plt.xlim(0,100)
    # plt.ylim(0,100)
    #
    #
    # title = 'all_boards' + '_' + "spearman = " + str(round(spear.correlation,2))
    # ax1.set_title(title)
    # # plt.show()
    # plt.savefig("dynamics/explore_exploit/explore_exploit_paths_"+ title +".png", format='png')

        # plt.show()

    # ------ states ------
    # entropies = pd.read_csv("stats/entropies_data.csv")
    # # g = sns.FacetGrid(entropies, col="sizeType", legend_out=False)
    # ax = sns.barplot(x="condition",y="entropy", data = entropies)
    # # print moves_s['board_name'].unique()
    # # plt.title("tt")
    # plt.show()
    # states = states.loc[(states['action'] == 'click') & (states['sizeType'] == '10medium')]
    # states = states.loc[(states['action'] == 'click')]
    # board_states = states['board_state'].unique()
    # entropies_pruned = []
    # entropies_full = []
    # states_data = []
    # boards = []
    # solved = []
    # entropies = []
    # for s in board_states:
    #     moves_s = states.loc[(states['board_state'] == s)]
    #     if (len(moves_s['path'].unique()) > 1):
    #         check = True
    #         for p in moves_s['path'].unique():
    #             m_p = moves_s.loc[moves_s['path'] == p]
    #             if m_p.shape[0] < 5:
    #                 check = False
    #                 break
    #         if check:
    #             # vals = moves_s['position'].unique()
    #             #
    #             # g = sns.FacetGrid(moves_s, col="condition", legend_out=False)
    #             # g.map(sns.countplot, "position", order= vals, color="steelblue", lw=0)
    #             # print moves_s['board_name'].unique()
    #             # # plt.title("tt")
    #             # plt.show()
    #
    #             pk = []
    #             moves_pruned = moves_s.loc[moves_s['condition'] == 'pruned']
    #             mp = moves_pruned['position'].unique()
    #             total = moves_pruned.shape[0] + 0.0
    #             for m in mp:
    #                 count = moves_pruned[moves_pruned['position'] == m].shape[0]
    #                 pk.append(float(count)/float(total))
    #             ent = stats.entropy(pk)
    #             entropies_pruned.append(ent)
    #             states_data.append(s)
    #             boards.append(moves_pruned['board_name'].unique()[0])
    #             solvers = moves_pruned[moves_pruned['solved'] == 'validatedCorrect'].shape[0]
    #             solved.append(float(solvers)/total)
    #             entropies.append(ent)
    #             pk = []
    #
    #             moves_full = moves_s.loc[moves_s['condition'] == 'full']
    #             mf = moves_full['position'].unique()
    #             total = moves_full.shape[0] + 0.0
    #             for m in mf:
    #                 count = moves_full[moves_full['position'] == m].shape[0]
    #                 pk.append(float(count)/float(total))
    #             ent = stats.entropy(pk)
    #             entropies_full.append(ent)
    #             states_data.append(s)
    #             solvers = moves_full[moves_full['solved'] == 'validatedCorrect'].shape[0]
    #             solved.append(float(solvers)/total)
    #             boards.append(moves_full['board_name'].unique()[0])
    #             entropies.append(ent)
    # entropies_data = {'board': boards, 'state': states_data, 'entropy':entropies, 'solvers':solved}
    # entropies_data =pd.DataFrame(entropies_data)
    #
    #
    # # plt.title("tt")
    # plt.show()
    # entropies_data.to_csv('stats/entropies_data_solvers.csv')
    # print entropies_data
    # print np.mean(entropies_full)
    # print np.mean(entropies_pruned)
    # entropies_full = np.asarray(entropies_full)
    # entropies_full = entropies.loc[entropies['condition']=='full']
    # entropies_pruned = entropies.loc[entropies['condition']=='pruned']
    # # entropies_pruned = np.asarray(entropies_pruned)
    # print bootstrap_t_pvalue(entropies_pruned['entropy'].values, entropies_full['entropy'].values)
    #--------- reset and undo distributions-----------

    # ax = sns.distplot(timeResets['time_before_sec'])
    # resetsDelta = resetsDelta.loc[resetsDelta['action'] == 'reset']
    # resetsDelta = pd.read_csv('stats/resetsPotential.csv')
    # delta_filtered = dynamics.loc[(dynamics['score_curr_move'] > -100) & (dynamics['score_curr_move'] < 100)]
    # delta_filtered = dynamics.loc[(dynamics['action'] == 'reset')]
    # delta_filtered = dynamics.loc[dynamics['move_number_in_path']==5]
    # # delta_filtered = dynamics.loc[(dynamics['move_number_in_path']>4) & (dynamics['board_name']=='6_hard_full')]
    # # delta_filtered = dynamics.loc[(dynamics['delta_score']>-100) & (dynamics['delta_score']<100) & (dynamics['board_name']=='6_hard_full')]
    # # delta_filtered = dynamics.loc[(dynamics['move_number_in_path']<11) &(dynamics['score_move']>-100) & (dynamics['score_move']<100) ]
    # # delta_filtered = dynamics.loc[ (dynamics['board_name']=='6_hard_full')]
    # #
    # # ax = sns.distplot(delta_filtered['potential_score'])
    #
    # g = sns.FacetGrid(delta_filtered, row="action", legend_out=False)
    # g = g.map(sns.distplot, "score_move_x")
    # bins = np.linspace(-20,20,num=100)
    # g.map(plt.hist, "score_move", color="steelblue",  bins=bins,lw=0)
    # g.map(plt.hist, "score_move", color="steelblue",  lw=0)
    # g.map(sns.regplot,"move_number_in_path", "delta_score");
    # # g.map(plt.hist, "deltaScoreByScore", color="steelblue",  lw=0)
    # ax = sns.regplot(x="move_number_in_path",y= "potential_score", data=delta_filtered)
    # ax = g.ax_joint
    # ax.set_yscale('symlog')
    # # g.set(yscale="symlog")
    # plt.show()
    # timeUndos_filtered = timeUndos.loc[(timeUndos['time_before_sec'] < 10)]
    # timeResets_filtered = timeResets.loc[(timeResets['time_before_sec'] < 10)]
    # # ax = sns.distplot(timeUndos_filtered['time_before_sec'])
    # ax = sns.distplot(timeResets_filtered['time_before_sec'])
    # plt.show()

    # reset events
    #
    # for board in boards:
    #     moves_to_win = 0
    #     if '6' in board:
    #         moves_to_win = 4
    #         if 'pruned' in board:
    #             moves_to_win = 3
    #
    #     elif '10' in board:
    #         moves_to_win = 5
    #         if 'pruned' in board:
    #             moves_to_win = 4
    #
    #     print board
    #     # resetsData_filtered = resetsData.loc[(resetsData['board_name'] == board) & (resetsData['delta_score'] != 99999) & (resetsData['delta_score'] < 1000) & (resetsData['delta_score'] > -1000)]
    #     # resetsData_filtered = resetsData.loc[(resetsData['board_name'] == board)]
    #     resetsData_filtered = resetsData.loc[(resetsData['board_name'] == board) & (resetsData['delta_score'] != 99999) & (resetsData['delta_score'] < 1000) & (resetsData['delta_score'] > -1000) & (resetsData['move_number_in_path']<moves_to_win)]
    #
    #     # print len(resetsData_filtered)
    #     # ax = sns.distplot(resetsData_filtered['move_number_in_path'])
    #     ax = sns.distplot(resetsData_filtered['delta_score'])
    #     # timeUndos_filtered = timeUndos.loc[(timeUndos['time_before_sec'] < 10)]
    #     # timeResets_filtered = timeResets.loc[(timeResets['time_before_sec'] < 10)]
    #     # # ax = sns.distplot(timeUndos_filtered['time_before_sec'])
    #     # ax = sns.distplot(timeResets_filtered['time_before_sec'])
    #     plt.show()

    # dynamics_filtered = dynamics.loc[(dynamics['move_number_in_path'] < 11) & (dynamics['move_number_in_path'] > 1) & (dynamics['player'] == 2)]
    # userids = dynamics['userid'].unique()
    #
    # for user in userids:
    #     # print user
    #     f, (ax1, ax2) = plt.subplots(2, figsize = (20,10))
    #     clicks_filtered = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='click')]
    #     clicks_filtered_p1 = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='click') & (dynamics['player']==1)]
    #     clicks_filtered_p2 = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='click') & (dynamics['player']==2)]
    #
    #     # ax = sns.FacetGrid(dynamics_filtered, row="userid")
    #     # ax = ax.map_dataframe(sns.tsplot, time='time_rel_sec', value='time_between', unit='userid', data=clicks_filtered, interpolate=False)
    #     if (len(clicks_filtered_p1) < 2) | (len(clicks_filtered_p2) < 2):
    #         continue
    #     sns.tsplot(time='time_rel_sec', value='time_from_action', unit='userid', data=clicks_filtered, interpolate=False, ax=ax1)
    #     sns.tsplot(time='time_rel_sec', value='score_move_x', unit='userid', data=clicks_filtered, interpolate=False, color='blue', ax=ax2)
    #     # sns.tsplot(time='time_rel_sec', value='top_possible_score', unit='userid', data=clicks_filtered_p1, interpolate=False, color='orange',  ax=ax2)
    #
    #     # sns.tsplot(time='time_rel_sec', value='score_move', unit='userid', data=clicks_filtered_p2, interpolate=False, color='black', ax=ax2)
    #     # sns.tsplot(time='time_rel_sec', value='top_possible_score', unit='userid', data=clicks_filtered_p2, interpolate=False, color='orange',  ax=ax2)
    #
    #     solved = clicks_filtered['solved'].iloc[0]
    #     board_name = clicks_filtered['board_name'].iloc[0]
    #     resets = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='reset')]
    #
    #     for index, event in resets.iterrows():
    #         time_reset = int(event['time_rel_sec'])
    #         # print time_reset
    #         ax1.axvline(time_reset, color="red", linestyle="--");
    #         ax2.axvline(time_reset, color="red", linestyle="--");
    #         # ax3.axvline(time_reset, color="red", linestyle="--");
    #
    #         solved = event['solved']
    #         board_name = event['board_name']
    #
    #     undos = dynamics.loc[(dynamics['userid']==user) & (dynamics['action']=='undo')]
    #
    #     for index, event in undos.iterrows():
    #         time_undo = int(event['time_rel_sec'])
    #         # print time_undo
    #         ax1.axvline(time_undo, color="purple", linestyle="--");
    #         ax2.axvline(time_undo, color="purple", linestyle="--");
    #         # ax3.axvline(time_undo, color="purple", linestyle="--");
    #
    #
    #
    #     ax2.set(yscale="symlog")
    #     ax2.set_ylim(-100,100)
    #
    #     # ax3.set(yscale="symlog")
    #     # ax3.set_ylim(-100000,100000)
    #     # plt.show()
    #
    #     ax2.set_ylabel('score vs. best')
    #     # ax3.set_ylabel('o score vs. best')
    #
    #     title = user + '_' + solved + '_' + board_name
    #     ax1.set_title(title)
    #     plt.show()
        # plt.savefig("dynamics/time_series3/timeSeries_"+ title +".png", format='png')

        # plt.clf()
        # plt.close()

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