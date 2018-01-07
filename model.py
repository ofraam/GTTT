import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import replay as rp

LOGFILE = ['logs/6_hard_full_dec19.csv','logs/6_hard_pruned_dec19.csv','logs/10_hard_full_dec19.csv','logs/10_hard_pruned_dec19.csv', 'logs/6_easy_full_dec19.csv','logs/6_easy_pruned_dec19.csv','logs/10_easy_full_dec19.csv','logs/10_easy_pruned_dec19.csv','logs/10_medium_full_dec19.csv','logs/10_medium_pruned_dec19.csv']
DIMENSION = 6
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


def compute_density(row, col, board, neighborhood_size):
    x_count = 0.0
    density_score = 0.0
    for i in range(-1*neighborhood_size,neighborhood_size+1):
        for j in range(-1*neighborhood_size,neighborhood_size+1):
            if (i != 0) | (j != 0):
                r = row + i
                c = col + j
                if (r < len(board)) & (r >= 0) & (c < len(board)) & (c >= 0):
                    # print r
                    # print c
                    if board[r][c] == 'X':
                        x_count += 1.0
                        density_score += 1.0/(8*max(abs(i), abs(j)))

    return density_score


def compute_density_guassian(row, col, board, guassian_kernel):
    density_score = 0.0
    for guas in guassian_kernel:
        density_score += guas[row][col]

    return density_score


def compute_scores_density_guassian(normalized=False, sig = 3, lamb = 1):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)


        # create guassians for each X square
        guassian_kernel = []
        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 'X':
                    guassian_kernel.append(makeGaussian(len(board_matrix),fwhm=sig,center=[r,c]))

        # compute density scores
        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    square_score = compute_density_guassian(r,c,board_matrix,guassian_kernel)
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    sum_scores_exp += math.pow(math.e,lamb*square_score)

        # score_matrix_normalized = copy.deepcopy(score_matrix)
        # for r in range(len(score_matrix_normalized)):
        #     for c in range(len(score_matrix_normalized[r])):
        #         score_matrix_normalized[r][c] = score_matrix_normalized[r][c]/sum_scores

        print 'score matrix:'
        print score_matrix
        # print 'score matrix normalized'
        # print score_matrix_normalized

        # heatmaps
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]=='X'):
                    score_matrix[r][j] = -0.00001
                elif (score_matrix[r][j]=='O'):
                    score_matrix[r][j] = -0.00002

        if normalized:
            for r in range(len(score_matrix)):
                for c in range(len(score_matrix[r])):
                    # score_matrix[r][c] = score_matrix[r][c]/sum_scores
                    if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                        score_matrix[r][c] = score_matrix[r][c]/sum_scores
                        # score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp


        a = np.array(score_matrix)
        a = np.flip(a,0)
        print a
        heatmap = plt.pcolor(a)

        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                if(a[y,x]==-1) | (a[y,x]==-0.00001):
                    plt.text(x + 0.5, y + 0.5, 'X',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif((a[y,x]==-2) | (a[y,x]==-0.00002)):
                    plt.text(x + 0.5, y + 0.5, 'O',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif(a[y,x]!=0):
                    plt.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='white'
                     )

        fig = plt.colorbar(heatmap)
        fig_file_name = LOGFILE[g]
        fig_file_name = fig_file_name[:-4]
        data_matrices[fig_file_name[5:-6]] = score_matrix
        if normalized:
            fig_file_name = fig_file_name + '_normalized_density_scores.png'
        else:
            fig_file_name = fig_file_name + '_density_scores.png'

        plt.savefig(fig_file_name)
        plt.clf()
    return data_matrices


def compute_scores_density(normalized=False, neighborhood_size=1, lamb=1):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)

        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    square_score = compute_density(r, c, board_matrix, neighborhood_size)  # check neighborhood
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    sum_scores_exp += math.pow(math.e,lamb*square_score)

        # score_matrix_normalized = copy.deepcopy(score_matrix)
        # for r in range(len(score_matrix_normalized)):
        #     for c in range(len(score_matrix_normalized[r])):
        #         score_matrix_normalized[r][c] = score_matrix_normalized[r][c]/sum_scores

        print 'score matrix:'
        print score_matrix
        # print 'score matrix normalized'
        # print score_matrix_normalized

        # heatmaps
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]=='X'):
                    score_matrix[r][j] = -0.00001
                elif (score_matrix[r][j]=='O'):
                    score_matrix[r][j] = -0.00002

        if normalized:
            for r in range(len(score_matrix)):
                for c in range(len(score_matrix[r])):
                    # score_matrix[r][c] = score_matrix[r][c]/sum_scores
                    if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                        score_matrix[r][c] = score_matrix[r][c]/sum_scores
                        # score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp


        a = np.array(score_matrix)
        a = np.flip(a,0)
        print a
        heatmap = plt.pcolor(a)

        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                if(a[y,x]==-1) | (a[y,x]==-0.00001):
                    plt.text(x + 0.5, y + 0.5, 'X',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif((a[y,x]==-2) | (a[y,x]==-0.00002)):
                    plt.text(x + 0.5, y + 0.5, 'O',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif(a[y,x]!=0):
                    plt.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='white'
                     )

        fig = plt.colorbar(heatmap)
        fig_file_name = LOGFILE[g]
        fig_file_name = fig_file_name[:-4]
        data_matrices[fig_file_name[5:-6]] = score_matrix
        if normalized:
            fig_file_name = fig_file_name + '_normalized_density_scores.png'
        else:
            fig_file_name = fig_file_name + '_density_scores.png'

        plt.savefig(fig_file_name)
        plt.clf()
    return data_matrices


def compute_open_paths(row, col, board, exp=1):
    streak_size = 4
    if len(board) == 10:
        streak_size = 5

    open_paths_lengths = []
    # check right-down diagonal
    for i in range(streak_size):
        r = row - i
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == 'O':
                blocked = True
            elif board[square_row][square_col] == 'X':
                path_x_count += 1

            square_row += 1
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked):
            open_paths_lengths.append(path_x_count+1)

    # check left-down diagonal
    for i in range(streak_size):
        r = row - i
        c = col + i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == 'O':
                blocked = True
            elif board[square_row][square_col] == 'X':
                path_x_count += 1

            square_row += 1
            square_col -= 1
            path_length += 1

        if (path_length == streak_size) & (not blocked):
            open_paths_lengths.append(path_x_count+1)

    # check vertical
    for i in range(streak_size):
        r = row - i
        c = col
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == 'O':
                blocked = True
            elif board[square_row][square_col] == 'X':
                path_x_count += 1

            square_row += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked):
            open_paths_lengths.append(path_x_count+1)

    # check horizontal
    for i in range(streak_size):
        r = row
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == 'O':
                blocked = True
            elif board[square_row][square_col] == 'X':
                path_x_count += 1

            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked):
            open_paths_lengths.append(path_x_count+1)

    print open_paths_lengths

    if len(open_paths_lengths) == 0:
        return 0.0

    open_paths_lengths.sort(reverse=True)

    if len(open_paths_lengths) == 0:
        return 0.0
    if len(open_paths_lengths) == 1:
        if open_paths_lengths[0] == streak_size:
            return 10000000
        score = 1.0/math.pow((streak_size-open_paths_lengths[0]), exp)
        return score
    score = 1.0/(math.pow((streak_size-open_paths_lengths[0]), exp)) + 1.0/(math.pow((streak_size-open_paths_lengths[1]), exp)) \
            + 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(open_paths_lengths[0]*open_paths_lengths[1]), exp))
    return score


def compute_scores_open_paths(normalized=False, exp=1, lamb = 1):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)


        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    square_score = compute_open_paths(r, c, board_matrix,exp=exp)  # check open paths for win
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    sum_scores_exp += math.pow(math.e,lamb*square_score)

        # score_matrix_normalized = copy.deepcopy(score_matrix)
        # for r in range(len(score_matrix_normalized)):
        #     for c in range(len(score_matrix_normalized[r])):
        #         score_matrix_normalized[r][c] = score_matrix_normalized[r][c]/sum_scores

        print 'score matrix:'
        print score_matrix
        # print 'score matrix normalized'
        # print score_matrix_normalized

        # heatmaps
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]=='X'):
                    score_matrix[r][j] = -0.00001
                elif (score_matrix[r][j]=='O'):
                    score_matrix[r][j] = -0.00002

        if normalized:
            for r in range(len(score_matrix)):
                for c in range(len(score_matrix[r])):
                    if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                        score_matrix[r][c] = score_matrix[r][c]/sum_scores
                    # if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                    #     score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp

        a = np.array(score_matrix)
        a = np.flip(a,0)
        print a
        heatmap = plt.pcolor(a)

        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                if(a[y,x]==-1) | (a[y,x]==-0.00001):
                    plt.text(x + 0.5, y + 0.5, 'X',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif((a[y,x]==-2) | (a[y,x]==-0.00002)):
                    plt.text(x + 0.5, y + 0.5, 'O',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif(a[y,x]!=0):
                    plt.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='white'
                     )

        fig = plt.colorbar(heatmap)
        fig_file_name = LOGFILE[g]
        fig_file_name = fig_file_name[:-4]
        data_matrices[fig_file_name[5:-6]] = score_matrix
        if normalized:
            fig_file_name = fig_file_name + '_normalized_path_scores' + 'exp=' + str(exp) + '.png'
        else:
            fig_file_name = fig_file_name + '_path_scores' + 'exp=' + str(exp) + '.png'

        plt.savefig(fig_file_name)
        plt.clf()
    return data_matrices


def compute_scores_composite(normalized=False, exp=1, neighborhood_size=1, density = 'guassian', lamb=None, sig=3):
    data_matrices = {}
    if (density=='guassian'):
        density_scores = compute_scores_density_guassian(True)
    else:
        density_scores = compute_scores_density(True, neighborhood_size=neighborhood_size)
    path_scores = compute_scores_open_paths(True,exp)

    for g in range(len(LOGFILE)):
        board_key = LOGFILE[g]
        board_key = board_key[:-4]
        board_key = board_key[5:-6]

        density_scores_board = density_scores[board_key]
        path_scores_board = path_scores[board_key]

        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)


        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    # if (density=='guassian'):
                    #     square_score_density = compute_density_guassian(r, c, board_matrix, guassian_kernel) # check density
                    # else:
                    #     square_score_density = compute_density(r, c, board_matrix, neighborhood_size) # check density
                    # square_score_path = compute_open_paths(r, c, board_matrix,exp=exp)  # check open paths for win
                    # square_score = square_score_density * square_score_path
                    square_score = density_scores_board[r][c] * path_scores_board[r][c]
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    if lamb != None:
                        sum_scores_exp += math.pow(math.e,lamb*square_score)

        # score_matrix_normalized = copy.deepcopy(score_matrix)
        # for r in range(len(score_matrix_normalized)):
        #     for c in range(len(score_matrix_normalized[r])):
        #         score_matrix_normalized[r][c] = score_matrix_normalized[r][c]/sum_scores

        print 'score matrix:'
        print score_matrix
        # print 'score matrix normalized'
        # print score_matrix_normalized

        # heatmaps
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]=='X'):
                    score_matrix[r][j] = -0.00001
                elif (score_matrix[r][j]=='O'):
                    score_matrix[r][j] = -0.00002

        if normalized:
            for r in range(len(score_matrix)):
                for c in range(len(score_matrix[r])):
                    if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                        if lamb is None:
                            score_matrix[r][c] = score_matrix[r][c]/sum_scores
                        else:
                            score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp
                    # if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                    #     score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp

        a = np.array(score_matrix)
        a = np.flip(a,0)
        print a
        heatmap = plt.pcolor(a)

        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                if(a[y,x]==-1) | (a[y,x]==-0.00001):
                    plt.text(x + 0.5, y + 0.5, 'X',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif((a[y,x]==-2) | (a[y,x]==-0.00002)):
                    plt.text(x + 0.5, y + 0.5, 'O',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif(a[y,x]!=0):
                    plt.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='white'
                     )

        fig = plt.colorbar(heatmap)
        fig_file_name = LOGFILE[g]
        fig_file_name = fig_file_name[:-4]
        data_matrices[fig_file_name[5:-6]] = score_matrix
        if normalized:
            fig_file_name = fig_file_name + '_normalized_composite_scores_' + 'exp=' + str(exp) + '_neighborhood=' + str(neighborhood_size) +'_lamb=' + str(lamb) + '.png'
        else:
            fig_file_name = fig_file_name + '_composite_scores_' + 'exp=' + str(exp) + '_neighborhood=' + str(neighborhood_size) +'_lamb=' + str(lamb) + '.png'

        plt.savefig(fig_file_name)
        plt.clf()


    return data_matrices


def run_models():
    data_composite_guassian = compute_scores_composite(True, exp=2, neighborhood_size=1)
    data_composite_reg = compute_scores_composite(True, exp=2, neighborhood_size=1, density='reg')
    data_density = compute_scores_density(True,neighborhood_size=1)
    data_density_guassian = compute_scores_density_guassian(True)
    data_paths = compute_scores_open_paths(True, exp=2)
    data_first_moves = rp.entropy_paths()

    print data_density.keys()
    for board in ['6_easy','6_hard','10_easy','10_hard','10_medium']:
        fig_file_name = 'heatmaps/guassian/' + board+ '_neighborhood=1.png'
        heatmaps = []
        full = board + '_full'
        pruned = board + '_pruned'
        if board.startswith('6'):
            fig, axes = plt.subplots(2, 3, figsize=(12,8))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(18,12))
        fig.suptitle(board)


        i = 0

        # heatmaps.append((data_density[full], 'density full'))
        # heatmaps.append((data_density_guassian[full], 'density guassian full'))
        # heatmaps.append((data_paths[full], 'paths full'))
        heatmaps.append((data_composite_reg[full], 'composite  full'))
        heatmaps.append((data_composite_guassian[full], 'composite guassian full'))
        heatmaps.append((data_first_moves[full], 'first moves full'))

        # heatmaps.append((data_density[pruned],'density pruned'))
        # heatmaps.append((data_density_guassian[pruned], 'density guassian pruned'))
        # heatmaps.append((data_paths[pruned],'paths pruned'))
        heatmaps.append((data_composite_reg[pruned], 'composite  pruned'))
        heatmaps.append((data_composite_guassian[pruned], 'composite guassian pruned'))
        heatmaps.append((data_first_moves[pruned], 'first moves pruned'))

        for ax in axes.flatten():  # flatten in case you have a second row at some point
            a = np.array(heatmaps[i][0])
            a = np.flip(a,0)
            img = ax.pcolormesh(a)
            for y in range(a.shape[0]):
                for x in range(a.shape[1]):
                    if(a[y,x]==-1) | (a[y,x]==-0.00001):
                        ax.text(x + 0.5, y + 0.5, 'X',
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='white'
                                 )
                    elif((a[y,x]==-2) | (a[y,x]==-0.00002)):
                        ax.text(x + 0.5, y + 0.5, 'O',
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='white'
                        )
                    elif(a[y,x]!=0):
                        ax.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 color='white'
                         )

            fig.colorbar(img, ax=ax)
            # plt.colorbar(img)
            ax.set_aspect('equal')
            ax.set_title(heatmaps[i][1])
            i+=1

        # a = np.random.rand(10,4)
        # img = axes[0,0].imshow(a,interpolation='nearest')
        # axes[0,0].set_aspect('auto')
        # plt.colorbar(img)
        # plt.title(board)
        # fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        plt.savefig(fig_file_name)
        plt.clf()


def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2 -1
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


if __name__ == "__main__":
    run_models()
    # gaus = makeGaussian(6,center=[0, 0])
    # print gaus
    # heatmap = plt.pcolor(gaus)
    # plt.colorbar(heatmap)
    # # plt.plot(gaus)
    # plt.show()

    # # plt.colorbar(img)
    # data_composite = compute_scores_composite(True, exp=2, neighborhood_size=1)
    # data_density = compute_scores_density(True,neighborhood_size=1)
    # data_paths = compute_scores_open_paths(True, exp=2)
    # data_first_moves = rp.entropy_paths()
    #
    # print data_density.keys()
    # for board in ['6_easy','6_hard','10_easy','10_hard','10_medium']:
    #     fig_file_name = 'heatmaps/noDensity/' + board+ '_neighborhood=1.png'
    #     heatmaps = []
    #     full = board + '_full'
    #     pruned = board + '_pruned'
    #     if board.startswith('6'):
    #         fig, axes = plt.subplots(2, 3, figsize=(12,8))
    #     else:
    #         fig, axes = plt.subplots(2, 3, figsize=(18,12))
    #     fig.suptitle(board)
    #
    #
    #     i = 0
    #
    #     # heatmaps.append((data_density[full], 'density full'))
    #     heatmaps.append((data_paths[full], 'paths full'))
    #     heatmaps.append((data_composite[full], 'composite full'))
    #     heatmaps.append((data_first_moves[full], 'first moves full'))
    #
    #     # heatmaps.append((data_density[pruned],'density pruned'))
    #     heatmaps.append((data_paths[pruned],'paths pruned'))
    #     heatmaps.append((data_composite[pruned], 'composite pruned'))
    #     heatmaps.append((data_first_moves[pruned], 'first moves pruned'))
    #
    #     for ax in axes.flatten():  # flatten in case you have a second row at some point
    #         a = np.array(heatmaps[i][0])
    #         a = np.flip(a,0)
    #         img = ax.pcolormesh(a)
    #         for y in range(a.shape[0]):
    #             for x in range(a.shape[1]):
    #                 if(a[y,x]==-1) | (a[y,x]==-0.00001):
    #                     ax.text(x + 0.5, y + 0.5, 'X',
    #                          horizontalalignment='center',
    #                          verticalalignment='center',
    #                          color='white'
    #                              )
    #                 elif((a[y,x]==-2) | (a[y,x]==-0.00002)):
    #                     ax.text(x + 0.5, y + 0.5, 'O',
    #                          horizontalalignment='center',
    #                          verticalalignment='center',
    #                          color='white'
    #                     )
    #                 elif(a[y,x]!=0):
    #                     ax.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
    #                              horizontalalignment='center',
    #                              verticalalignment='center',
    #                              color='white'
    #                      )
    #
    #         fig.colorbar(img, ax=ax)
    #         # plt.colorbar(img)
    #         ax.set_aspect('equal')
    #         ax.set_title(heatmaps[i][1])
    #         i+=1
    #
    #     # a = np.random.rand(10,4)
    #     # img = axes[0,0].imshow(a,interpolation='nearest')
    #     # axes[0,0].set_aspect('auto')
    #     # plt.colorbar(img)
    #     # plt.title(board)
    #     # fig.tight_layout()
    #     # fig.subplots_adjust(top=0.88)
    #     plt.savefig(fig_file_name)
    #     plt.clf()
    #     # plt.show()

    # compute_scores_density(True,neighborhood_size=2)
    # compute_scores_density(True)
    # compute_scores_composite(True, exp=2, neighborhood_size=1)
    # compute_scores_open_paths(True, 2)
    # compute_scores_open_paths(True)