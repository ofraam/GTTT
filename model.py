import copy
import numpy as np
import matplotlib.pyplot as plt
import math

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
    neighborhood_effective_size = 0.0
    x_count = 0.0
    for i in range(-1*neighborhood_size,neighborhood_size+1):
        for j in range(-1*neighborhood_size,neighborhood_size+1):
            if (i != 0) | (j != 0):
                r = row + i
                c = col + j
                if (r < len(board)) & (r >= 0) & (c < len(board)) & (c >= 0):
                    # print r
                    # print c
                    neighborhood_effective_size+=1.0
                    if board[r][c] == 'X':
                        x_count += 1.0

    return x_count/8  # divide by 8 anyway


def compute_scores_density(normalized=False, neighborhood_size=1):
    lamb = 3
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
                if(a[y,x]==-1) | (a[y,x]==-0.00001/sum_scores):
                    plt.text(x + 0.5, y + 0.5, 'X',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif((a[y,x]==-2) | (a[y,x]==-0.00002/sum_scores)):
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
        if normalized:
            fig_file_name = fig_file_name + '_normalized_density_scores.png'
        else:
            fig_file_name = fig_file_name + '_density_scores.png'

        plt.savefig(fig_file_name)
        plt.clf()


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


def compute_scores_open_paths(normalized=False, exp=1):
    lamb = 3
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
        if normalized:
            fig_file_name = fig_file_name + '_normalized_path_scores' + 'exp=' + str(exp) + '.png'
        else:
            fig_file_name = fig_file_name + '_path_scores' + 'exp=' + str(exp) + '.png'

        plt.savefig(fig_file_name)
        plt.clf()


if __name__ == "__main__":
    # compute_scores_density()
    compute_scores_density(True)
    compute_scores_open_paths(True, 2)
    # compute_scores_open_paths(True)