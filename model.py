import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import replay as rp
import computational_model as cm
import json
from emd import emd
# from pyemd import emd
from scipy import stats
# from cv2 import *


# these are the files with user data foeach of the board
LOGFILE = ['logs/6_hard_full_dec19.csv','logs/6_hard_pruned_dec19.csv','logs/10_hard_full_dec19.csv','logs/10_hard_pruned_dec19.csv', 'logs/6_easy_full_dec19.csv','logs/6_easy_pruned_dec19.csv','logs/10_easy_full_dec19.csv','logs/10_easy_pruned_dec19.csv', 'logs/10_medium_full_dec19.csv','logs/10_medium_pruned_dec19.csv']
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

'''
computes the density score for a cell
@neighborhood_size: how many cells around the square to look at
'''
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

'''
computes the density score for a cell using guassians
@guassian_kernel: this is created separately and sent to the function
'''
def compute_density_guassian(row, col, board, guassian_kernel):
    density_score = 0.0
    for guas in guassian_kernel:
        density_score += guas[row][col]

    return density_score

'''
computes the density score for all cells in all boards using guassian approach
@normalized: whether to normalize the scores such that they sum up to 1
@sig: the standard deviation to use in guassian
@lamb: this is what I used before for the quantal response, you can ignore it
'''
def compute_scores_density_guassian(normalized=False, sig = 3, lamb = 1):
    data_matrices = {}  # this will be a dictionary that holds the matrics for all the boards, each will be indexed by the board name
    for g in range(len(LOGFILE)):  # iterate over all boards
        board_matrix = copy.deepcopy(START_POSITION[g])  # gets the board starting position
        for i in range(len(board_matrix)):  # replaces 1s with Xs and 2s with Os
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        # print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)  # this will hold the matrix of scores

        # create guassians for each X square
        guassian_kernel = []
        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 'X':
                    guassian_kernel.append(makeGaussian(len(board_matrix),fwhm=sig,center=[r,c]))

        # compute density scores
        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if cell is free
                    square_score = compute_density_guassian(r,c,board_matrix,guassian_kernel)
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    sum_scores_exp += math.pow(math.e,lamb*square_score)

        # put a negative small value instead of X and O so it doesn't affect heatmaps
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]=='X'):
                    score_matrix[r][j] = -0.00001
                elif (score_matrix[r][j]=='O'):
                    score_matrix[r][j] = -0.00002

        if normalized:  # normalize cell scores to 1
            for r in range(len(score_matrix)):
                for c in range(len(score_matrix[r])):
                    # score_matrix[r][c] = score_matrix[r][c]/sum_scores
                    if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                        score_matrix[r][c] = score_matrix[r][c]/sum_scores
                        # score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp

        board_name = LOGFILE[g]
        board_name = board_name[:-4]
        data_matrices[board_name[5:-6]] = score_matrix  # store matrix for this board in the dictionary

    return data_matrices

'''
computes the density score for all cells based on neighbors
@normalized: whether to normalize the scores such that they sum up to 1
@neighborhood_size: how far to look for neighbors
@lamb: this is what I used before for the quantal response, you can ignore it
'''
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

        # print board_matrix

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

        board_name = LOGFILE[g]
        board_name = board_name[:-4]
        data_matrices[board_name[5:-6]] = score_matrix
    matrix_name = 'data_matrices/model_density' + '_nbr=' +str(neighborhood_size)
    write_matrices_to_file(data_matrices, matrix_name + '.json')
    return data_matrices

'''
checks if the two paths can be blocked by a shared cell
'''
def check_path_overlap(empty1, empty2):
    for square in empty1:
        if square in empty2:
            return True
    return False


'''
computes the potential winning paths the cell is part of, of if you send it player = 'O' it will check how many
paths X destroys if it places an X on this cell
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
'''
def compute_open_paths_data(row, col, board, exp=1, player = 'X'):
    other_player = 'O'
    if player == 'O':
        other_player = 'X'

    threshold = 0
    if len(board)==10:
        threshold +=1

    streak_size = 4  # how many Xs in a row you need to win
    if len(board) == 10:  # if it's a 10X10 board you need 5.
        streak_size = 5

    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    paths = []
    # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
    for i in range(streak_size):
        r = row - i
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))

    # check left-down diagonal
    for i in range(streak_size):
        r = row - i
        c = col + i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col -= 1
            path_length += 1

        if (path_length == streak_size) & (not blocked)  & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))

    # check vertical
    for i in range(streak_size):
        r = row - i
        c = col
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])

            path.append([square_row,square_col])
            square_row += 1
            path_length += 1



        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))

    # check horizontal
    for i in range(streak_size):
        r = row
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row, square_col])

            path.append([square_row, square_col])
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))


    score = 0.0
    # compute the score for the cell based on the potential paths
    for i in range(len(open_paths_data)):
        p1 = open_paths_data[i]
        score += 1.0/math.pow((streak_size-p1[0]), exp)  # score for individual path
        for j in range(i+1, len(open_paths_data)):
            p2 = open_paths_data[j]
            if not(check_path_overlap(p1[1],p2[1])):  # interaction score if the paths don't overlap
                score += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(p1[0]*p2[0]), exp))

    return (score, open_paths_data)


def compute_relative_path_score(row, col, path_data, score_matrix, lamb = 1):
    raw_score = score_matrix[row][col]
    relative_score = 0.0

    for p in path_data:
        path_score = 0.0
        path_score_exp = 0.0
        for square in p:
            if (score_matrix[square[0]][square[1]]!='X') & (score_matrix[square[0]][square[1]]!='O'):
                path_score += score_matrix[square[0]][square[1]]
                path_score_exp += math.exp(score_matrix[square[0]][square[1]]*lamb)
        # if raw_score > 0:
        relative_score += (math.exp(raw_score)/path_score_exp)*path_score
    return relative_score






'''
computes the potential winning paths the cell is part of, of if you send it player = 'O' it will check how many
paths X destroys if it places an X on this cell
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
'''
def compute_open_paths(row, col, board, exp=1, player = 'X'):
    other_player = 'O'
    if player == 'O':
        other_player = 'X'

    threshold = 0
    if len(board)==10:
        threshold +=1

    streak_size = 4  # how many Xs in a row you need to win
    if len(board) == 10:  # if it's a 10X10 board you need 5.
        streak_size = 5

    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
    for i in range(streak_size):
        r = row - i
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])

            square_row += 1
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares))
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares))

    # check left-down diagonal
    for i in range(streak_size):
        r = row - i
        c = col + i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])

            square_row += 1
            square_col -= 1
            path_length += 1

        if (path_length == streak_size) & (not blocked)  & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares))
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares))

    # check vertical
    for i in range(streak_size):
        r = row - i
        c = col
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])

            square_row += 1
            path_length += 1



        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares))
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares))

    # check horizontal
    for i in range(streak_size):
        r = row
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row, square_col])

            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares))
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares))

    # print open_paths_lengths


    # if ((row==0) & (col==3)) |((row==3) & (col==0)):
    #     print 'here'
    score = 0.0
    # compute the score for the cell based on the potential paths
    for i in range(len(open_paths_data)):
        p1 = open_paths_data[i]
        score += 1.0/math.pow((streak_size-p1[0]), exp)  # score for individual path
        for j in range(i+1, len(open_paths_data)):
            p2 = open_paths_data[j]
            if not(check_path_overlap(p1[1],p2[1])):  # interaction score if the paths don't overlap
                score += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(p1[0]*p2[0]), exp))


    return score

'''
computes the score for each cell in each of the boards based on the paths score
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
'''
def compute_scores_open_paths(normalized=False, exp=1, lamb = 1):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        print LOGFILE[g]
        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        # print board_matrix

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

        board_name = LOGFILE[g]
        board_name = board_name[:-4]
        data_matrices[board_name[5:-6]] = score_matrix


    return data_matrices

'''
computes the score for each cell in each of the boards based on the paths score, also considers opponent paths
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
@o_weight says how much weight to give to blocking the opponent paths when scoring the cell
'''
def compute_scores_open_paths_opponent(normalized=False, exp=1, lamb = 1, o_weight = 0.5):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        print LOGFILE[g]
        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        # print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)


        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    # x_potential = compute_open_paths(r, c, board_matrix,exp=exp)  # check open paths for win
                    x_potential = compute_open_paths_data(r, c, board_matrix,exp=exp)  # check open paths for win
                    o_potential = compute_open_paths(r, c, board_matrix,exp=exp, player='O')  # check opponent paths that are blocked
                    square_score = (1-o_weight)*x_potential + o_weight*o_potential  # combine winning paths with blocked paths
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    sum_scores_exp += math.pow(math.e,lamb*square_score)

        absolute_score_matrix = copy.deepcopy(score_matrix)
        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    score_matrix[r][c] = compute_relative_path_score(r,c,x_potential,absolute_score_matrix)


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


        board_name = LOGFILE[g]
        board_name = board_name[:-4]
        data_matrices[board_name[5:-6]] = score_matrix

    return data_matrices

'''
computes the score for each cell in each of the boards based on a combination of density and path scores (not layers, multiplication)
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
@density says which density function to use
@neighborhood size is for density score computation when using neighbors
@sig is for guassians when using guassian density
'''
def compute_scores_composite(normalized=False, exp=1, neighborhood_size=1, density = 'guassian', lamb=None, sig=3):
    data_matrices = {}
    # compute scores based on density  for all boards (so we can prune squares)
    if (density=='guassian'):
        density_scores = compute_scores_density_guassian(True,sig=sig)
    else:
        density_scores = compute_scores_density(True, neighborhood_size=neighborhood_size)
    path_scores = compute_scores_open_paths(True,exp)  # compute path-based scores for all boards

    for g in range(len(LOGFILE)):
        board_key = LOGFILE[g]
        board_key = board_key[:-4]
        board_key = board_key[5:-6]

        density_scores_board = density_scores[board_key]  # get density scores for this board
        path_scores_board = path_scores[board_key] # get path-based scores for this board

        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        # print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)


        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    square_score = density_scores_board[r][c] * path_scores_board[r][c]  # combine density and path scores
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    if lamb != None:
                        sum_scores_exp += math.pow(math.e,lamb*square_score)

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


        board_name = LOGFILE[g]
        board_name = board_name[:-4]
        data_matrices[board_name[5:-6]] = score_matrix

    return data_matrices

'''
same as the function above but also considers O paths
'''
def compute_scores_composite_opponent(normalized=False, exp=1, neighborhood_size=1, density = 'guassian', lamb=None, sig=3, opponent = False, o_weight=0.5):
    data_matrices = {}
    if (density=='guassian'):
        density_scores = compute_scores_density_guassian(True,sig=sig)
    else:
        density_scores = compute_scores_density(True, neighborhood_size=neighborhood_size)
    if opponent:
        path_scores = compute_scores_open_paths_opponent(True,exp, o_weight=o_weight)
    else:
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

        # print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)


        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    square_score = density_scores_board[r][c] * path_scores_board[r][c]
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    if lamb != None:
                        sum_scores_exp += math.pow(math.e,lamb*square_score)

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

        board_name = LOGFILE[g]
        board_name = board_name[:-4]
        data_matrices[board_name[5:-6]] = score_matrix

    return data_matrices


'''
computes the score for each cell in each of the boards based in the layers approach (first filter cells by density)
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
@density says which density function to use
@neighborhood size is for density score computation when using neighbors
@sig is for guassians when using guassian density
@threshold says how much to prune (0.2 means we will remove cells that have score<0.2*maxScore)
@o_weight says how much weight to give for blocking O paths
@integrate says whether to combine density and path scores (done if = True), or just use path score after the initial filtering (done if = False)
'''
def compute_scores_layers(normalized=False, exp=1, neighborhood_size=1, density = 'reg', lamb=None, sig=3,
                          threshold=0.2, o_weight=0.0, integrate = False):
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

        # print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        density_score_matrix = copy.deepcopy(board_matrix)

        if density=='guassian':
            # create guassians for each X square
            guassian_kernel = []
            for r in range(len(board_matrix)):
                for c in range(len(board_matrix[r])):
                    if board_matrix[r][c] == 'X':
                        guassian_kernel.append(makeGaussian(len(board_matrix),fwhm=sig,center=[r,c]))

        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    if density == 'guassian':
                        square_score = compute_density_guassian(r, c, board_matrix, guassian_kernel)  # check neighborhood
                    else:
                        square_score = compute_density(r, c, board_matrix, neighborhood_size)  # check neighborhood
                    density_score_matrix[r][c] = square_score
                    sum_scores += square_score
                    # if lamb!=None:
                    #     sum_scores_exp += math.pow(math.e,lamb*square_score)

        # compute maximal density score for filtering
        max_density_score = -1000000
        for r in range(len(density_score_matrix)):
            for c in range(len(density_score_matrix[r])):
                # score_matrix[r][c] = score_matrix[r][c]/sum_scores
                if (density_score_matrix[r][c]!='X') & (density_score_matrix[r][c]!='O'):
                    density_score_matrix[r][c] = density_score_matrix[r][c]/sum_scores
                    if density_score_matrix[r][c] > max_density_score:
                        max_density_score = density_score_matrix[r][c]
                    # score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp

        # run path score on remaining squares
        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)

        paths_data = copy.deepcopy(board_matrix)

        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if (board_matrix[r][c] == 0) & (density_score_matrix[r][c]>threshold*max_density_score):  # only check if free & passed threshold
                    x_paths = compute_open_paths_data(r, c, board_matrix,exp=exp)  # check open paths for win
                    square_score_x = x_paths[0]
                    x_paths_data = []
                    for path in x_paths[1]:
                        x_paths_data.append(path[2])
                    paths_data[r][c] = copy.deepcopy(x_paths_data)
                    square_score_o = compute_open_paths(r, c, board_matrix, exp=exp, player = 'O')
                    square_score = (1-o_weight)*square_score_x + o_weight*square_score_o
                    if integrate:
                        square_score = square_score*density_score_matrix[r][c]
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    if lamb!=None:
                        score_matrix[r][c] = math.pow(math.e,lamb*square_score)
                        sum_scores_exp += math.pow(math.e,lamb*square_score)

        # x_paths_data = []
        # for path in x_paths[1]:
        #     x_paths_data.append(path[2])
        # sum_scores = 0.0
        # absolute_score_matrix = copy.deepcopy(score_matrix)
        # for r in range(len(board_matrix)):
        #     for c in range(len(board_matrix[r])):
        #         if (board_matrix[r][c] == 0) & (density_score_matrix[r][c]>threshold*max_density_score):  # only check if free
        #             score_matrix[r][c] = compute_relative_path_score(r,c,paths_data[r][c],absolute_score_matrix)
        #             sum_scores+=score_matrix[r][c]

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
                    # if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                    if (score_matrix[r][c]>0):
                        if lamb is None:
                            score_matrix[r][c] = score_matrix[r][c]/sum_scores
                        else:
                            score_matrix[r][c] = score_matrix[r][c]/sum_scores_exp

        board_name = LOGFILE[g]
        board_name = board_name[:-4]


        data_matrices[board_name[5:-6]] = score_matrix
    matrix_name = 'data_matrices/model_layers' + '_e=' + str(exp) + '_nbr=' +str(neighborhood_size) + '_o=' +str(o_weight)
    if integrate:
        matrix_name = matrix_name + '_integrated'
    if density == 'guassian':
        matrix_name = matrix_name + 'guassian'
    # matrix_name = matrix_name+ 't=1'
    write_matrices_to_file(data_matrices, matrix_name + '.json')
    return data_matrices

'''
auxilary function, you can ignore
'''
def transform_matrix_to_list(mat):
    list_rep = []
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j]<0.00001:
                mat[i][j] = 0.00001
            list_rep.append(mat[i][j])

    sum_values = sum(list_rep)
    for i in range(len(list_rep)):
        list_rep[i]=list_rep[i]/sum_values
    return list_rep

'''
auxilary function, you can ignore (unless you want to play with the guassians)
'''
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

    # return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return np.exp(-1 * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def write_matrices_to_file(data_matrices, filename):
  with open(filename, 'w') as fp:
      json.dump(data_matrices, fp)


def read_matrices_from_file(filename):
  json1_file = open(filename)
  json1_str = json1_file.read()
  json1_data = json.loads(json1_str)
  return json1_data


'''
use this method to define which models to run, it will create the heatmaps and compute distances
'''
def run_models():
    # generate the models you want to include in the heatmaps
    # For example, say that I want to show the layers model (just with path scores, with and without opponent,
    # and compare it to the first moves made by participants) --
    # I create model without the opponent using the layers model
    data_layers_reg = compute_scores_layers(normalized=False,exp=3,neighborhood_size=2,density='reg',o_weight=0.0, integrate=False)
    # # and the model with the opponent
    # data_layers_reg_withO = compute_scores_layers(normalized=True,exp=3,neighborhood_size=2,density='reg',o_weight=0.5, integrate=False)
    # and then the actual distribution of moves (it's computed from another file but you don't need to edit it)
    data_test = read_matrices_from_file('data_matrices/computational_model.json')
    data_computational_model = cm.get_heatmaps_alpha_beta()

    # write_matrices_to_file(data_computational_model, 'data_matrices/computational_model.json')
    # return
    # data_first_moves = rp.entropy_paths()
    data_clicks = rp.entropy_board_average()

    # go over all boards
    for board in ['6_easy','6_hard','10_easy','10_hard','10_medium']:
        # this tells it where to save the heatmaps and what to call them:
        fig_file_name = 'heatmaps/compVsPeople/' + board + '_peopleAvgVsAlphaBeta.png'
        heatmaps = []  # this object will store all the heatmaps and later save to a file
        full = board + '_full'
        pruned = board + '_pruned'
        if board.startswith('6'):  # adjust sizes of heatmaps depending on size of boards
            fig, axes = plt.subplots(2, 2, figsize=(12,8))  # this will create a 2X3 figure with 6 heatmaps, you can modify if you want fewer/more
            # fig, axes = plt.subplots(2, 4, figsize=(10,6))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(18,12)) # this will create a 2X3 figure with 6 heatmaps, you can modify if you want fewer/more
            # fig, axes = plt.subplots(2, 4, figsize=(18,12))

        fig.suptitle(board)  # add subtitle to the figure based on board name

        i = 0  # this will be used to index into the list of heatmaps
        print board  # just printing the board name so I know if they finish, you can remove


        # here you will append to the heatmap list any heatmap you want to present. For each heatmap you append a pair -
        # the first element is the score matrix, the second element is a title for the heatmap. Note that ordering of adding to the heatmap lists
        #determines where each heatmap is shown (goes from top left to bottom right)


        heatmaps.append((data_computational_model[full][0], 'alpha-beta' +'\n' +str(round(data_computational_model[full][1],3))))
        heatmaps.append((data_clicks[full][0], 'people'+'\n' +str(round(data_clicks[full][1],3))))
        heatmaps.append((data_computational_model[pruned][0], 'alpha-beta' +'\n' +str(round(data_computational_model[pruned][1],3))))
        heatmaps.append((data_clicks[pruned][0], 'people'+'\n' +str(round(data_clicks[pruned][1],3))))
        # dist = emd(data_layers_reg[full],data_first_moves[full]) # earth mover distance for the full board
        # # print dist
        # heatmaps.append((data_layers_reg[full], 'layers' + '\n' +str(round(dist,3)))) # add the model to the heatmap list with name and distance
        #
        # dist = emd(data_layers_reg_withO[full],data_first_moves[full]) # earth mover distance for the full board
        # heatmaps.append((data_layers_reg_withO[full], 'layers with O '+'\n' +str(round(dist,3)))) # add the model to the heatmap list with name and distance
        # # add the empirical distribution heatmap
        # heatmaps.append((data_first_moves[full], 'first moves'))
        #
        # # and then the same for the pruned boards
        # dist = emd(data_layers_reg[pruned],data_first_moves[pruned]) # earth mover distance for the full board
        # heatmaps.append((data_layers_reg[pruned], 'layers' + '\n' +str(round(dist,3)))) # add the model to the heatmap list with name and distance
        #
        # dist = emd(data_layers_reg_withO[pruned],data_first_moves[pruned]) # earth mover distance for the full board
        # heatmaps.append((data_layers_reg_withO[pruned], 'layers with O '+'\n' +str(round(dist,3)))) # add the model to the heatmap list with name and distance
        # # add the empirical distribution heatmap
        # heatmaps.append((data_first_moves[pruned], 'first moves'))

        # this creates the actual heatmaps
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


def run_models_from_list(models_file_list, base_heatmap_name, base_matrix_index = None):

    # generate the models you want to include in the heatmaps
    # For example, say that I want to show the layers model (just with path scores, with and without opponent,
    # and compare it to the first moves made by participants) --
    # I create model without the opponent using the layers model
    # data_layers_reg = compute_scores_layers(normalized=False,exp=3,neighborhood_size=2,density='reg',o_weight=0.0, integrate=False)
    # # and the model with the opponent
    # data_layers_reg_withO = compute_scores_layers(normalized=True,exp=3,neighborhood_size=2,density='reg',o_weight=0.5, integrate=False)
    # and then the actual distribution of moves (it's computed from another file but you don't need to edit it)
    data = []

    for file in models_file_list:
        matrices = read_matrices_from_file('data_matrices/'+file)
        data_matrices = {}
        for mat in matrices:
            # for k,v in mat:
            if mat.endswith('.json'):
                data_matrices[mat[:-5]] = matrices[mat]
            else:
                data_matrices[mat] = matrices[mat]

        data.append(copy.deepcopy(data_matrices))

    for board in ['6_easy','6_hard','10_easy','10_hard','10_medium']:
        plt.rcParams.update({'font.size': 9})
        fig_file_name = base_heatmap_name + '_' + board + '.png'
        heatmaps = []
        full = board + '_full'
        pruned = board + '_pruned'
        if board.startswith('6'):  # adjust sizes of heatmaps depending on size of boards
            fig, axes = plt.subplots(2, len(data), figsize=(16,8))  # this will create a 2X3 figure with 6 heatmaps, you can modify if you want fewer/more
            # fig, axes = plt.subplots(2, 4, figsize=(10,6))
        else:
            fig, axes = plt.subplots(2, len(data), figsize=(24,12)) # this will create a 2X3 figure with 6 heatmaps, you can modify if you want fewer/more
            # fig, axes = plt.subplots(2, 4, figsize=(18,12))

        fig.suptitle(board)  # add subtitle to the figure based on board name
        i = 0  # this will be used to index into the list of heatmaps
        print board  # just printing the board name so I know if they finish, you can remove

        for j in range(len(data)):
            matrix_name_full = models_file_list[j][:-5]
            matrix_name_pruned = models_file_list[j][:-5]
            if base_matrix_index != None:
                dist_full = emd(data[j][full],data[base_matrix_index][full]) # earth mover distance for the full board
                dist_pruned = emd(data[j][pruned],data[base_matrix_index][pruned]) # earth mover distance for the full board
                matrix_name_full = matrix_name_full + '\n' + str(round(dist_full, 3))
                matrix_name_pruned = matrix_name_pruned + '\n' + str(round(dist_pruned, 3))
            heatmaps.append((data[j][full], matrix_name_full))

        for j in range(len(data)):
            matrix_name_full = models_file_list[j][:-5]
            matrix_name_pruned = models_file_list[j][:-5]
            if base_matrix_index != None:
                dist_full = emd(data[j][full],data[base_matrix_index][full]) # earth mover distance for the full board
                dist_pruned = emd(data[j][pruned],data[base_matrix_index][pruned]) # earth mover distance for the full board
                matrix_name_full = matrix_name_full + '\n' + str(round(dist_full, 3))
                matrix_name_pruned = matrix_name_pruned + '\n' + str(round(dist_pruned, 3))
            heatmaps.append((data[j][pruned], matrix_name_pruned))



        # this creates the actual heatmaps
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
            # ax.tick_params(labelsize=8)
            ax.set_title(heatmaps[i][1])
            i += 1

        plt.savefig(fig_file_name)
        plt.clf()



if __name__ == "__main__":
    # compute_scores_density(normalized=True,neighborhood_size=2)
    # compute_scores_layers(normalized=True, exp=1, neighborhood_size=2, o_weight=0.5, integrate=False)
    # compute_scores_layers(normalized=True, exp=2, neighborhood_size=2, o_weight=0.0, integrate=True)
    # # compute_scores_layers(normalized=True, exp=1, neighborhood_size=2, o_weight=0.0, integrate=False)
    # compute_scores_layers(normalized=True, exp=2, neighborhood_size=2, o_weight=0.5, integrate=False)
    # model_files = ['model_density_nbr=2.json','model_layers_e=1_nbr=2_o=0.5.json','model_layers_e=2_nbr=2_o=0.5.json', 'avg_people_clicks_all.json']
    # model_files = ['avg_people_first_moves_all.json', 'avg_people_clicks_all.json']
    # model_files = ['model_layers_e=2_nbr=2_o=0.5.json','model_layers_e=2_nbr=2_o=0.0.json', 'avg_people_clicks_all.json']
    # # # # run_models()  # calls the function that runs the models
    # # # # model_files = ['paths_linear_square_opp.json', 'paths_non-linear_square_opp.json', 'avg_people_clicks_solvedCorrect.json']
    # model_files = ['model_layers_e=2_nbr=2_o=0.5.json','paths_non-linear_square_layers_opp.json', 'avg_people_clicks_all.json']
    model_files = ['model_layers_e=2_nbr=2_o=0.5.json','paths_non-linear_square_layers_opp.json', 'avg_people_clicks_all.json']
    run_models_from_list(model_files, 'heatmaps/cogsci/alphaBetaVsModelComparisonWithClicks',2)
    # model_files = ['density.json','paths_linear_square_layers_opp.json','paths_non-linear_square_layers_opp.json', 'avg_people_first_moves_all.json']
    model_files = ['model_layers_e=2_nbr=2_o=0.5.json','paths_non-linear_square_layers_opp.json', 'avg_people_first_moves_all.json']
    run_models_from_list(model_files, 'heatmaps/cogsci/alphaBetaVsModelComparisonWithFirstMoves',2)
