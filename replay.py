import csv
import re
import copy
from user_game import *
from scipy import stats
import numpy
import matplotlib

LOGFILE = ['logs/6_hard_full_nov29.csv','logs/6_hard_pruned_nov29.csv','logs/10_hard_full_nov29.csv','logs/10_hard_pruned_nov29.csv', 'logs/6_easy_full_nov29.csv','logs/6_easy_pruned_nov29.csv','logs/10_easy_full_nov29.csv','logs/10_easy_pruned_nov29.csv','logs/10_medium_full_nov29.csv','logs/10_medium_pruned_nov29.csv']
USERID = '11e212ff'
DIMENSION = 6
START_POSITION = [[[0,2,0,0,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[1,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
                  [[0,2,0,1,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[2,1,0,2,0,0],[0,1,1,0,0,0],[0,2,0,0,2,0]],
                  [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,0,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[0,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,1,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
                  [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,1,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,1,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[2,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
                  [[0,1,0,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,1],[2,0,1,1,2,0],[1,0,2,2,0,0],[0,0,0,0,0,0]],
                  [[0,1,2,2,0,0],[0,2,1,1,0,1],[1,2,2,2,1,0],[2,0,1,1,2,1],[1,0,2,2,0,0],[0,0,0,0,0,0]],
                  [[0,0,0,0,1,0,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,1],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
                  [[0,0,0,0,1,2,2,0,0,0],[0,0,0,0,2,1,1,1,0,1],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,1],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
                  [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,1,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]],
                  [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,1,1,1,2,0,0,0,0],[0,0,1,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]]
                  ]

IGNORE_LIST = [[[0,3],[3,0]],
                  None,
                  [[2,4],[4,4]],
                None,
               [[0,2],[1,5]],
                None,
                [[0,5],[1,9]],
               None,
            [[3,2],[4,2]],
               None]

def replay():
    with open(LOGFILE, 'rb') as csvfile:
        log_reader = csv.DictReader(csvfile)
        for row in log_reader:
            if row['userid'] == USERID:
                if row['key'] in ('click','undo','reset'):
                    draw_board(row)


def heat_map_game():
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
        # move_matrix = []
        # for row in range(DIMENSION):
        #     row_positions = []
        #     for col in range(DIMENSION):
        #         row_positions.append(0)
        #     move_matrix.append(copy.deepcopy(row_positions))
        to_ignore = IGNORE_LIST[g];
        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            for row in log_reader:
                # if row['userid'] == USERID:
                if row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                        if (to_ignore!=None):
                            if ((rowPos!=to_ignore[0][0] | colPos!=to_ignore[0][1]) & (rowPos!=to_ignore[1][0] | colPos!=to_ignore[1][1])):
                                move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                        else:
                            move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                    #
                    # # print move_matrix
                    #     move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
        print LOGFILE[g]
        for row in move_matrix:
            print row

def entropy_board(positionIgnore = None):
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        move_counter = 0.0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
        # move_matrix = []
        # for row in range(DIMENSION):
        #     row_positions = []
        #     for col in range(DIMENSION):
        #         row_positions.append(0)
        #     move_matrix.append(copy.deepcopy(row_positions))
        to_ignore = IGNORE_LIST[g];
        to_ignore = None
        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            for row in log_reader:
                # if row['userid'] == USERID:
                if row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                        if (to_ignore!=None):
                            if ((rowPos!=to_ignore[0][0] | colPos!=to_ignore[0][1]) & (rowPos!=to_ignore[1][0] | colPos!=to_ignore[1][1])):
                                move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                                move_counter+=1.0
                            # else:
                            #     print 'ignore'
                        else:
                            move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                            move_counter+=1.0
        print LOGFILE[g]
        for row in move_matrix:
            print row

        #compute entropy
        pk = []
        cell_counter = 0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!='X') & (move_matrix[i][j]!='O')):
                    pk.append(move_matrix[i][j]/move_counter)
        print pk
        print stats.entropy(pk)


def run_analysis():
    with open(LOGFILE, 'rb') as csvfile:
        log_reader = csv.DictReader(csvfile)
        all_games = []
        curr_game = []
        curr_user = ''
        for row in log_reader:
            userid = row['userid']
            if userid == curr_user:
                curr_game.append(row)
            else:
                all_games.append(gameInstance(copy.deepcopy(curr_game)))
                curr_game = []
                curr_game.append(row)
                curr_user = userid


    print all_games


    filtered_games = []
    for game in all_games:
        if (len(game.actions) > 5) & (game.solution != "") & (game.time > 10000):
            filtered_games.append(game)


    print filtered_games

    get_solutions(filtered_games)
    construct_heat_map(filtered_games)


def get_solutions(games):
    solutions = {}
    for game in games:
        if game.solution not in solutions.keys():
            solutions[game.solution] = 1
        else:
            solutions[game.solution] +=1
    print solutions

def draw_board(move):
    print move['key']
    board = move['value']
    transform_board(board)
    board_to_matrix(board)
    # print board


def transform_board(board):
    char = board[0]
    board2 = board[1:len(board)-1]
    b = board2.split(']')
    # b = board2.replace(char,"")
    # c = b.replace(']','\n')
    for row in b:
        # print row
        row_new = row[1:]
        row_new_final = row_new.replace('[',"")
        print '-----'
        print row_new_final
        print '-----'
        row_ttt = row_new_final.replace('0','_')
        row_ttt = row_ttt.replace('1','X')
        row_ttt = row_ttt.replace('2', 'O')
        row_ttt = row_ttt.replace(',', ' ')
        print row_ttt
        # marks = row_new_final.split(',')
        # for mark in marks:
        #     if mark == '0':
        #         print '_'
        #     elif mark == '1':
        #         print 'X'
        #     elif mark == '2':
        #         print 'O'
    # test = "alpha.Customer[cus_Y4o9qMEZAugtnW] ..."
    # a = re.search(r"\[([A-Za-z0-9_]+)\]",board2)
    # p = re.compile(r"\[(. *?)\]")
    #
    # # a = p.search(board2)
    # print a.group()

def board_to_matrix(board):
    positions = []
    board2 = board[1:len(board)-1]
    b = board2.split(']')
    for row in b:
        position_row = []
        # print row
        row_new = row[1:]
        row_new_final = row_new.replace('[',"")
        if (len(row))>2:
            marks = row_new_final.split(',')
            for mark in marks:
                position_row.append(int(mark))
            positions.append(copy.deepcopy(position_row))
    print positions


def board_to_matrix_list(board):
    positions = []
    board2 = board[1:len(board)-1]
    b = board2.split(']')
    for row in b:
        position_row = []
        # print row
        row_new = row[1:]
        row_new_final = row_new.replace('[',"")
        if (len(row))>2:
            marks = row_new_final.split(',')
            for mark in marks:
                position_row.append(mark)
            positions.append(copy.deepcopy(position_row))
    return positions


def construct_heat_map(games, move = 1):
    move_matrix = []
    for row in range(DIMENSION):
        row_positions = []
        for col in range(DIMENSION):
            row_positions.append(0)
        move_matrix.append(copy.deepcopy(row_positions))


    for game in games:
        move = game.get_action_by_index(1)
        if move is not None:
            move_matrix[move[0]][move[1]] += 1

    print move_matrix

    initial_position = game.board_positions[0]

    for r in range(DIMENSION):
        for c in range(DIMENSION):
            if initial_position[r][c] == 2:
                initial_position[r][c] = 'O'
            elif initial_position[r][c] == 1:
                initial_position[r][c] = 'X'
            else:
                if initial_position[r][c] == 0:
                    initial_position[r][c] = ' '


    for row in initial_position:
        print row

    for row in move_matrix:
        print row






if __name__ == "__main__":
    entropy_board()
    # heat_map_game()
    # run_analysis()
    # replay()
