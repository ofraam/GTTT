import csv
import re
import copy
from user_game import *
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import math
import json

LOGFILE = ['logs/6_hard_full_dec19.csv','logs/6_hard_pruned_dec19.csv','logs/10_hard_full_dec19.csv','logs/10_hard_pruned_dec19.csv', 'logs/6_easy_full_dec19.csv','logs/6_easy_pruned_dec19.csv','logs/10_easy_full_dec19.csv','logs/10_easy_pruned_dec19.csv','logs/10_medium_full_dec19.csv','logs/10_medium_pruned_dec19.csv']

# LOGFILE = ['logs/6_hard_verify_dec19.csv','logs/10_hard_verify_dec19.csv', 'logs/6_easy_verify_dec19.csv','logs/10_easy_verify_dec19.csv','logs/10_medium_verify_dec3.csv']

USERID = '11e212ff'
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

# START_POSITION = [[[0,2,0,1,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[0,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
#                  [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,1,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[0,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
#                   [[0,1,0,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,0],[2,0,1,1,2,1],[1,0,2,2,0,0],[0,0,0,0,0,0]],
#                   [[0,0,0,0,1,0,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,1],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
#                  [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,1,1,1,0,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]]
#                   ]

IGNORE_LIST = [[[0,3],[3,0]],
                  None,
                  [[2,4],[6,0]],
                None,
               [[0,2],[3,5]],
                None,
                [[0,5],[4,9]],
               None,
            [[3,2],[3,5]],
               None]

def replay():
    with open(LOGFILE, 'rb') as csvfile:
        log_reader = csv.DictReader(csvfile)
        for row in log_reader:
            if row['userid'] == USERID:
                if row['key'] in ('click','undo','reset'):
                    draw_board(row)


def seperate_log(log_file):
    with open(log_file, 'rb') as csvfile:
        log_reader = csv.DictReader(csvfile)
        curr_log = ''
        curr_log_records = []
        for row in log_reader:
            log = row['boardSize']+'_'+row['boardType']+'_'+row['condition']
            if log == curr_log:
                curr_log_records.append(row)
            elif len(curr_log_records)>0:
                dataFile = open('logs/'+curr_log+'_dec19.csv', 'wb')
                print curr_log_records[0]
                dataWriter = csv.DictWriter(dataFile, fieldnames=curr_log_records[0].keys(), delimiter=',')
                dataWriter.writeheader()
                for record in curr_log_records:
                    dataWriter.writerow(record)
                curr_log_records = []
                curr_log = log
            else:
                curr_log = log

def heat_map_game(normalized = False):
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
        # to_ignore = IGNORE_LIST[g];
        move_count = 0.0
        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            for row in log_reader:
                # if row['userid'] == USERID:
                if row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O') & (player==1)):
                        move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1.0
                        move_count+=1.0
                    # else:
                    #     print 'bad click'
                    #
                    # # print move_matrix
                    #     move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
        print LOGFILE[g]
        for row in move_matrix:
            print row

        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[i])):
                if (move_matrix[r][j]=='X'):
                    move_matrix[r][j] = -1
                elif (move_matrix[r][j]=='O'):
                    move_matrix[r][j] = -2
                #     print move_matrix[i][j]
                # else:
                #     print  move_matrix[i][j]

        if (normalized):
            for r in range(0,len(move_matrix)):
                for j in range(0,len(move_matrix[i])):
                    # if (move_matrix[r][j]>0):
                    move_matrix[r][j] = move_matrix[r][j]/move_count
                    # else:
        #

        print move_matrix
        a = np.array(move_matrix)
        a = np.flip(a,0)
        print a
        heatmap = plt.pcolor(a)

        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                if((a[y,x]==-1) | (a[y,x]==-1.0/move_count)):
                    plt.text(x + 0.5, y + 0.5, 'X',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white'
                    )
                elif((a[y,x]==-2) | (a[y,x]==-2.0/move_count)):
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
        fig_file_name=fig_file_name[:-4]
        fig_file_name = fig_file_name + '.png'
        plt.savefig(fig_file_name)
        plt.clf()

        # plt.imshow(a, cmap='hot', interpolation='nearest')

        # plt.show()





def heat_map_solution(normalized = False):
    data_matrices = {}
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
        # to_ignore = IGNORE_LIST[g];
        move_count = 0.0
        rows = ['1','2','3','4','5','6','7','8','9','10']
        cols = ['a','b','c','d','e','f','g','h','i','j']
        user_count = 0
        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            for row in log_reader:
                # if row['userid'] == USERID:
                if row['key']=='best_move':
                    # print row
                    move = str(row['value']).lower()
                    user_count+=1
                    # print move
                    if(len(move)==2):
                        if (move[0] in cols):
                            colPos = cols.index(move[0])
                            if(move[1] in rows):
                                rowPos = len(move_matrix)-rows.index(move[1])-1


                                if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                                    move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1.0
                                    move_count+=1.0
                        elif (move[0] in rows):
                            rowPos = rows.index(move[0])
                            if(move[1] in cols):
                                colPos = len(move_matrix)-cols.index(move[1])-1


                                if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                                    move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1.0
                                    move_count+=1.0
                    # else:
                    #     print 'bad click'
                    #
                    # # print move_matrix
                    #     move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
        print LOGFILE[g]
        print user_count
        for row in move_matrix:
            print row

        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[i])):
                if (move_matrix[r][j]=='X'):
                    move_matrix[r][j] = -0.00001
                elif (move_matrix[r][j]=='O'):
                    move_matrix[r][j] = -0.00002
                #     print move_matrix[i][j]
                # else:
                #     print  move_matrix[i][j]

        if (normalized):
            for r in range(0,len(move_matrix)):
                for j in range(0,len(move_matrix[r])):
                    if (move_matrix[r][j]>0):
                        move_matrix[r][j] = move_matrix[r][j]/move_count
                    # else:

        fig_file_name = LOGFILE[g]
        fig_file_name=fig_file_name[:-4]
        data_matrices[fig_file_name[5:-6]] = move_matrix

        print move_matrix

    write_matrices_to_file(data_matrices,'data_matrices/participant_solutions.json')
        # a = np.array(move_matrix)
        # a = np.flip(a,0)
        # print a
        # heatmap = plt.pcolor(a)
        #
        # for y in range(a.shape[0]):
        #     for x in range(a.shape[1]):
        #         if((a[y,x]==-1) | (a[y,x]==-1.0/move_count)):
        #             plt.text(x + 0.5, y + 0.5, 'X',
        #                  horizontalalignment='center',
        #                  verticalalignment='center',
        #                  color='white'
        #             )
        #         elif((a[y,x]==-2) | (a[y,x]==-2.0/move_count)):
        #             plt.text(x + 0.5, y + 0.5, 'O',
        #                  horizontalalignment='center',
        #                  verticalalignment='center',
        #                  color='white'
        #             )
        #         elif(a[y,x]!=0):
        #             plt.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
        #                      horizontalalignment='center',
        #                      verticalalignment='center',
        #                      color='white'
        #              )
        #
        # fig = plt.colorbar(heatmap)
        # fig_file_name = LOGFILE[g]
        # fig_file_name=fig_file_name[:-4]
        # fig_file_name = fig_file_name + 'solutionHeatmap.png'
        # plt.savefig(fig_file_name)
        # plt.clf()

        # plt.imshow(a, cmap='hot', interpolation='nearest')

        # plt.show()

def compare_paths(p1,p2):
    if len(p1)!=len(p2):
        return False
    for i in range(len(p1)):
        if p1[i]!=p2[i]:
            return False

    return True

def add_path_count(paths_counts, new_path):
    for path in paths_counts:
        if(compare_paths(path[0],new_path)):
            path[1] = path[1]+1.0
            return
    paths_counts.append([new_path,1.0])

def add_path_count_subpaths(paths_counts, new_path):
    for i in range(0,len(new_path)):
        add_path_count(paths_counts,new_path[0:i+1])


def user_stats(subpaths=False):
    user_data_headers = ['boardSize','boardType','condition','userid','curr_user_nodes','curr_user_num_paths','curr_user_sum_depth','curr_user_undo','curr_user_restart', 'confidence', 'correctness']
    users_data = []
    user_counter = 0
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        path_counter = 0.0
        taken_cells = 0.0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                    taken_cells+=1
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
                    taken_cells+=1
        free_cells = len(move_matrix)*len(move_matrix) - taken_cells



        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            paths = []
            paths_counts = []
            curr_path = []
            curr_user = ''
            curr_user_num_paths = 0
            curr_user_undo = 0
            curr_user_restart = 0
            curr_user_nodes = 0
            curr_user_sum_depth = 0
            curr_user_data = {}

            # user_data_headers = ['boardSize','boardType','condition','userid','curr_user_nodes','curr_user_num_paths','curr_user_sum_depth','curr_user_undo','curr_user_restart']


            for row in log_reader:
                # if row['userid'] == USERID:
                if curr_user=='':
                    curr_user = row['userid']



                if row['userid']!=curr_user:
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        add_path_count(paths_counts,copy.deepcopy(curr_path))
                        path_counter+=1
                        curr_user_num_paths+=1
                        curr_user_sum_depth = curr_user_sum_depth + len(curr_path)


                    cond = LOGFILE[g][5:-10]
                    condition_details = cond.split('_')
                    curr_user_data['boardSize'] = condition_details[0]
                    curr_user_data['boardType'] = condition_details[1]
                    curr_user_data['condition'] = condition_details[2]
                    curr_user_data['userid'] = curr_user


                    # print condition_details

                    curr_user_data['curr_user_num_paths'] = curr_user_num_paths
                    curr_user_data['curr_user_undo'] = curr_user_undo
                    curr_user_data['curr_user_restart'] = curr_user_restart
                    curr_user_data['curr_user_nodes'] = curr_user_nodes
                    curr_user_data['curr_user_sum_depth'] = curr_user_sum_depth
                    users_data.append(copy.deepcopy(curr_user_data))

                    curr_user_num_paths = 0
                    curr_user_undo = 0
                    curr_user_restart = 0
                    curr_user_nodes = 0
                    curr_user_sum_depth = 0
                    curr_user_data = {}

                    curr_path = []
                    curr_user = row['userid']
                    user_counter+=1

                elif row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                        # move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                        curr_path.append([rowPos,colPos,player])
                        curr_user_nodes+=1

                elif row['key']=='reset':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1

                        curr_user_num_paths+=1
                        # curr_user_nodes+=1
                        curr_user_sum_depth = curr_user_sum_depth + len(curr_path)
                        curr_user_restart+=1
                        curr_path = []
                elif row['key']=='undo':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1

                        curr_user_num_paths+=1
                        # curr_user_nodes+=1
                        curr_user_sum_depth = curr_user_sum_depth + len(curr_path)
                        curr_user_undo+=1
                        curr_path = curr_path[:-1]

                elif row['key'] == 'confidence':
                    curr_user_data['confidence'] = row['value']

                elif row['key'] == 'solvedCorrect':
                    curr_user_data['correctness'] = row['value']

        print LOGFILE[g][5:]

    dataFile = open('userStats/participantsStats_jan20.csv', 'wb')

    dataWriter = csv.DictWriter(dataFile, fieldnames=user_data_headers, delimiter=',')
    dataWriter.writeheader()
    for record in users_data:
        # print record
        dataWriter.writerow(record)
        # for path in paths:
        #     print path
        #     print '-----------'

        # pk = []
        # for p in paths_counts:
        #     # print p
        #     pk.append(p[1]/path_counter)
        #
        # ent = stats.entropy(pk)
        # print ent
    print user_counter



def entropy_paths(subpaths = False):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        path_counter = 0.0
        taken_cells = 0.0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                    taken_cells+=1
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
                    taken_cells+=1
        free_cells = len(move_matrix)*len(move_matrix) - taken_cells

        # to_ignore = IGNORE_LIST[g];
        # to_ignore = None
        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            paths = []
            paths_counts = []
            curr_path = []
            curr_user = ''
            move_count = 0.0
            for row in log_reader:
                # if row['userid'] == USERID:
                if curr_user=='':
                    curr_user = row['userid']

                if row['userid']!=curr_user:
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        add_path_count(paths_counts,copy.deepcopy(curr_path))
                        path_counter+=1
                    curr_path = []
                    curr_user = row['userid']

                elif row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                        # move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                        curr_path.append([rowPos,colPos,player])
                        if len(curr_path)==1 & player == 1:
                            move_matrix[rowPos][colPos] += 1
                            move_count +=1
                        # move_counter+=1.0
                            # else:
                            #     print 'ignore'
                elif row['key']=='reset':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1
                        curr_path = []
                elif row['key']=='undo':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1
                        curr_path = curr_path[:-1]

        # print LOGFILE[g]
        # for path in paths:
        #     print path
        #     print '-----------'

        pk = []
        for p in paths_counts:
            # print p
            pk.append(p[1]/path_counter)

        ent = stats.entropy(pk)

        condition = LOGFILE[g][5:-10].replace("_",",")

        measure = 'path entropy (no subpaths)'
        if (subpaths):
            measure = 'path entropy (subpaths)'

        # print condition+',' + measure + ',' + str(ent)

        condition = condition + "," + str(subpaths)
        # if (subpaths):
        #     condition = condition + "," + "(subpaths)"
        # else:
        #     condition = condition + "_" + "with pruned cell"
        # for i in range(len(entropy_values)):
        print condition + ',' + str(ent)

        ###heatmap first moves
        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[i])):
                if (move_matrix[r][j]=='X'):
                    move_matrix[r][j] = -0.00001
                elif (move_matrix[r][j]=='O'):
                    move_matrix[r][j] = -0.00002

        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[i])):
                if (move_matrix[r][j]!=-0.00001) & (move_matrix[r][j]!=-0.00002):
                    move_matrix[r][j] = move_matrix[r][j]/move_count
        a = np.array(move_matrix)
        a = np.flip(a,0)
        print a
        heatmap = plt.pcolor(a)

        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                if((a[y,x]==-1) | (a[y,x]==-0.00001)):
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
        fig_file_name=fig_file_name[:-4]
        data_matrices[fig_file_name[5:-6]] = move_matrix
        fig_file_name = fig_file_name + 'first_moves.png'
        # plt.savefig(fig_file_name)
        plt.clf()
        # print condition+',entropy, aggregated,' + str(subpaths)+ ',' + str(ent)
    return data_matrices
        # print ent


def entropy_paths_average(subpaths = False):
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        path_counter = 0.0
        taken_cells = 0.0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                    taken_cells+=1
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
                    taken_cells+=1
        free_cells = len(move_matrix)*len(move_matrix) - taken_cells

        to_ignore = IGNORE_LIST[g];
        # to_ignore = None
        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            paths = []
            paths_counts = []
            entropy_values = []
            curr_path = []
            curr_user = ''
            for row in log_reader:
                # if row['userid'] == USERID:
                if curr_user=='':
                    curr_user = row['userid']

                if row['userid']!=curr_user:
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1
                    pk = []
                    for p in paths_counts:
                        # print p
                        pk.append(p[1]/path_counter)

                    ent = stats.entropy(pk)
                    if (path_counter>0):
                        entropy_values.append(ent)
                    curr_path = []
                    curr_user = row['userid']
                    paths_counts = []
                    path_counter = 0
                    paths = []

                elif row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                        # move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                        curr_path.append([rowPos,colPos,player])
                        # move_counter+=1.0
                            # else:
                            #     print 'ignore'
                elif row['key']=='reset':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1
                        curr_path = []
                elif row['key']=='undo':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        if(subpaths):
                            add_path_count_subpaths(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=len(curr_path)
                        else:
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter+=1
                        curr_path = curr_path[:-1]

        # print LOGFILE[g]
        avg_ent = sum(entropy_values)/len(entropy_values)

        condition = LOGFILE[g][5:-10].replace("_",",")
        measure = 'avg path entropy (no subpaths)'
        if (subpaths):
            measure = 'avg path entropy (subpaths)'

        # print condition+',' + measure + ',' + str(avg_ent)
        # print avg_ent

        condition = condition + "," + str(subpaths)
        for i in range(len(entropy_values)):
            print condition + ',' + str(entropy_values[i])


def check_participant_answer(userid):
    with open('logs/tttResultsCogSci.csv', 'rb') as csvfile:
        log_reader = csv.DictReader(csvfile)
        for row in log_reader:
            if row['userid']==userid:
                if row['solvedCorrect'] == 'TRUE':
                    if row['validatedCorrect'] == 'TRUE':
                        return 'validatedCorrect'
                    else:
                        return 'solvedCorrect'
        return 'wrong'


def moves_stats(output_file):
    moves_data_matrics = {}
    data_first_moves = {}

    first_moves_data_matrices = {}
    results_table = []
    for g in range(len(LOGFILE)):
        # print g
        initial_board = copy.deepcopy(START_POSITION[g])

        path_counter = 0.0
        path_counter_subpaths = 0.0
        taken_cells = 0.0

        for i in range(len(initial_board)):
            for j in range(len(initial_board[i])):
                if ((initial_board[i][j]!=1) & (initial_board[i][j]!=2)):
                    initial_board[i][j] = int(initial_board[i][j])
                elif (initial_board[i][j]==1):
                    initial_board[i][j]='X'
                    taken_cells+=1
                elif (initial_board[i][j]==2):
                    initial_board[i][j]='O'
                    taken_cells+=1

        move_matrix = copy.deepcopy(initial_board)
        first_move_matrix = copy.deepcopy(initial_board)

        curr_move_matrix = copy.deepcopy(move_matrix)
        curr_first_move_matrix = copy.deepcopy(move_matrix)



        with open(LOGFILE[g], 'rb') as csvfile:
            print LOGFILE[g]
            condition = LOGFILE[g][5:-10].replace("_",",")
            board_name = LOGFILE[g]
            board_name=board_name[:-4]
            log_reader = csv.DictReader(csvfile)
            move_number = 1
            move_stack = []
            curr_path = []
            curr_user = ''
            prev_time = None
            prev_action = None
            initial_time = None
            for row in log_reader:
                curr_data = {}
                if curr_user == '':
                    curr_user = row['userid']
                    participant_answer = check_participant_answer(curr_user)

                if row['userid'] != curr_user:
                    participant_answer = check_participant_answer(curr_user)

                    # reset all values for next user
                    prev_time = None
                    prev_action = None
                    prev_row = None
                    first_move_for_player = True
                    move_number = 1
                    curr_path = []
                    move_stack = []
                    curr_user = row['userid']

                elif row['key'] == 'clickPos':
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    move_stack.append((rowPos, colPos))
                    player = int(row['value'][4])
                    first_move = False
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if (curr_move_matrix[rowPos][colPos]!='X') & (curr_move_matrix[rowPos][colPos]!='O'):
                        curr_move_matrix[rowPos][colPos] = player
                        curr_path.append([rowPos, colPos, player])

                        if len(curr_path) == 1:
                            first_move = True
                    time_between = 0
                    if prev_time == None:
                        if prev_row != None:
                            # print 'why'
                            prev_time = prev_row['time']
                            initial_time = int(prev_row['time'])
                    if prev_time == None:
                        print row
                    time_between = int(row['time']) - int(prev_time)
                    time_between = time_between/1000.0



                    curr_data['userid'] = curr_user
                    curr_data['condition'] = condition
                    curr_data['board_name'] = board_name
                    curr_data['solved'] = participant_answer
                    curr_data['player'] = player
                    curr_data['position'] = str(rowPos) + '_' + str(colPos)
                    curr_data['time'] = row['time']
                    curr_data['time_rel'] = int(row['time']) - initial_time
                    curr_data['time_prev_move'] = prev_time
                    curr_data['time_between'] = time_between
                    curr_data['prev_action'] = prev_action
                    curr_data['first_move_in_path'] = first_move
                    curr_data['move_number'] = move_number
                    curr_data['move_number_in_path'] = len(curr_path)
                    results_table.append(copy.deepcopy(curr_data))
                    move_number += 1
                    prev_action = 'click'
                    prev_time = row['time']

                elif row['key'] == 'reset':
                    curr_move_matrix = copy.deepcopy(initial_board)
                    prev_action = 'reset'
                    prev_time = row['time']
                    curr_path = []

                elif row['key'] == 'undo':
                    if len(move_stack) > 0:
                        undo_move = move_stack.pop()
                        curr_move_matrix[undo_move[0]][undo_move[1]] = 0
                        curr_path = curr_path[:-1]
                    else:
                        print 'problem undo'

                    prev_action = 'undo'
                    prev_time = row['time']
                # else:
                #     prev_time = row['time']

                prev_row = copy.deepcopy(row)
    dataFile = open(output_file, 'wb')
    dataWriter = csv.DictWriter(dataFile, fieldnames=results_table[0].keys(), delimiter=',')
    dataWriter.writeheader()
    for record in results_table:
        dataWriter.writerow(record)




def paths_stats(participants = 'all'):
    moves_data_matrics = {}
    data_first_moves = {}

    first_moves_data_matrices = {}
    for g in range(len(LOGFILE)):
        # print g
        initial_board = copy.deepcopy(START_POSITION[g])

        path_counter = 0.0
        path_counter_subpaths = 0.0
        taken_cells = 0.0

        for i in range(len(initial_board)):
            for j in range(len(initial_board[i])):
                if ((initial_board[i][j]!=1) & (initial_board[i][j]!=2)):
                    initial_board[i][j] = int(initial_board[i][j])
                elif (initial_board[i][j]==1):
                    initial_board[i][j]='X'
                    taken_cells+=1
                elif (initial_board[i][j]==2):
                    initial_board[i][j]='O'
                    taken_cells+=1

        move_matrix = copy.deepcopy(initial_board)
        first_move_matrix = copy.deepcopy(initial_board)

        curr_move_matrix = copy.deepcopy(move_matrix)
        curr_first_move_matrix = copy.deepcopy(move_matrix)


        with open(LOGFILE[g], 'rb') as csvfile:
            first_moves_values = []
            data_first_moves_board = {}
            log_reader = csv.DictReader(csvfile)
            paths = []
            paths_counts = []
            paths_counts_subpaths = []
            entropy_values = []
            entropy_values_subpaths = []
            entropy_values_clicks = []
            entropy_values_first_moves = []
            avg_times_after_click = []
            avg_times_after_undo = []
            avg_times_after_reset = []
            curr_path = []
            curr_user = ''
            move_counter = 0.0
            total_moves = 0.0
            first_move_counter = 0.0
            num_participants = 0.0
            prev_action = None
            prev_time = None
            prev_row = None
            curr_times_after_click = []
            curr_times_after_undo = []
            curr_times_after_reset = []


            for row in log_reader:
                if curr_user == '':
                    curr_user = row['userid']

                if row['userid']!=curr_user:
                    participant_answer = check_participant_answer(curr_user)
                    if ((participants == 'all') | ((participants == 'validatedCorrect') & (participant_answer=='validatedCorrect')) | ((participants == 'solvedCorrect') & ((participant_answer=='solvedCorrect') | (participant_answer=='validatedCorrect'))) | ((participants == 'wrong') & (participant_answer=='wrong'))):
                        if len(curr_path)>0:
                            paths.append(copy.deepcopy(curr_path))

                            add_path_count_subpaths(paths_counts_subpaths,copy.deepcopy(curr_path))
                            path_counter_subpaths+=len(curr_path)
                            add_path_count(paths_counts,copy.deepcopy(curr_path))
                            path_counter += 1

                        pk = []
                        for p in paths_counts:
                            # print p
                            pk.append(p[1]/path_counter)

                        ent = stats.entropy(pk)
                        if (path_counter>0):
                            entropy_values.append(ent)

                        pk = []
                        for p in paths_counts_subpaths:
                            # print p
                            pk.append(p[1]/path_counter_subpaths)

                        ent = stats.entropy(pk)
                        if (path_counter_subpaths>0):
                            entropy_values_subpaths.append(ent)

                        if move_counter > 0:
                            pk = []
                            # normalize move matrices
                            for i in range(len(curr_move_matrix)):
                                for j in range(len(curr_move_matrix[i])):
                                    if ((curr_move_matrix[i][j]!='X') & (curr_move_matrix[i][j]!='O')):
                                        pk.append(curr_move_matrix[i][j]/move_counter)
                                        curr_move_matrix[i][j] = curr_move_matrix[i][j]/move_counter
                                        move_matrix[i][j] += curr_move_matrix[i][j]
                            ent_moves = stats.entropy(pk)

                            pk_uniform = []
                            for i in range(len(pk)):
                                pk_uniform.append(1.0/len(pk))

                            ent_moves_norm_max = ent/stats.entropy(pk_uniform)
                            condition = LOGFILE[g][5:-10].replace("_",",")
                            # print condition + ',' + curr_user + ',' + str(ent_moves_norm_max) + ',' + participant_answer
                            entropy_values_clicks.append(ent_moves_norm_max)

                            # normalize move matrices
                            pk = []
                            for i in range(len(curr_first_move_matrix)):
                                for j in range(len(curr_first_move_matrix[i])):
                                    if ((curr_first_move_matrix[i][j]!='X') & (curr_first_move_matrix[i][j]!='O')):
                                        pk.append(curr_first_move_matrix[i][j]/first_move_counter)
                                        curr_first_move_matrix[i][j] = curr_first_move_matrix[i][j]/first_move_counter
                                        first_move_matrix[i][j] += curr_first_move_matrix[i][j]
                            ent_first_moves = stats.entropy(pk)
                            pk_uniform = []
                            for i in range(len(pk)):
                                pk_uniform.append(1.0/len(pk))

                            ent_first_moves_norm_max = ent_first_moves/stats.entropy(pk_uniform)
                            entropy_values_first_moves.append(ent_first_moves_norm_max)

                            # times
                            if len(curr_times_after_click) > 0:
                                avg_times_after_click.append(sum(curr_times_after_click)/len(curr_times_after_click))
                            if len(curr_times_after_reset) > 0:
                                avg_times_after_reset.append(sum(curr_times_after_reset)/len(curr_times_after_reset))
                            if len(curr_times_after_undo) > 0:
                               avg_times_after_undo.append(sum(curr_times_after_undo)/len(curr_times_after_undo))

                            data_first_moves_board[curr_user] = copy.deepcopy(first_moves_values)
                            total_moves += move_counter
                            num_participants += 1.0

                    # reset all values for next user
                    curr_path = []
                    curr_user = row['userid']
                    paths_counts = []
                    path_counter = 0
                    path_counter_subpaths = 0
                    paths_counts_subpaths = []
                    paths = []
                    move_counter = 0.0
                    first_move_counter = 0
                    curr_first_move_matrix = copy.deepcopy(initial_board)
                    curr_move_matrix = copy.deepcopy(initial_board)
                    curr_times_after_click = []
                    curr_times_after_undo = []
                    curr_times_after_reset = []
                    prev_time = None
                    prev_action = None
                    first_moves_values = []



                if row['key'] == 'clickPos':
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((curr_move_matrix[rowPos][colPos]!='X') & (curr_move_matrix[rowPos][colPos]!='O')):
                        curr_move_matrix[rowPos][colPos] += 1
                        move_counter+=1.0
                        curr_path.append([rowPos,colPos,player])
                        if len(curr_path) == 1:
                            curr_first_move_matrix[rowPos][colPos] +=1
                            first_moves_values.append((rowPos,colPos))
                            first_move_counter += 1.0
                    time_between = 0
                    if prev_time == None:
                        if prev_row != None:
                            # print 'why'
                            prev_time = prev_row['time']

                    if int(row['time']) < int(prev_time):
                        print int(row['time'])
                        print int(prev_time)
                        print 'problem'
                    time_between = int(row['time']) - int(prev_time)
                    if (time_between < 120000):
                        time_between = time_between/1000.0
                        # print time_between
                        # if (prev_action is None) | (prev_action == 'reset'):
                        if len(curr_path) == 1:
                            curr_times_after_reset.append(time_between)
                        elif prev_action == 'undo':
                            curr_times_after_undo.append(time_between)
                        elif prev_action == 'click':
                            curr_times_after_click.append(time_between)

                    prev_action = 'click'
                    prev_time = row['time']

                elif row['key'] == 'reset':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                        add_path_count_subpaths(paths_counts_subpaths,copy.deepcopy(curr_path))
                        path_counter_subpaths+=len(curr_path)
                        add_path_count(paths_counts,copy.deepcopy(curr_path))
                        path_counter += 1
                        curr_path = []
                    prev_action = 'reset'
                    prev_time = row['time']

                elif row['key'] == 'undo':
                    if len(curr_path) > 0:
                        paths.append(copy.deepcopy(curr_path))
                        add_path_count_subpaths(paths_counts_subpaths,copy.deepcopy(curr_path))
                        path_counter_subpaths += len(curr_path)
                        add_path_count(paths_counts,copy.deepcopy(curr_path))
                        path_counter += 1
                        curr_path = curr_path[:-1]
                    prev_action = 'undo'
                    prev_time = row['time']
                prev_row = copy.deepcopy(row)
        # print LOGFILE[g]
        avg_ent = sum(entropy_values)/len(entropy_values)
        avg_ent_subpaths = sum(entropy_values_subpaths)/len(entropy_values_subpaths)
        avg_ent_moves = sum(entropy_values_clicks)/len(entropy_values_clicks)
        # print 'entropy_values_clicks'
        # print entropy_values_clicks
        avg_ent_first_moves = sum(entropy_values_first_moves)/len(entropy_values_first_moves)
        # avg_times_after_click_agg = sum(avg_times_after_click)/len(avg_times_after_click)
        avg_times_after_click_agg = np.median(avg_times_after_click)
        std_times_after_click_agg = np.std(avg_times_after_click)
        # avg_times_after_undo_agg = sum(avg_times_after_undo)/len(avg_times_after_undo)
        avg_times_after_undo_agg = np.median(avg_times_after_undo)
        std_times_after_undo_agg = np.std(avg_times_after_undo)
        # avg_times_after_reset_agg = sum(avg_times_after_reset)/len(avg_times_after_reset)
        avg_times_after_reset_agg = np.median(avg_times_after_reset)
        std_times_after_reset_agg = np.std(avg_times_after_reset)
        #
        # avg_times_after_click_agg = np.median(curr_times_after_click)
        # std_times_after_click_agg = np.std(curr_times_after_click)
        # # avg_times_after_undo_agg = sum(curr_times_after_undo)/len(curr_times_after_undo)
        # avg_times_after_undo_agg = np.median(curr_times_after_undo)
        # std_times_after_undo_agg = np.std(curr_times_after_undo)
        # # avg_times_after_reset_agg = sum(curr_times_after_reset)/len(curr_times_after_reset)
        # avg_times_after_reset_agg = np.median(curr_times_after_reset)
        # std_times_after_reset_agg = np.std(curr_times_after_reset)


        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!='X') & (move_matrix[i][j]!='O')):
                    move_matrix[i][j] = move_matrix[i][j]/num_participants
                elif move_matrix[i][j] == 'X':
                    move_matrix[i][j]  = -0.00001
                elif move_matrix[i][j] == 'O':
                    move_matrix[i][j]  = -0.00002

        for i in range(len(first_move_matrix)):
            for j in range(len(first_move_matrix[i])):
                if ((first_move_matrix[i][j]!='X') & (first_move_matrix[i][j]!='O')):
                    first_move_matrix[i][j] = first_move_matrix[i][j]/num_participants
                elif first_move_matrix[i][j] == 'X':
                    first_move_matrix[i][j]  = -0.00001
                elif first_move_matrix[i][j] == 'O':
                    first_move_matrix[i][j]  = -0.00002


        # data_first_moves[condition] =
        condition = LOGFILE[g][5:-10].replace("_",",")
        # print condition
        # print first_moves_values

        # print condition + "," + str(avg_ent_moves) + "," +str(avg_ent_first_moves) + "," + str(avg_ent) + ","+ str(avg_ent_subpaths) + ","\
        #       + str(avg_times_after_click_agg) + ","+ str(avg_times_after_undo_agg) + "," + str(avg_times_after_reset_agg) + "," \
        # + str(std_times_after_click_agg) + ","+ str(std_times_after_undo_agg) + "," + str(std_times_after_reset_agg) + "," + str(total_moves/num_participants)\
        #       + "," + str(num_participants) + "," + participants

        board_name = LOGFILE[g]
        board_name=board_name[:-4]
        moves_data_matrics[board_name[5:-6]] = move_matrix
        first_moves_data_matrices[board_name[5:-6]] = first_move_matrix
        data_first_moves[board_name[5:-6]] = copy.deepcopy(data_first_moves_board)
    # write_matrices_to_file(moves_data_matrics, 'data_matrices/avg_people_clicks_' +participants +  '.json')
    # write_matrices_to_file(first_moves_data_matrices, 'data_matrices/avg_people_first_moves_' +participants +  '.json')
    write_matrices_to_file(data_first_moves, 'data_matrices/people_first_moves_values_byParticipant_' +participants +  '.json')
    return (move_matrix, first_move_matrix, first_moves_values)


# def normalize_matrix(am)


def first_moves_average():
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        path_counter = 0.0
        taken_cells = 0.0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                    taken_cells+=1
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
                    taken_cells+=1
        free_cells = len(move_matrix)*len(move_matrix) - taken_cells

        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            paths = []
            paths_counts = []
            entropy_values = []
            curr_path = []
            curr_user = ''
            for row in log_reader:
                if curr_user=='':
                    curr_user = row['userid']

                if row['userid']!=curr_user:
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))
                    pk = []
                    for p in paths_counts:
                        # print p
                        pk.append(p[1]/path_counter)

                    ent = stats.entropy(pk)
                    if (path_counter>0):
                        entropy_values.append(ent)
                    curr_path = []
                    curr_user = row['userid']
                    paths_counts = []
                    path_counter = 0
                    paths = []

                elif row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])

                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                        # move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                        curr_path.append([rowPos,colPos,player])
                        # move_counter+=1.0
                            # else:
                            #     print 'ignore'
                elif row['key']=='reset':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))

                        curr_path = []
                elif row['key']=='undo':
                    if len(curr_path)>0:
                        paths.append(copy.deepcopy(curr_path))

                        curr_path = curr_path[:-1]

        # print LOGFILE[g]
        avg_ent = sum(entropy_values)/len(entropy_values)

        condition = LOGFILE[g][5:-10].replace("_",",")
        measure = 'avg path entropy (no subpaths)'

        for i in range(len(entropy_values)):
            print condition + ',' + str(entropy_values[i])



def entropy_board(ignore = False):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        move_counter = 0.0
        taken_cells = 0.0;
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                    taken_cells+=1
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
                    taken_cells+=1
        free_cells = len(move_matrix)*len(move_matrix) - taken_cells
        # free_cells = free_cells - 2 #ignoring
        # print free_cells
        # move_matrix = []
        # for row in range(DIMENSION):
        #     row_positions = []
        #     for col in range(DIMENSION):
        #         row_positions.append(0)
        #     move_matrix.append(copy.deepcopy(row_positions))
        # to_ignore = IGNORE_LIST[g];
        to_ignore = None
        if ignore == True:
            to_ignore = IGNORE_LIST[g];
        if to_ignore!=None:
            free_cells = free_cells - len(to_ignore) # ignoring

        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            for row in log_reader:
                # if row['userid'] == USERID:
                if row['key']=='clickPos':
                    # print row
                    rowPos = int(row['value'][0])
                    colPos = int(row['value'][2])
                    player = int(row['value'][4])
                    # print rowPos
                    # print colPos
                    # print move_matrix[rowPos][colPos]
                    if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                        if (to_ignore!=None):
                            if ((rowPos!=to_ignore[0][0] | colPos!=to_ignore[0][1]) & (rowPos!=to_ignore[1][0] | colPos!=to_ignore[1][1])):
                                if(player==1):
                                    move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                                    move_counter+=1.0
                            # else:
                            #     print 'ignore'
                        elif (player==1):
                            move_matrix[rowPos][colPos] = move_matrix[rowPos][colPos]+1
                            move_counter+=1.0
        # print LOGFILE[g]
        # for row in move_matrix:
        #     print row

        #compute entropy
        pk = []
        cell_counter = 0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!='X') & (move_matrix[i][j]!='O')):
                    pk.append(move_matrix[i][j]/move_counter)
        # print pk
        ent = stats.entropy(pk)
        pk_uniform = []
        for i in range(len(pk)):
            pk_uniform.append(1.0/len(pk))

        # print pk_uniform

        ent_norm = ent/free_cells
        ent_norm_max = ent/stats.entropy(pk_uniform)
        # print ent
        # print ent_norm
        condition = LOGFILE[g][5:-10].replace("_",",")
        # if (ignore):
        #     print condition+',entropy (without pruned cells),' + str(ent)
        #     print condition+',entropy normalized free cells (without pruned cells),'+ str(ent_norm)
        #     print condition+',entropy normalized max (without pruned cells),' + str(ent_norm_max)
        # else:
        #     print condition+',entropy (with pruned cells),' + str(ent)
        #     print condition+',entropy normalized free cells (with pruned cells),'+ str(ent_norm)
        #     print condition+',entropy normalized max (with pruned cells),' + str(ent_norm_max)
        # # print condition+',std entropy normalized (participant),' + str(std_entropy_norm)

        if (ignore):
            condition = condition + "_" + "without pruned cell"
        else:
            condition = condition + "_" + "with pruned cell"
        # for i in range(len(entropy_values)):
        print condition + ',' + str(ent) + ',' + str(ent_norm_max) + "," + str(ent_norm)


        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[r])):
                if (move_matrix[r][j]=='X'):
                    move_matrix[r][j] = -0.00001
                elif (move_matrix[r][j]=='O'):
                    move_matrix[r][j] = -0.00002

        for r in range(0,len(move_matrix)):
            for j in range(0,len(move_matrix[r])):
                if (move_matrix[r][j]!=-0.00001) & (move_matrix[r][j]!=-0.00002):
                    move_matrix[r][j] = move_matrix[r][j]/move_counter

        fig_file_name = LOGFILE[g]
        fig_file_name=fig_file_name[:-4]
        data_matrices[fig_file_name[5:-6]] = (move_matrix, ent_norm_max)
        print ent_norm_max
    return data_matrices


def entropy_board_average(ignore = False):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        # print g
        move_matrix = copy.deepcopy(START_POSITION[g])
        move_counter = 0.0
        taken_cells = 0.0
        for i in range(len(move_matrix)):
            for j in range(len(move_matrix[i])):
                if ((move_matrix[i][j]!=1) & (move_matrix[i][j]!=2)):
                    move_matrix[i][j] = int(move_matrix[i][j])
                elif (move_matrix[i][j]==1):
                    move_matrix[i][j]='X'
                    taken_cells+=1
                elif (move_matrix[i][j]==2):
                    move_matrix[i][j]='O'
                    taken_cells+=1
        free_cells = len(move_matrix)*len(move_matrix) - taken_cells

        to_ignore = None
        if ignore == True:
            to_ignore = IGNORE_LIST[g];
        if to_ignore!=None:
            free_cells = free_cells - len(to_ignore)

        curr_user = ''
        entropy_values = []
        entropy_values_norm = []
        curr_move_matrix = copy.deepcopy(move_matrix)
        prob_matrix = copy.deepcopy(move_matrix)
        with open(LOGFILE[g], 'rb') as csvfile:
            log_reader = csv.DictReader(csvfile)
            for row in log_reader:
                if curr_user == '':
                    curr_user = row['userid']

                if row['userid']!=curr_user:
                    #compute entropy
                    if move_counter>0:
                        pk = []
                        for i in range(len(move_matrix)):
                            for j in range(len(move_matrix[i])):
                                # prob_matrix[i][j] = np.around(move_matrix[i][j],2)
                                if ((move_matrix[i][j]!='X') & (move_matrix[i][j]!='O')):
                                    pk.append(curr_move_matrix[i][j]/move_counter)
                                    prob_matrix[i][j] = curr_move_matrix[i][j]/move_counter
                                    # print prob_matrix[i][j]
                                else:
                                    prob_matrix[i][j] = 0
                        # print pk
                        ent = stats.entropy(pk)
                        entropy_values.append(ent)
                        ent_norm = ent/free_cells
                        entropy_values_norm.append(ent_norm)

                        curr_move_matrix = copy.deepcopy(move_matrix)
                        move_counter = 0.0
                    curr_user = row['userid']

                else:
                    if row['key']=='clickPos':
                        # print row
                        rowPos = int(row['value'][0])
                        colPos = int(row['value'][2])
                        player = int(row['value'][4])
                        # print rowPos
                        # print colPos
                        # print move_matrix[rowPos][colPos]
                        if ((move_matrix[rowPos][colPos]!='X') & (move_matrix[rowPos][colPos]!='O')):
                            if (to_ignore!=None):
                                if ((rowPos!=to_ignore[0][0] | colPos!=to_ignore[0][1]) & (rowPos!=to_ignore[1][0] | colPos!=to_ignore[1][1])):
                                    if(player==1):
                                        curr_move_matrix[rowPos][colPos] = curr_move_matrix[rowPos][colPos]+1
                                        move_counter+=1.0
                                # else:
                                #     print 'ignore'
                            elif (player==1):
                                curr_move_matrix[rowPos][colPos] = curr_move_matrix[rowPos][colPos]+1
                                move_counter+=1.0
        # print LOGFILE[g][5:-10]
        # for row in move_matrix:
        #     print row
        avg_entropy = sum(entropy_values)/len(entropy_values)
        avg_entropy_norm = sum(entropy_values_norm)/len(entropy_values_norm)

        uniform_moves = []
        for i in range(int(free_cells)):
            uniform_moves.append(1.0/free_cells)



        max_entropy = stats.entropy(uniform_moves)


        # print max_entropy

        std_entropy = np.std(entropy_values);
        std_entropy_norm = np.std(entropy_values_norm);

        entropy_values_norm_max = []
        # print entropy_values
        for i in range(len(entropy_values)):
            entropy_values_norm_max.append(entropy_values[i]/max_entropy)

        avg_entropy_norm_max = sum(entropy_values_norm_max)/len(entropy_values_norm_max)
        std_entropy_norm_max = np.std(entropy_values_norm_max);
        # std_entropy_norm_max = ' '
        # avg_entropy_norm_max = ''

        # print np.around(0.555,2)

        condition = LOGFILE[g][5:-10].replace("_",",")
        # if (ignore):
        #     print condition+',avg entropy (without pruned cells),' + str(avg_entropy)
        #     print condition+',std entropy (without pruned cells),' + str(std_entropy)
        #     print condition+',avg entropy normalized free cells (without pruned cells),' + str(avg_entropy_norm)
        #     print condition+',std entropy normalized free cells (without pruned cells),' + str(std_entropy_norm)
        #     print condition+',avg entropy normalized max (without pruned cells),' + str(avg_entropy_norm_max)
        #     print condition+',std entropy normalized max (without pruned cells),' + str(std_entropy_norm_max)
        #
        # else:
        #     print condition+',avg entropy (with pruned cells),' + str(avg_entropy)
        #     print condition+',std entropy (with pruned cells),' + str(std_entropy)
        #     print condition+',avg entropy normalized free cells (with pruned cells),' + str(avg_entropy_norm)
        #     print condition+',std entropy normalized free cells (with pruned cells),' + str(std_entropy_norm)
        #     print condition+',avg entropy normalized max (with pruned cells),' + str(avg_entropy_norm_max)
        #     print condition+',std entropy normalized max (with pruned cells),' + str(std_entropy_norm_max)

        if (ignore):
            condition = condition + "_" + "without pruned cell"
        else:
            condition = condition + "_" + "with pruned cell"
        # for i in range(len(entropy_values)):
        #     print condition + ',' + str(entropy_values[i]) + ',' + str(entropy_values_norm_max[i]) + "," + str(entropy_values_norm[i])

        fig_file_name = LOGFILE[g]
        fig_file_name=fig_file_name[:-4]
        data_matrices[fig_file_name[5:-6]] = (move_matrix, avg_entropy_norm_max)
        print avg_entropy_norm_max
    return data_matrices


        # print avg_entropy_norm

        # prob_mat = np.matrix(prob_matrix)
        # prob_mat = prob_mat.round(2)
        #
        # prob_matrix = np.around(prob_matrix,2)
        # print prob_matrix
        # print len(entropy_values)
        # print entropy_values
        # entropy_values = np.sort(entropy_values)
        # print entropy_values


def get_games():
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

        return all_games


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



def write_matrices_to_file(data_matrices, filename):
  with open(filename, 'w') as fp:
      json.dump(data_matrices, fp)



if __name__ == "__main__":
    # heat_map_solution(normalized=True)
    # paths_stats(participants='all')
    # paths_stats(participants='solvedCorrect')
    # paths_stats(participants='wrong')
    # paths_stats(participants='wrong')
    moves_stats('stats/testMovesStats3.csv')
    # seperate_log('logs/fullLogCogSci.csv')
    # # entropy_board()
    # # entropy_board(ignore=True)
    # # entropy_board_average()
    # # entropy_board_average(ignore=True)
    # entropy_paths(subpaths=True)
    # entropy_paths_average(subpaths=True)
    # entropy_paths(subpaths=False)
    # entropy_paths_average(subpaths=False)

    # user_stats()

    # print stats.entropy([0.25,0.25,0.25,0.25])
    # heat_map_game(normalized=True)
    # heat_map_solution(normalized=True)
    # run_analysis()
    # replay()
