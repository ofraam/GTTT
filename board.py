import config as c
import math

class Board:
  def __init__(self, num_spaces, winning_paths, board = None):
    ''' Initializes a Board object '''
    self.size = num_spaces
    self.winning_paths = winning_paths
    if board:
      self.board = board
    else:
      self.board = {}
      for space in xrange(1, num_spaces+1):
        self.board[space] = c.BLANK
    self.last_space = 0

  def get_board(self):
    ''' Returns a dictionary representing the state of the board'''
    return self.board
    
    
  def get_board_copy(self):
    ''' Returns a copy of the Board object '''
    board_copy = self.board.copy()
    return Board(self.size, self.winning_paths, board_copy)
    
  def get_player(self, space):
    ''' Returns the player who occupies <space>. If no player occupies
    <space>, it returns BLANK. 
    '''
    return self.board[space]
    
  def is_empty(self):
    ''' Returns True if the board is empty, False if not. '''
    for space in self.board:
      if self.board[space] != c.BLANK:
        return False
    return True
    
  def add_marker(self, space, player):
    self.last_space = space
    ''' If <space> is unoccupied, add <player>'s marker to <space>.
    Else, throw an exception, indicating that either the human or the AI
    is making an invalid move. '''
    if self.get_player(space) is c.BLANK:
      self.board[space] = player
    else:
      raise Exception("Space has to be c.BLANK!!")

  def get_HUMAN_spaces(self):
    ''' Return a list of spaces that human occupies. '''
    list_of_spaces = []
    for space in self.board:
      if self.board[space] == c.HUMAN:
        list_of_spaces.append(space)
    return list_of_spaces
    
  def get_COMPUTER_spaces(self):
    ''' Return a list of spaces that COMPUTER occupies. '''
    list_of_spaces = []
    for space in self.board:
      if self.board[space] == c.COMPUTER:
        list_of_spaces.append(space)
    return list_of_spaces
    
  def get_free_spaces(self):
    ''' Return a list of unoccupied spaces. '''
    list_of_spaces = []
    for space in self.board:
      if self.board[space] == c.BLANK:
        list_of_spaces.append(space)
    return list_of_spaces   

  def get_manhattan_dist(self,space1, space2):
    dimension = math.sqrt(self.size)
    row1 = float(space1)/float(dimension)
    row1 = math.ceil(row1)
    row2 = float(space2)/float(dimension)
    row2 = math.ceil(row2)

    col1 = ((space1 - 1) % dimension) + 1
    col2 = ((space2 - 1) % dimension) + 1
    dist = abs(row1-row2) + abs(col1-col2)
    return dist

  def compute_avg_distance(self, space, list_of_occupied):
    sum_distances = 0.0
    for occupied_space in list_of_occupied:
      sum_distances = sum_distances + self.get_manhattan_dist(space,occupied_space)
    avg_dist = 0
    if len(list_of_occupied)>0:
      avg_dist = sum_distances/float(len(list_of_occupied))
    return avg_dist


  def get_free_spaces_ranked(self):
    ''' Return a list of unoccupied spaces. '''
    list_of_spaces = []
    list_of_occupied = []
    list_of_spaces_with_dist = []
    for space in self.board:
      if self.board[space] == c.BLANK:
        list_of_spaces.append(space)
      else:
        list_of_occupied.append(space)
    for free_space in list_of_spaces:
      list_of_spaces_with_dist.append((free_space,self.compute_avg_distance(free_space,list_of_occupied)))

    sorted_list = sorted(list_of_spaces_with_dist, key=lambda x: x[1])
    ranked_list  = [x[0] for x in sorted_list]
    return ranked_list

  def get_children(self, player):
    ''' Return a list of Board objects that are possible options for <player>'''
    free_spaces = self.get_free_spaces()
    children = []
    for space in free_spaces:
      board_copy = self.board.copy()
      board_copy[space] = player
      children.append(board_copy)
      
    return children

  def get_outcome(self):
    """return win score for win, lose for lose and zero for tie. No clever heuristics, just outcome"""
    score = 0

    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0

      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1

      if c.COMPUTER_count == len_path:
        return c.WIN_SCORE

      elif c.HUMAN_count == len_path:
        return c.LOSE_SCORE

    return score


  def obj(self, turns = 0):
    """ Heurisitc function to be used for the minimax search. 
    If it is a winning board for the COMPUTER, returns WIN_SCORE.
    If it is a losing board for the COMPUTER, returns LOSE_SCORE. 
    
    Then, for all the winning paths in which the COMPUTER has i spaces,
    and the HUMAN has 0 spaces, returns 10*3^i. 
    For all the winning paths in which the c.HUMAN is i away 
    from winning, returns -10*3^i
    """
    score = 0
      
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0
      
      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1
          
      if c.COMPUTER_count == len_path:
        # print path
        # Player wins!
        # print turns
        # print self.last_space
        # print c.WIN_SCORE-turns

        return c.WIN_SCORE-turns
      elif c.HUMAN_count == len_path:
        # print path
        # Opponent wins :(
        return c.LOSE_SCORE
      elif c.HUMAN_count == 0:
        # Opponent not on path, so count number of player's tokens on path  
        score += 10*3**(c.COMPUTER_count - 1)
      elif c.COMPUTER_count == 0:
        # Player not on path, so count number of opponent's tokens on path
        score -= 10*3**(c.HUMAN_count - 1)
      else:
        # Path cannot be won, so it has no effect on score
        pass 
    if score ==10000:
      print 'ahm'
    return score
    
  def is_terminal(self):
    ''' Returns True if the board is terminal, False if not. '''
    # First, check to see if the board is won
    # objective_score = self.obj()
    # if objective_score == c.WIN_SCORE:
    #   return c.COMPUTER
    # elif objective_score == c.LOSE_SCORE:
    #   return c.HUMAN
    for path in self.winning_paths:
      len_path = len(path)
      c.COMPUTER_count, c.HUMAN_count = 0, 0

      for space in path:
        if self.board[space] == c.COMPUTER:
          c.COMPUTER_count += 1
        elif self.board[space] == c.HUMAN:
          c.HUMAN_count += 1

      if c.COMPUTER_count == len_path:
        # print path
        # Player wins!
        return c.COMPUTER
      elif c.HUMAN_count == len_path:
        # print path
        # Opponent wins :(
        return c.HUMAN
    # else:
      # Then, check to see if there are any c.BLANK spaces
    for space in self.board:
      if self.board[space] == c.BLANK:
        return False
    return c.TIE
