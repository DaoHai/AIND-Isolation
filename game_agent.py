"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""

import random
import math
import operator

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return -math.inf
    if game.is_winner(player):
        return math.inf
    
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    blank_spaces = game.get_blank_spaces()
    percent_board_occupied = int((len(blank_spaces)/(game.width * game.height)) * 100)
    
    if percent_board_occupied < 30 :
        return float(player_moves - 2*opponent_moves)
    else:
        return float(2*player_moves - opponent_moves)
    

    # list of locations that fall onto the walls of board 
#    walls = [
#        [(0, i) for i in range(game.width)],
#        [(i, 0) for i in range(game.height)],
#        [(game.width - 1, i) for i in range(game.width)],
#        [(i, game.height - 1) for i in range(game.height)]
#    ]
#
#    # list of corner locations of the board  
#    corners = [(0,0), (0,game.width-1), (game.height-1,0), (game.height-1,game.width-1)]
#    
#    
#
#    player_moves = len(game.get_legal_moves(player))
#    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
#    
#    moves_wall = []
#    moves_corner = []
#    for move in game.get_legal_moves(player):
#        if move in walls:
#           moves_wall.append(move)
#        elif move in corners:
#           moves_corner.append(move)
#           
#    moves_wall_corner = len(moves_wall) + len(moves_corner)
#    better_moves= player_moves - moves_wall_corner
#    
#    

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return -math.inf
    if game.is_winner(player):
        return math.inf

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(2*player_moves-opponent_moves)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return -math.inf
    if game.is_winner(player):
        return math.inf

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(player_moves-2*opponent_moves)



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=30, score_fn=custom_score, timeout=35.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """


    def isTimeout(self):
        if self.time_left() < self.TIMER_THRESHOLD:
           raise SearchTimeout()

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left



        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move


    def maxvalue(self, game, max_depth, curr_depth):
        """
        Search for the branch with the highest score value.  Return that value.
        Parameters
        ---------------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        max_depth: int
            The maximum number of plies to check (this remains the same for
            Minimax, and will increase on each run in the AlphaBetaPlayer 
            which uses iterative deepening)
        curr_depth: int
            The current depth being checked (incremented on each call until it reaches
            max_depth)
        Returns
        -----------------
        val : int
            The score for the best move available (the maximum given all the legal moves available)
        """

        if self.time_left() < self.TIMER_THRESHOLD:
           raise SearchTimeout()

        # Reached the depth we wanted to consider. Return the score for that
        #   move (based on the heuristics in improved_score, or custom_score)
        if (max_depth==curr_depth):
            return self.score(game, self)
        else:

            # Increase depth counter for when we next call minvalue
            curr_depth+=1
            moves = game.get_legal_moves(self)
            if len(moves)==0:
                # No legals moves left, player would lose if we chose this branch
                return float("-inf")

            best_move = moves[0]
            maxv=float("-inf")

            # Consider each move. If it returns a higher predicted value than
            #   what was previously stored in best_move, replace best_move and maxv
            for m in moves:
                val = self.minvalue(game.forecast_move(m), max_depth, curr_depth)
                if (val>maxv):
                    maxv = val
                    best_move = m    
            return maxv

    

    def minvalue(self, game, max_depth, curr_depth):
        """
        Search for the branch with the highest score value.  Return that value.
        Parameters
        ---------------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        max_depth: int
            The maximum number of plies to check (this remains the same for
            Minimax, and will increase on each run in the AlphaBetaPlayer 
            which uses iterative deepening)
        curr_depth: int
            The current depth being checked (incremented on each call until it reaches
            max_depth)
        Returns
        -----------------
        val : int
            The score for the best move available (the maximum given all the legal moves available)
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Reached the depth we wanted to consider. Return the score for that
        #   move (based on the heuristics in improved_score, or custom_score)
        if (max_depth==curr_depth):
            return self.score(game, self)
        else:
            # Increase depth counter for when we next call maxvalue
            curr_depth+=1

            moves = game.get_legal_moves()
            if len(moves)==0:
                #no legal moves left for opponent, opponent would lose on this branch
                return float("+inf")

            best_move = moves[0]
            minv=float("inf")

            # Consider each move. If it returns a lower predicted value than
            #   what was previously stored in best_move, replace best_move and minv            
            for m in moves:
                val = self.maxvalue(game.forecast_move(m), max_depth, curr_depth)
                if (val<minv):
                    minv = val
                    best_move = m     
            return minv



    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

  
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


        moves = game.get_legal_moves()

        if len(moves)==0:
            best_move = (-1, -1)   # no moves left, forfeit game

        else:              
            best_move = moves[0]
            maxval=float("-inf")

            # We're not doing iterative deepening, so we put the try/except around 
            #  testing each move. If we run out of time, we return the best move that
            #  we've found so far.

            for m in moves:
                try:
                    val = self.minvalue(game.forecast_move(m), depth, 1)
            
                except SearchTimeout:
                    return best_move
                
                if val>maxval:
                    maxval = val
                    best_move = m    


        return best_move



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def isTimeout(self):
        if self.time_left() < self.TIMER_THRESHOLD:
           raise SearchTimeout()

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left


       # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        depth = 1

        # Test if first move of the game -- if so pick the center square
        if len(game.get_blank_spaces()) == game.width * game.height:
            return (3,3)


        while depth <= self.search_depth:   
            try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
                best_move = self.alphabeta(game, depth)
                depth+=1

            except SearchTimeout:
#                print("Timeout- spaces remaining: ", len(game.get_blank_spaces()))
                return best_move


#        print("Spaces remaining: ", len(game.get_blank_spaces()))
        # Return the best move from the last completed search iteration
        return best_move



    def maxvalue(self, game, alpha, beta, max_depth, curr_depth):
        """
        Search for the branch with the highest score value.  Return that value.
        Parameters
        ---------------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        alpha : float
            The low end of alpha-beta pruning -- while you are in minvalue() function,
            if minv is lower than this, you know this branch won't get considered in the
            maxvalue node above it, so you can stop testing
            
        beta : float
            The high end of alpha-beta pruning -- while you are in maxvalue() function, 
            if maxv is higher than this, you know this branch won't get considered in the 
            minvalue node above it, so you can stop testing 
        max_depth: int
            The maximum number of plies to check (this remains the same for
            Minimax, and will increase on each run in the AlphaBetaPlayer 
            which uses iterative deepening)
        curr_depth: int
            The current depth being checked (incremented on each call until it reaches
            max_depth)
        Returns
        -----------------
        val : float
            The score for the best move available (the maximum given all the legal moves available)
        """

        if self.time_left() < self.TIMER_THRESHOLD:
           raise SearchTimeout()


        # Reached the depth we wanted to consider. Return the score for that
        #   move (based on the heuristics in improved_score, or custom_score)
        if (max_depth==curr_depth):
            return self.score(game, self)

        else:
            # Increase depth counter for when we next call minvalue
            curr_depth+=1
            moves = game.get_legal_moves(self)
            if len(moves)==0:
                # No legals moves left
                return float("-inf")
            
            maxv = float("-inf")

            # Consider each move. If it returns a higher predicted value than
            #   what was previously stored in best_move, replace maxv

            for m in moves:
                val = self.minvalue(game.forecast_move(m), alpha, beta, max_depth, curr_depth)
                if val>maxv:
                    maxv = val

                # Check if maxv is more than beta - this means it is more than a neighboring node, and the
                # minvalue above will never choose this branch. You can stop searching this branch and 
                # return upwards.
                if maxv >= beta:
                    return maxv

                # Need to continue searching this branch. If maxv is higher than the previously saved alpha,
                #  set alpha to be equal to this new max
                else:
                    alpha = max(alpha, maxv) 

            return maxv


    def minvalue(self, game, alpha, beta, max_depth, curr_depth):
        """
        Search for the branch with the highest score value.  Return that value.
        Parameters
        ---------------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        alpha : float
            The low end of alpha-beta pruning -- while you are in minvalue() function,
            if minv is lower than this, you know this branch won't get considered in the
            maxvalue node above it, so you can stop testing
            
        beta : float
            The high end of alpha-beta pruning -- while you are in maxvalue() function, 
            if maxv is higher than this, you know this branch won't get considered in the 
            minvalue node above it, so you can stop testing 
        max_depth: int
            The maximum number of plies to check (this remains the same for
            Minimax, and will increase on each run in the AlphaBetaPlayer 
            which uses iterative deepening)
        curr_depth: int
            The current depth being checked (incremented on each call until it reaches
            max_depth)
        Returns
        -----------------
        val : float
            The score for the best move available (the maximum given all the legal moves available)
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Reached the depth we wanted to consider. Return the score for that
        #   move (based on the heuristics in improved_score, or custom_score)
        if (max_depth==curr_depth):
            return self.score(game, self)
        else:
            # Increase depth counter for when we next call maxvalue
            curr_depth+=1

            moves = game.get_legal_moves()
            if len(moves)==0:
                #no legal moves left for opponent, opponent would lose on this branch
                return float("+inf")

            minv = float("inf")

            # Consider each move. If a given move returns a lower predicted value than
            #   what was previously found, replace minv
            for m in moves:
                val = self.maxvalue(game.forecast_move(m), alpha, beta, max_depth, curr_depth)
                if val < minv:
                    minv = val

                # Check if minv is less than alpha - this means it is less than a neighboring node, and the
                # maxvalue above will never choose this branch. You can stop searching this branch and 
                # return upwards.
                if minv <= alpha:
                    return minv

                # Need to continue searching this branch. If minv is lower than the previously saved beta,
                #  set beta to be equal to this new min
                else:
                    beta = min(beta, minv)
            return minv


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


        moves = game.get_legal_moves()
        if len(moves)==0:    # no legal moves available, forfeit the game
            return (-1, -1)
             
        best_move = moves[0]
        maxval=float("-inf")

        # Consider each available move, and select the move with the highest
        #   score (returned from searching the tree "depth" nodes down)
        #   Each time this alphabeta() function is called, depth will be one
        #   higher, and the search will be that much deeper

        for m in moves:
            val = self.minvalue(game.forecast_move(m), alpha, beta, depth, 1)
                
            if val>maxval:
                maxval = val
                best_move = m    

            # Increase alpha to the max value already found - this way minvalue() 
            #  can stop searching if it knows it will return a value lower than
            #  a max that was already found
            alpha = max(alpha, maxval)

        return best_move