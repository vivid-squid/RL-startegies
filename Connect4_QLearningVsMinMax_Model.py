#!/usr/bin/env python
# coding: utf-8

import random
import math
from IPython.display import display
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

class Connect4_Game:
    
    def initialise_board(self) : 
        self.rows = 6
        self.columns = 7
        self.connect4_board = np.zeros((self.rows, self.columns))
    
    validateMove = lambda self, column: self.connect4_board[len(self.connect4_board)-1][column] == 0
    
    getNextAvailableRow = lambda self, column: next((row for row in range(len(self.connect4_board)) if self.connect4_board[row][column] == 0), None)

    getValidMove = lambda self: [column for column in range(self.columns) if self.validateMove(column)]

    def getNextAvailablePosition(self, letter):
        rows, cols = self.rows, self.columns
        for row, row_vals in enumerate(self.connect4_board):
            for col, col_val in enumerate(row_vals[:-3]):
                if all(elem == letter for elem in row_vals[col:col+4]):
                    return row, col
            for col, col_vals in zip(range(cols), (self.connect4_board[r][col] for r in range(row, min(row+4, rows)))):
                if all(elem == letter for elem in col_vals):
                    return row, col
            for col, col_vals in enumerate(row_vals[:-3]):
                if row < rows-3 and col < cols-3:
                    diag_vals = [self.connect4_board[row+i][col+i] for i in range(4)]
                    if all(elem == letter for elem in diag_vals):
                        return row, col
            for col, col_vals in enumerate(row_vals[:-3]):
                if row >= 3 and col < cols-3:
                    diag_vals = [self.connect4_board[row-i][col+i] for i in range(4)]
                    if all(elem == letter for elem in diag_vals):
                        return row, col
        else:
            return -1, -1

    def validateWin(self, letter):
        for row in range(self.rows):
            for col in range(self.columns - 3):
                if all(self.connect4_board[row][col + i] == letter for i in range(4)):
                    return True

        for row in range(self.rows - 3):
            for col in range(self.columns):
                if all(self.connect4_board[row + i][col] == letter for i in range(4)):
                    return True

        for row in range(self.rows - 3):
            for col in range(self.columns - 3):
                if all(self.connect4_board[row + i][col + i] == letter for i in range(4)):
                    return True

        for row in range(3, self.rows):
            for col in range(self.columns - 3):
                if all(self.connect4_board[row - i][col + i] == letter for i in range(4)):
                    return True

        return False
        
    def tossForFirstMove(self):
        choices = [1,2]
        return random.choice(choices)
        
    def validateFinalMove(self, SI_Agent_Letter, MinMax_Letter):
        return any(self.validateWin(letter) for letter in (SI_Agent_Letter, MinMax_Letter)) or not self.getValidMove()

class QLearning:
    def __init__(self):
        self.epsilon = 1.0
        self.QLearningStates = {}
    
    getPosition = lambda self, positions: int(''.join([str(int(position)) for position in positions.flatten()]))

    def getQLearningValue_For_Action(self, current_board, current_position):
        position = self.getPosition(current_board)
        if position not in self.QLearningStates:
            self.QLearningStates[(position, current_position)] = 0
        return self.QLearningStates[(position, current_position)]
    
    def getBestPositionFromQLearning(self, current_board, possible_positions):
        return random.choice(possible_positions) if random.random() < self.epsilon else max([(self.getQLearningValue_For_Action(current_board, position), position) for position in possible_positions], key=lambda x: x[0])[1]
            
    def loadQLearningModel(self):
        with open("Connect4QLearningTraningModel.pickle", "rb") as file:
            self.QLearningStates = pickle.load(file)

class MinMax : 
    
    def evaluate_MinMax_score(self, c4Game, letter, SIAgentLetter, MinMaxLetter):
        score = 0
        OtherPlayerLetter = MinMaxLetter if letter == SIAgentLetter else SIAgentLetter

        for i in range(c4Game.rows):
            row_array = [int(x) for x in list(c4Game.connect4_board[i,:])]
            col_array = [int(x) for x in list(c4Game.connect4_board[:,i])]
            for j in range(c4Game.columns-3):
                sub_row = row_array[j:j+4]
                sub_col = col_array[j:j+4]
                if sub_row.count(letter) == 4:
                    score += 1000
                elif sub_row.count(letter) == 3 and sub_row.count(0) == 1:
                    score += 100
                elif sub_row.count(letter) == 2 and sub_row.count(0) == 2:
                    score += 10
                if sub_row.count(OtherPlayerLetter) == 3 and sub_row.count(0) == 1:
                    score -= 10
                if sub_col.count(letter) == 4:
                    score += 1000
                elif sub_col.count(letter) == 3 and sub_col.count(0) == 1:
                    score += 100
                elif sub_col.count(letter) == 2 and sub_col.count(0) == 2:
                    score += 10
                if sub_col.count(OtherPlayerLetter) == 3 and sub_col.count(0) == 1:
                    score -= 10

        for i in range(c4Game.rows-3):
            for j in range(c4Game.columns-3):
                sub_diagonal1 = [c4Game.connect4_board[i+k][j+k] for k in range(4)]
                sub_diagonal2 = [c4Game.connect4_board[i+3-k][j+k] for k in range(4)]
                if sub_diagonal1.count(letter) == 4:
                    score += 1000
                elif sub_diagonal1.count(letter) == 3 and sub_diagonal1.count(0) == 1:
                    score += 100
                elif sub_diagonal1.count(letter) == 2 and sub_diagonal1.count(0) == 2:
                    score += 10
                if sub_diagonal1.count(OtherPlayerLetter) == 3 and sub_diagonal1.count(0) == 1:
                    score -= 10
                if sub_diagonal2.count(letter) == 4:
                    score += 1000
                elif sub_diagonal2.count(letter) == 3 and sub_diagonal2.count(0) == 1:
                    score += 100
                elif sub_diagonal2.count(letter) == 2 and sub_diagonal2.count(0) == 2:
                    score += 10
                if sub_diagonal2.count(OtherPlayerLetter) == 3 and sub_diagonal2.count(0) == 1:
                    score -= 10

        return score
    
    def Min_Max_Move_with_alpha_beta_pruning_and_depth(self, c4Game, connect4_board, current_depth, isMinMaxMove, MinMaxLetter, SIAgentLetter, alpha, beta):

        if c4Game.validateFinalMove(SIAgentLetter, MinMaxLetter):

            if c4Game.validateWin(MinMaxLetter) :
                return (None, 10000000)

            elif c4Game.validateWin(SIAgentLetter) :
                return (None, -10000000)

            else:
                return (None, 0)

        if current_depth == 0 :     
            return (None, self.evaluate_MinMax_score(c4Game, MinMaxLetter, SIAgentLetter, MinMaxLetter))

        possible_positions = c4Game.getValidMove()

        if isMinMaxMove:
            optimisedScore = -math.inf
            optimisedPosition = random.choice(possible_positions)

            for position in possible_positions:
                random_row = c4Game.getNextAvailableRow(position)
                connect4_board = c4Game.connect4_board.copy()
                connect4_board[random_row][position] = MinMaxLetter
                current_minmax_score = self.Min_Max_Move_with_alpha_beta_pruning_and_depth(c4Game, connect4_board, current_depth - 1, False, MinMaxLetter, SIAgentLetter, alpha, beta)[1]

                if current_minmax_score > optimisedScore:
                    optimisedScore = current_minmax_score
                    optimisedPosition = position

                alpha = max(optimisedScore, alpha)

                if alpha >= beta:
                    break

            return optimisedPosition, optimisedScore

        else:
            optimisedScore = math.inf
            optimisedPosition = random.choice(possible_positions)

            for position in possible_positions:
                random_row = c4Game.getNextAvailableRow(position)
                connect4_board = c4Game.connect4_board.copy()
                connect4_board[random_row][position] = MinMaxLetter
                current_minmax_score = self.Min_Max_Move_with_alpha_beta_pruning_and_depth(c4Game, connect4_board, current_depth - 1, True, MinMaxLetter, SIAgentLetter, alpha, beta)[1]

                if current_minmax_score < optimisedScore:
                    optimisedScore = current_minmax_score
                    optimisedPosition = position

                beta = min(beta, optimisedScore)

                if alpha >= beta:
                    break

        return optimisedPosition, optimisedScore

def play_connect4(MinMaxPlaysFirst, qLearningPlayer, minmaxPlayer, c4Game):
        
        QLearningLetter = 1
        MinMaxLetter = 2
        
        while True:
            if MinMaxPlaysFirst:
                
                MinMaxPossible_Positions = c4Game.getValidMove()
               
                if len(MinMaxPossible_Positions) == 0:
                    return "Draw"
                
                minmax_chosen_column, _ = minmaxPlayer.Min_Max_Move_with_alpha_beta_pruning_and_depth(c4Game, c4Game.connect4_board, 
                                        8, True, MinMaxLetter, QLearningLetter, -math.inf, math.inf)
                
                minmax_chosen_row = c4Game.getNextAvailableRow(minmax_chosen_column)
                c4Game.connect4_board[minmax_chosen_row][minmax_chosen_column] = MinMaxLetter
                
                if c4Game.validateWin(MinMaxLetter) : 
                    return "MinMaxWon"

                if c4Game.validateWin(QLearningLetter):
                    return "QLearningWon"

                if len(c4Game.getValidMove()) == 0 :
                    return "Draw"

                QLearningPossible_Positions = c4Game.getValidMove()
                
                if len(QLearningPossible_Positions) == 0:
                    return "Draw"

                QLearning_chosen_column = qLearningPlayer.getBestPositionFromQLearning(c4Game.connect4_board, QLearningPossible_Positions)
                QLearning_chosen_row = c4Game.getNextAvailableRow(QLearning_chosen_column)
                c4Game.connect4_board[QLearning_chosen_row][QLearning_chosen_column] = QLearningLetter

                if c4Game.validateWin(QLearningLetter):
                    return "QLearningWon"
                
                if c4Game.validateWin(MinMaxLetter) : 
                    return "MinMaxWon"
                
                if len(c4Game.getValidMove()) == 0 :
                    return "Draw"

            else:
                QLearningPossible_Positions = c4Game.getValidMove()
                
                if len(QLearningPossible_Positions) == 0:
                    return "Draw"

                QLearning_chosen_column = qLearningPlayer.getBestPositionFromQLearning(c4Game.connect4_board, QLearningPossible_Positions)
                QLearning_chosen_row = c4Game.getNextAvailableRow(QLearning_chosen_column)
                c4Game.connect4_board[QLearning_chosen_row][QLearning_chosen_column] = QLearningLetter

                if c4Game.validateWin(QLearningLetter):
                    return "QLearningWon"
                
                if c4Game.validateWin(MinMaxLetter) : 
                    return "MinMaxWon"
                
                if len(c4Game.getValidMove()) == 0 :
                    return "Draw"


                MinMaxPossible_Positions = c4Game.getValidMove()
               
                if len(MinMaxPossible_Positions) == 0:
                    return "Draw"
                
                minmax_chosen_column, _ = minmaxPlayer.Min_Max_Move_with_alpha_beta_pruning_and_depth(c4Game, c4Game.connect4_board, 
                                        8, True, MinMaxLetter, QLearningLetter, -math.inf, math.inf)
                
                minmax_chosen_row = c4Game.getNextAvailableRow(minmax_chosen_column)
                c4Game.connect4_board[minmax_chosen_row][minmax_chosen_column] = MinMaxLetter
                
                if c4Game.validateWin(MinMaxLetter) : 
                    return "MinMaxWon"

                if c4Game.validateWin(QLearningLetter):
                    return "QLearningWon"

                if len(c4Game.getValidMove()) == 0 :
                    return "Draw"
                

# Playing the Connect4 with first move as Random

games = 100
MinMaxWin = QLearningWin = Draw = 0

qLearningPlayer = QLearning() 
qLearningPlayer.loadQLearningModel()

minmaxPlayer = MinMax()

print(f"Current Q Learning model has {len(qLearningPlayer.QLearningStates)} states")

for _ in tqdm(range(games)):
    c4Game = Connect4_Game()
    c4Game.initialise_board()
    
    MinMaxPlaysFirst = False
    if c4Game.tossForFirstMove() == 1 :
        MinMaxPlaysFirst = True
    else : 
        MinMaxPlaysFirst = False
    
    winner = play_connect4(MinMaxPlaysFirst, qLearningPlayer, minmaxPlayer, c4Game)
    
    if winner == 'QLearningWon':
        QLearningWin += 1
    elif winner == 'MinMaxWon':
        MinMaxWin += 1
    else:
        Draw += 1

statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Number of Games QLearning Won', 'Number of Games MinMax Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Random '
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games QLearning Won'] = QLearningWin
QLearningWin_FM_Rand = QLearningWin
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_FM_Rand = MinMaxWin
statistics_dict['Number of Games Drawn'] = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# Playing the Connect4 with first move as always Q-Learning

games = 100
MinMaxWin = QLearningWin = Draw = 0

qLearningPlayer = QLearning() 
qLearningPlayer.loadQLearningModel()

minmaxPlayer = MinMax()


print(f"Current Q Learning model has {len(qLearningPlayer.QLearningStates)} states")

for _ in tqdm(range(games)):
    c4Game = Connect4_Game()
    c4Game.initialise_board()
    
    MinMaxPlaysFirst = False
    
    winner = play_connect4(MinMaxPlaysFirst, qLearningPlayer, minmaxPlayer, c4Game)
    
    if winner == 'QLearningWon':
        QLearningWin += 1
    elif winner == 'MinMaxWon':
        MinMaxWin += 1
    else:
        Draw += 1


statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Number of Games QLearning Won', 'Number of Games MinMax Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Q-Learning Player'
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games QLearning Won'] = QLearningWin
QLearningWin_FM_Ql = QLearningWin
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_FM_Ql = MinMaxWin
statistics_dict['Number of Games Drawn'] = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# Playing the Connect4 with first move as always Min-Max

games = 100
MinMaxWin = QLearningWin = Draw = 0

qLearningPlayer = QLearning() 
qLearningPlayer.loadQLearningModel()

minmaxPlayer = MinMax()


print(f"Current Min-Max Learning model has {len(qLearningPlayer.QLearningStates)} states")

for _ in tqdm(range(games)):
    c4Game = Connect4_Game()
    c4Game.initialise_board()
    
    MinMaxPlaysFirst = True
    
    winner = play_connect4(MinMaxPlaysFirst, qLearningPlayer, minmaxPlayer, c4Game)
    
    if winner == 'QLearningWon':
        QLearningWin += 1
    elif winner == 'MinMaxWon':
        MinMaxWin += 1
    else:
        Draw += 1


statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Number of Games QLearning Won', 'Number of Games MinMax Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Min-Max Player'
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games QLearning Won'] = QLearningWin
QLearningWin_FM_MM = QLearningWin
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_FM_MM = MinMaxWin
statistics_dict['Number of Games Drawn'] = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# Plot the result

game_types = ['Random', 'Q-Learning First Move', 'Min-Max First Move']
QLearning_wins = [QLearningWin_FM_Rand, QLearningWin_FM_Ql, QLearningWin_FM_MM]
MinMax_wins = [MinMaxWin_FM_Rand, MinMaxWin_FM_Ql, MinMaxWin_FM_MM] 

# Plotting the bar graph
bar_width = 0.35
index = range(len(game_types))
plt.bar(index, QLearning_wins, bar_width, label='Q-Learning')
plt.bar([i + bar_width for i in index], MinMax_wins, bar_width, label='MinMax')
plt.xlabel('Game Type')
plt.ylabel('Number of Games Won')
plt.title('Connect4: Q-Learning vs MinMax')
plt.xticks([i + bar_width / 2 for i in index], game_types)
plt.legend()
plt.tight_layout()
plt.show()