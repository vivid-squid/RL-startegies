#!/usr/bin/env python
# coding: utf-8

import random
import math
from IPython.display import display
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt

class TicTacToe_Game:
    
    def initialise_baord(self) : 
        self.ttt_board = {
                            1: ' ', 2:' ', 3: ' ',
                            4: ' ', 5:' ', 6: ' ',
                            7: ' ', 8:' ', 9: ' '
                         }
    def display_board(self):
        print("\n")
        for row in range(3):
            for col in range(3):
                cell = row * 3 + col + 1
                print(self.ttt_board[cell], end="")
                if col < 2:
                    print(" | ", end="")
            print()
            if row < 2:
                print("---------")
        print()


    def tossForFirstMove(self):
        choices = [1,2]
        return random.choice(choices)
    
    def get_random_generated_move(self):
        position = random.randint(1, 9)
        if self.validateMove(position):
            return position
        else:
            position = self.get_random_generated_move()
            return position
        
    def display_board(self):
        print("\n")
        print( self.ttt_board[1], '|', self.ttt_board[2], '|', self.ttt_board[3])
        print(' -+---+-')
        print(self.ttt_board[4], '|', self.ttt_board[5], '|', self.ttt_board[6])
        print(' -+---+-')
        print(self.ttt_board[7], '|', self.ttt_board[8], '|', self.ttt_board[9], "\n")
        
    def validateMove(self, move):
        return self.ttt_board[move] == ' '
        
    def validateDraw(self):
        return all(self.ttt_board[key] != ' ' for key in self.ttt_board.keys())
    
    
    def validateWin(self):
        win_combinations = [
            (1, 2, 3), (4, 5, 6), (7, 8, 9), 
            (1, 4, 7), (2, 5, 8), (3, 6, 9),  
            (1, 5, 9), (7, 5, 3)
        ]

        for combo in win_combinations:
            if (self.ttt_board[combo[0]] == self.ttt_board[combo[1]] == self.ttt_board[combo[2]] != ' '):
                return True

        return False
                        
    def validateWinForLetter(self, mark):
        winning_positions = [
            (1, 2, 3), (4, 5, 6), (7, 8, 9),
            (1, 4, 7), (2, 5, 8), (3, 6, 9),
            (1, 5, 9), (7, 5, 3)
        ]
        for pos in winning_positions:
            if all(self.ttt_board[i] == mark for i in pos):
                return True
        return False

class QLearning:
    def __init__(self):
        self.epsilon = 1.0
        self.QLearningStates = {}
  
    getPosition = lambda self, current_board: tuple(tuple(current_board[i+j] for j in range(3)) for i in range(1, 10, 3))

    def getQLearningValue_For_Action(self, current_board, current_position):
        position = self.getPosition(current_board)
        if position not in self.QLearningStates:
            self.QLearningStates[position] = np.zeros((9,))
        return self.QLearningStates[position][current_position - 1]
  

    def getBestPositionFromQLearning(self, current_board, possible_positions):
        return random.choice(possible_positions) if random.random() < self.epsilon else max(possible_positions, key=lambda x: self.getQLearningValue_For_Action(current_board, x))
            
    def loadQLearningModel(self):
        with open("TicTacToeQLearningTraningModel.pickle", "rb") as file:
            self.QLearningStates = pickle.load(file)

class TicTacToe_MinMax:
    
    def Min_Max_Move_with_alpha_beta_pruning(self, ttt_game, MinMax_Letter, QLearing_Letter):
        optimised_score =  -math.inf
        optimised_position = ttt_game.get_random_generated_move()

        for possible_position in ttt_game.ttt_board.keys() : 

            if ttt_game.ttt_board[possible_position] == ' ' :
                ttt_game.ttt_board[possible_position] = MinMax_Letter
                current_score = self.evaluate_MinMax_score_with_alpha_beta_pruning(ttt_game, MinMax_Letter, QLearing_Letter, False, -math.inf, math.inf)
                ttt_game.ttt_board[possible_position] = ' '

                if current_score > optimised_score :
                    optimised_score = current_score
                    optimised_position = possible_position
        
        return optimised_position
    
    
    def evaluate_MinMax_score_with_alpha_beta_pruning(self, ttt_game, MinMax_Letter, QLearing_Letter, isMinMaxMove, alpha, beta):
        if ttt_game.validateWinForLetter(MinMax_Letter) :
            return 1
        elif ttt_game.validateWinForLetter(QLearing_Letter) :
            return -1
        elif ttt_game.validateDraw() :
            return 0

        if isMinMaxMove :
            optimisedScore = -math.inf

            for possible_position in ttt_game.ttt_board.keys():

                if ttt_game.ttt_board[possible_position] == ' ' :
                    ttt_game.ttt_board[possible_position] = MinMax_Letter
                    current_score = self.evaluate_MinMax_score_with_alpha_beta_pruning(ttt_game, MinMax_Letter, QLearing_Letter, False, alpha, beta)
                    ttt_game.ttt_board[possible_position] = ' '

                    optimisedScore = max(optimisedScore, current_score)
                    alpha = max(alpha, optimisedScore)

                    if alpha >= beta :
                        break

            return optimisedScore

        else:
            optimisedScore = math.inf
            
            for possible_position in ttt_game.ttt_board.keys():
                if ttt_game.ttt_board[possible_position] == ' ':
                    ttt_game.ttt_board[possible_position] = QLearing_Letter
                    current_score = self.evaluate_MinMax_score_with_alpha_beta_pruning(ttt_game, MinMax_Letter, QLearing_Letter, True, alpha, beta)
                    ttt_game.ttt_board[possible_position] = ' '

                    optimisedScore = min(optimisedScore, current_score)
                    beta = min(beta, optimisedScore)

                    if alpha >= beta :
                        break

            return optimisedScore


def play_tic_tac_toe(MinMaxPlaysFirst, QLearning, MinMax, ttt_game):
        MinMaxLetter = 'O'
        QLearning_Letter = 'X'

        while True:
            if MinMaxPlaysFirst:
                
                MinMaxPossible_Positions = [i for i in range(1, 10) if ttt_game.validateMove(i)]

                if len(MinMaxPossible_Positions) == 0:
                    return "Draw"
            
                MinMaxPosition = MinMax.Min_Max_Move_with_alpha_beta_pruning(ttt_game, MinMaxLetter, QLearning_Letter)
                
                if ttt_game.validateMove(MinMaxPosition):
                    ttt_game.ttt_board[MinMaxPosition] = MinMaxLetter
      
                if ttt_game.validateWinForLetter(MinMaxLetter) : 
                    return "MinMaxWon"

                if ttt_game.validateDraw():
                    return "Draw"

                QLearningPossible_Positions = [i for i in range(1, 10) if ttt_game.validateMove(i)]

                if len(QLearningPossible_Positions) == 0:
                    break

                QLearningPosition = QLearning.getBestPositionFromQLearning(ttt_game.ttt_board, QLearningPossible_Positions)

                if ttt_game.validateMove(QLearningPosition):
                    ttt_game.ttt_board[QLearningPosition] = QLearning_Letter

                if ttt_game.validateWinForLetter(QLearning_Letter) : 
                    return "QLearningWon"

                if ttt_game.validateDraw():
                    return "Draw"

            else:
                QLearningPossible_Positions = [i for i in range(1, 10) if ttt_game.validateMove(i)]

                if len(QLearningPossible_Positions) == 0:
                    break

                QLearningPosition = QLearning.getBestPositionFromQLearning(ttt_game.ttt_board, QLearningPossible_Positions)

                if ttt_game.validateMove(QLearningPosition):
                    ttt_game.ttt_board[QLearningPosition] = QLearning_Letter

                if ttt_game.validateWinForLetter(QLearning_Letter) : 
                    return "QLearningWon"

                if ttt_game.validateDraw():
                    return "Draw"


                MinMaxPossible_Positions = [i for i in range(1, 10) if ttt_game.validateMove(i)]

                if len(MinMaxPossible_Positions) == 0:
                    return "Draw"
            
                MinMaxPosition = MinMax.Min_Max_Move_with_alpha_beta_pruning(ttt_game, MinMaxLetter, QLearning_Letter)
                
                if ttt_game.validateMove(MinMaxPosition):
                    ttt_game.ttt_board[MinMaxPosition] = MinMaxLetter
      
                if ttt_game.validateWinForLetter(MinMaxLetter) : 
                    return "MinMaxWon"

                if ttt_game.validateDraw():
                    return "Draw"


# First Move: Random

games = 2000
MinMaxWin = QLearningWin = Draw = 0

qLearningPlayer = QLearning() 
qLearningPlayer.loadQLearningModel()

ttt_min_max = TicTacToe_MinMax()
print(f"Current Q Learning model has {len(qLearningPlayer.QLearningStates)} states")

for _ in tqdm(range(games)):
    ttt_game = TicTacToe_Game()
    ttt_game.initialise_baord()
    
    MinMaxPlaysFirst = False
    if ttt_game.tossForFirstMove() == 1 :
        MinMaxPlaysFirst = True
    else : 
        MinMaxPlaysFirst = False
    
    
    winner = play_tic_tac_toe(MinMaxPlaysFirst, qLearningPlayer, ttt_min_max, ttt_game)
    
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
Draw_FM_Rand = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# First Move: Always Q-Learning Player

games = 2000
MinMaxWin = QLearningWin = Draw = 0

qLearningPlayer = QLearning() 
qLearningPlayer.loadQLearningModel()

ttt_min_max = TicTacToe_MinMax()
print(f"Current Q Learning model has {len(qLearningPlayer.QLearningStates)} states")

for _ in tqdm(range(games)):
    ttt_game = TicTacToe_Game()
    ttt_game.initialise_baord()
    
    MinMaxPlaysFirst = False
    
    winner = play_tic_tac_toe(MinMaxPlaysFirst, qLearningPlayer, ttt_min_max, ttt_game)
    
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
Draw_FM_Ql = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# First Move: Always Min-Max Player

games = 2000
MinMaxWin = QLearningWin = Draw = 0

qLearningPlayer = QLearning() 
qLearningPlayer.loadQLearningModel()

ttt_min_max = TicTacToe_MinMax()
print(f"Current Q Learning model has {len(qLearningPlayer.QLearningStates)} states")

for _ in tqdm(range(games)):
    ttt_game = TicTacToe_Game()
    ttt_game.initialise_baord()
    
    MinMaxPlaysFirst = True
    
    winner = play_tic_tac_toe(MinMaxPlaysFirst, qLearningPlayer, ttt_min_max, ttt_game)

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
Draw_FM_MM = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)

# Plot for the final result

game_types = ['Random', 'Q-Learning First', 'MinMax First']
QLearning_wins = [QLearningWin_FM_Rand, QLearningWin_FM_Ql, QLearningWin_FM_MM] 
MinMax_wins = [MinMaxWin_FM_Rand, MinMaxWin_FM_Ql, MinMaxWin_FM_MM] 
draws = [Draw_FM_Rand, Draw_FM_Ql, Draw_FM_MM]  

positions = range(len(game_types))
plt.figure(figsize=(10, 6))
plt.bar(positions, QLearning_wins, width=0.2, label='QLearning Wins', align='center', color='blue')
plt.bar(positions, MinMax_wins, width=0.2, label='MinMax Wins', align='center', color='green', bottom=QLearning_wins)
plt.bar(positions, draws, width=0.2, label='Draws', align='center', color='red', bottom=[QLearning_wins[i] + MinMax_wins[i] for i in range(len(QLearning_wins))])
plt.xlabel('Game Type')
plt.ylabel('Number of Games')
plt.title('Q-Learning Vs MinMax player for Tic-Tac-Toe game')
plt.xticks(positions, game_types)
plt.legend()
plt.tight_layout()
plt.show()