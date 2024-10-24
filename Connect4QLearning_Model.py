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

class SI_Agent : 
    
    def Semi_Intelligent_Agent_Move(self, c4_game, SIAgentLetter, MinMaxLetter) : 
        if c4_game.validateFinalMove(SIAgentLetter, MinMaxLetter):
            siagent_row, siagent_col = c4_game.getNextAvailablePosotion(SIAgentLetter)
            if siagent_row != -1:
                return siagent_row, siagent_col
            else:
                minmax_row, minmax_col = c4_game.getNextAvailablePosotion(MinMaxLetter)
                if minmax_row != -1:
                    return minmax_row, minmax_col
                else:
                    possible_positions = c4_game.getValidMove()
                    random_row = c4_game.getNextAvailableRow(random.choice(possible_positions))
                    random_col = random.choice(possible_positions)
                    return random_row, random_col
        else:
            possible_positions = c4_game.getValidMove()
            random_row = c4_game.getNextAvailableRow(random.choice(possible_positions)) 
            random_col = random.choice(possible_positions)

            return random_row, random_col

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
    
    def updateQLearningModel(self, current_board, current_position, reward, successive_board, possible_positions):
        bestQValue = max([self.getQLearningValue_For_Action(successive_board, next_position) for next_position in possible_positions], default=0)
        optimisedQValue = self.getQLearningValue_For_Action(current_board, current_position) + 0.1 * ((reward + 0.99 * bestQValue) - self.getQLearningValue_For_Action(current_board, current_position))
        position = self.getPosition(current_board)
        self.QLearningStates[(position, current_position)] = optimisedQValue
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * 0.999, 0.1)
        
    def saveQLearningModel(self):
        with open("Connect4QLearningTraningModel.pickle", "wb") as file:
            pickle.dump(self.QLearningStates, file)
            
    def loadQLearningModel(self):
        with open("Connect4QLearningTraningModel.pickle", "rb") as file:
            self.QLearningStates = pickle.load(file)
            
    def trainQLearningModel(self):
        QLearningWin = SIAgentWin = Draw = 0
        QLearningLetter = 1
        SIAgentLetter = 2
        total_episodes = 3000000
        si_agent = SI_Agent()
        
        for episode in tqdm(range(total_episodes)):
            c4Game = Connect4_Game()
            c4Game.initialise_board()
            current_board = c4Game.connect4_board
            
            while True:
            
                QLearningPossible_Positions = c4Game.getValidMove()
                
                if len(QLearningPossible_Positions) == 0:
                    break
                
                QLearning_chosen_column = self.getBestPositionFromQLearning(current_board, QLearningPossible_Positions)
                QLearning_chosen_row = c4Game.getNextAvailableRow(QLearning_chosen_column)
                c4Game.connect4_board[QLearning_chosen_row][QLearning_chosen_column] = QLearningLetter
                
                possibleMoves = c4Game.getValidMove()
                
                if c4Game.validateWin(QLearningLetter):
                    QLearningWin += 1
                    self.updateQLearningModel(current_board, QLearning_chosen_column, 1, c4Game.connect4_board, [])
                    break
                    
                elif c4Game.validateWin(SIAgentLetter):
                    SIAgentWin += 1
                    self.updateQLearningModel(current_board, QLearning_chosen_column, -1, c4Game.connect4_board, [])
                    break

                elif len(possibleMoves) == 0:
                    Draw += 1
                    self.updateQLearningModel(current_board, QLearning_chosen_column, 0, c4Game.connect4_board, [])
                    break
                
                else:
                    self.updateQLearningModel(current_board, QLearning_chosen_column, 0, c4Game.connect4_board, possibleMoves)
                    
                    
                SIAgent_chosen_row, SIAgent_chosen_column = si_agent.Semi_Intelligent_Agent_Move(c4Game, SIAgentLetter, QLearningLetter)
                c4Game.connect4_board[SIAgent_chosen_row][SIAgent_chosen_column] = SIAgentLetter
                
                possibleMoves = c4Game.getValidMove()
                
                if c4Game.validateWin(QLearningLetter):
                    QLearningWin += 1
                    self.updateQLearningModel(current_board, SIAgent_chosen_column, 1, c4Game.connect4_board, [])
                    break
                    
                elif c4Game.validateWin(SIAgentLetter):
                    SIAgentWin += 1
                    self.updateQLearningModel(current_board, SIAgent_chosen_column, -1, c4Game.connect4_board, [])
                    break

                elif len(possibleMoves) == 0:
                    Draw += 1
                    self.updateQLearningModel(current_board, SIAgent_chosen_column, 0, c4Game.connect4_board, [])
                    break
                
                else:
                    self.updateQLearningModel(current_board, SIAgent_chosen_column, 0, c4Game.connect4_board, possibleMoves)
                
                current_board = c4Game.connect4_board
            self.update_epsilon()
                    
        return QLearningWin, SIAgentWin, Draw, total_episodes


# Train the Model

qLearning = QLearning()
QLearningWin, SIAgentWin, Draw, total_episodes = qLearning.trainQLearningModel()
qLearning.saveQLearningModel()


statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Number of Games Qlearning Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'Training'
statistics_dict['Total Number of Games'] = total_episodes
statistics_dict['Number of Games Qlearning Won'] = QLearningWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw

statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# Display the plot for training
plt.figure(figsize=(8, 6))
plt.bar(['QLearning', 'Semi-Intelligent Player', 'Draw'], [QLearningWin, SIAgentWin, Draw], color=['blue', 'orange', 'green'])
plt.xlabel('Outcome')
plt.ylabel('Number of Games')
plt.title('Performance of Q-Learning vs Semi-Intelligent Player (Training for 3,000,000 episodes)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


def play_connect4(SIAgent_plays_first, c4Game, si_agent, qLearningPlayer):
    QLearningLetter = 1
    SIAgentLetter = 2

    while True:
        if SIAgent_plays_first:
            
            SIAgentPossible_Positions = c4Game.getValidMove()

            if len(SIAgentPossible_Positions) == 0:
                return "Draw"

            SIAgent_chosen_row, SIAgent_chosen_column = si_agent.Semi_Intelligent_Agent_Move(c4Game, SIAgentLetter, QLearningLetter)
            c4Game.connect4_board[SIAgent_chosen_row][SIAgent_chosen_column] = SIAgentLetter
            
            if c4Game.validateWin(SIAgentLetter) : 
                return "SIAgentWon"

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
            
            if c4Game.validateWin(SIAgentLetter) : 
                return "SIAgentWon"

            if c4Game.validateWin(QLearningLetter):
                return "QLearningWon"

            if len(c4Game.getValidMove()) == 0 :
                return "Draw"
            
        else:
            QLearningPossible_Positions = c4Game.getValidMove()
                
            if len(QLearningPossible_Positions) == 0:
                return "Draw"
                
            QLearning_chosen_column = qLearningPlayer.getBestPositionFromQLearning(c4Game.connect4_board, QLearningPossible_Positions)
            QLearning_chosen_row = c4Game.getNextAvailableRow(QLearning_chosen_column)
            c4Game.connect4_board[QLearning_chosen_row][QLearning_chosen_column] = QLearningLetter
            
            if c4Game.validateWin(SIAgentLetter) : 
                return "SIAgentWon"

            if c4Game.validateWin(QLearningLetter):
                return "QLearningWon"

            if len(c4Game.getValidMove()) == 0 :
                return "Draw"


            SIAgentPossible_Positions = c4Game.getValidMove()

            if len(SIAgentPossible_Positions) == 0:
                return "Draw"

            SIAgent_chosen_row, SIAgent_chosen_column = si_agent.Semi_Intelligent_Agent_Move(c4Game, SIAgentLetter, QLearningLetter)
            c4Game.connect4_board[SIAgent_chosen_row][SIAgent_chosen_column] = SIAgentLetter
            
            if c4Game.validateWin(SIAgentLetter) : 
                return "SIAgentWon"

            if c4Game.validateWin(QLearningLetter):
                return "QLearningWon"

            if len(c4Game.getValidMove()) == 0 :
                return "Draw"


# First Move: Random

games = 2000
SIAgentWin = QLearningWin = Draw = 0

qLearningPlayer = QLearning() 
qLearningPlayer.loadQLearningModel()

si_agent = SI_Agent()


print(f"Current Q Learning model has {len(qLearningPlayer.QLearningStates)} states")

for _ in tqdm(range(games)):
    c4Game = Connect4_Game()
    c4Game.initialise_board()
    
    SIAgent_plays_first = False
    if c4Game.tossForFirstMove() == 1 :
        SIAgent_plays_first = True
    else : 
        SIAgent_plays_first = False
    
    winner = play_connect4(SIAgent_plays_first, c4Game, si_agent, qLearningPlayer)

    if winner == 'QLearningWon':
        QLearningWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1

statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Number of Games QLearning Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Random'
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games QLearning Won'] = QLearningWin
QLearningWin_FM_Rand = QLearningWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_FM_Rand = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_FM_Rand = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# First Move: Always Semi-Intelligent Agent

games = 2000
SIAgentWin = QLearningWin = Draw = 0

qLearningPlayer = QLearning() 
qLearningPlayer.loadQLearningModel()

si_agent = SI_Agent()


print(f"Current Q Learning model has {len(qLearningPlayer.QLearningStates)} states")

for _ in tqdm(range(games)):
    c4Game = Connect4_Game()
    c4Game.initialise_board()
    
    SIAgent_plays_first = True
    
    winner = play_connect4(SIAgent_plays_first, c4Game, si_agent, qLearningPlayer)

    if winner == 'QLearningWon':
        QLearningWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1

statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Number of Games QLearning Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: First Move: Semi Intelligent Player'
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games QLearning Won'] = QLearningWin
QLearningWin_FM_SI = QLearningWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_FM_SI = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_FM_SI = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# First Move: Always Q-Learning player

games = 2000
SIAgentWin = QLearningWin = Draw = 0

qLearningPlayer = QLearning() 
qLearningPlayer.loadQLearningModel()

si_agent = SI_Agent()


print(f"Current Q Learning model has {len(qLearningPlayer.QLearningStates)} states")

for _ in tqdm(range(games)):
    c4Game = Connect4_Game()
    c4Game.initialise_board()
    
    SIAgent_plays_first = False
    
    winner = play_connect4(SIAgent_plays_first, c4Game, si_agent, qLearningPlayer)

    if winner == 'QLearningWon':
        QLearningWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1

statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Number of Games QLearning Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Q-Learning Player'
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games QLearning Won'] = QLearningWin
QLearningWin_FM_Ql = QLearningWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_FM_Ql = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_FM_Ql = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)

# Plot the results

moves_made_first = ['Random', 'Q-Learning', 'Semi-Intelligent']
QLearningWins = [QLearningWin_FM_Rand, QLearningWin_FM_Ql, QLearningWin_FM_SI]  # Number of games Q-learning won for each configuration
SIAgentWins = [SIAgentWin_FM_Rand, SIAgentWin_FM_Ql, SIAgentWin_FM_SI]      # Number of games semi-intelligent player won for each configuration
Draws = [Draw_FM_Rand, Draw_FM_Ql, Draw_FM_SI]                # Number of drawn games for each configuration

# Plotting the grouped bar graph
bar_width = 0.25
index = range(len(moves_made_first))
plt.figure(figsize=(10, 6))
plt.bar(index, QLearningWins, bar_width, label='Q-Learning', color='blue')
plt.bar([i + bar_width for i in index], SIAgentWins, bar_width, label='Semi-Intelligent', color='orange')
plt.bar([i + 2 * bar_width for i in index], Draws, bar_width, label='Draw', color='green')
plt.xlabel('Starting Move Configuration')
plt.ylabel('Number of Games')
plt.title('Q-Learning Vs Semi-Intelligent player for Connect4 game')
plt.xticks([i + bar_width for i in index], moves_made_first)
plt.legend()
plt.tight_layout()
plt.show()