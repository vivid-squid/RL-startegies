#!/usr/bin/env python
# coding: utf-8

# TicTacToe QLearning

import random
import math
from IPython.display import display
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
#!pip install Jinja2

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

    def updateQLearningModel(self, current_board, current_position, reward, successive_board, possible_positions):
        bestQValue = max([self.getQLearningValue_For_Action(successive_board, current_position) for next_action in possible_positions], default=0)
        optimisedQVlaue = self.getQLearningValue_For_Action(current_board, current_position) + 0.1 * ((reward + 0.99 * bestQValue) - self.getQLearningValue_For_Action(current_board, current_position))
        position = self.getPosition(current_board)
        self.QLearningStates[position][current_position - 1] = optimisedQVlaue

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * 0.999, 0.1)
        
    def saveQLearningModel(self):
        with open("TicTacToeQLearningTraningModel.pickle", "wb") as file:
            pickle.dump(self.QLearningStates, file)
            
    def loadQLearningModel(self):
        with open("TicTacToeQLearningTraningModel.pickle", "rb") as file:
            self.QLearningStates = pickle.load(file)
            
    def trainQLearningModel(self):
        QLearningWin = SIAgentWin = Draw = 0
        total_episodes = 3000000
        for episode in tqdm(range(total_episodes)):
            ttt_game = TicTacToe_Game()
            ttt_game.initialise_baord()
            current_board = ttt_game.ttt_board

            while True:
                QLearningPossible_Positions = [i for i in range(1, 10) if ttt_game.validateMove(i)]

                if len(QLearningPossible_Positions) == 0:
                    break

                QLearningPosition = self.getBestPositionFromQLearning(current_board, QLearningPossible_Positions)

                if ttt_game.validateMove(QLearningPosition):
                    ttt_game.ttt_board[QLearningPosition] = 'X'

                isQLearningWinner = ttt_game.validateWinForLetter('X')
                isSIAgentWinner = ttt_game.validateWinForLetter('O')
                possibleMoves = [i for i in range(1, 10) if ttt_game.validateMove(i)]

                if isQLearningWinner:
                    QLearningWin += 1
                    self.updateQLearningModel(current_board, QLearningPosition, 1, ttt_game.ttt_board, [])
                    break

                elif isSIAgentWinner:
                    SIAgentWin += 1
                    self.updateQLearningModel(current_board, QLearningPosition, -1, ttt_game.ttt_board, [])
                    break

                elif ttt_game.validateDraw():
                    Draw += 1
                    self.updateQLearningModel(current_board, QLearningPosition, 0, ttt_game.ttt_board, [])
                    break
                else:
                    self.updateQLearningModel(current_board, QLearningPosition, 0, ttt_game.ttt_board, possibleMoves)

                SIAgentPossible_Positions = [i for i in range(1, 10) if ttt_game.validateMove(i)]
                SIAgentPosition = SIAgentPossible_Positions[random.randint(0, len(SIAgentPossible_Positions)-1)]

                if ttt_game.validateMove(SIAgentPosition):
                    ttt_game.ttt_board[SIAgentPosition] = 'O'

                isQLearningWinner = ttt_game.validateWinForLetter('X')
                isSIAgentWinner = ttt_game.validateWinForLetter('O')
                possibleMoves = [i for i in range(1, 10) if ttt_game.validateMove(i)]

                if isQLearningWinner:
                    QLearningWin += 1
                    self.updateQLearningModel(current_board, SIAgentPosition, 1, ttt_game.ttt_board, [])
                    break

                elif isSIAgentWinner:
                    SIAgentWin += 1
                    self.updateQLearningModel(current_board, SIAgentPosition, -1, ttt_game.ttt_board, [])
                    break

                elif ttt_game.validateDraw():
                    Draw += 1
                    self.updateQLearningModel(current_board, SIAgentPosition, 0, ttt_game.ttt_board, [])
                    break
                else:
                    self.updateQLearningModel(current_board, SIAgentPosition, 0, ttt_game.ttt_board, possibleMoves)

                current_board = ttt_game.ttt_board
            self.update_epsilon()

        return QLearningWin, SIAgentWin, Draw, total_episodes

    def play_tic_tac_toe(self, SIAgent_plays_first, ttt_game):
        SI_Agent_Letter = 'O'
        QLearning_Letter = 'X'

        while True:
            if SIAgent_plays_first:
                SIAgentPossible_Positions = [i for i in range(1, 10) if ttt_game.validateMove(i)]

                if len(SIAgentPossible_Positions) == 0:
                    return "Draw"

                SIAgentPosition = SIAgentPossible_Positions[random.randint(0, len(SIAgentPossible_Positions)-1)]

                if ttt_game.validateMove(SIAgentPosition):
                    ttt_game.ttt_board[SIAgentPosition] = SI_Agent_Letter

                if ttt_game.validateWinForLetter(SI_Agent_Letter) : 
                    return "SIAgentWon"

                if ttt_game.validateDraw():
                    return "Draw"

                QLearningPossible_Positions = [i for i in range(1, 10) if ttt_game.validateMove(i)]

                if len(QLearningPossible_Positions) == 0:
                    "Draw"

                QLearningPosition = self.getBestPositionFromQLearning(ttt_game.ttt_board, QLearningPossible_Positions)

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

                QLearningPosition = self.getBestPositionFromQLearning(ttt_game.ttt_board, QLearningPossible_Positions)

                if ttt_game.validateMove(QLearningPosition):
                    ttt_game.ttt_board[QLearningPosition] = QLearning_Letter

                if ttt_game.validateWinForLetter(QLearning_Letter) : 
                    return "QLearningWon"

                if ttt_game.validateDraw():
                    return "Draw"


                SIAgentPossible_Positions = [i for i in range(1, 10) if ttt_game.validateMove(i)]

                if len(SIAgentPossible_Positions) == 0:
                    return "Draw"

                SIAgentPosition = SIAgentPossible_Positions[random.randint(0, len(SIAgentPossible_Positions)-1)]

                if ttt_game.validateMove(SIAgentPosition):
                    ttt_game.ttt_board[SIAgentPosition] = SI_Agent_Letter

                if ttt_game.validateWinForLetter(SI_Agent_Letter) : 
                    return "SIAgentWon"

                if ttt_game.validateDraw():
                    return "Draw"


# Tranning the Model Based on 300000 episodes


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

# Display the result

def add_values(bar):
    for rect in bar:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.figure(figsize=(10, 6))
fig, ax = plt.subplots(figsize=(10, 6))
ind = np.arange(1)
width = 0.2  
p1 = ax.bar(ind, QLearningWin, width, color='blue', label='Q-Learning Wins')
p2 = ax.bar(ind + width, SIAgentWin, width, color='orange', label='Semi-Intelligent Wins')
p3 = ax.bar(ind + width * 2, Draw, width, color='grey', label='Draws')
ax.set_xlabel('Game Type')
ax.set_ylabel('Number of Games')
ax.set_title('Q-Learning Vs Semi-Intelligent player for Training 3,000,000 episodes for Tic-Tac-Toe game')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Training',))
ax.legend()
add_values(p1)
add_values(p2)
add_values(p3)
plt.show()

# First Move for Semi Intelliigent Agent 

games = 20000
SIAgentWin = QLearningWin = Draw = 0

qLearningPlayer = QLearning() 
qLearningPlayer.loadQLearningModel()

print(f"Current Q Learning model has {len(qLearningPlayer.QLearningStates)} states")

for _ in tqdm(range(games)):
    ttt_game = TicTacToe_Game()
    ttt_game.initialise_baord()
    
    SIAgent_plays_first = True

    winner = qLearningPlayer.play_tic_tac_toe(SIAgent_plays_first, ttt_game)

    if winner == 'QLearningWon':
        QLearningWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1

statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Number of Games QLearning Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Semi Intelligent Player'
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games QLearning Won'] = QLearningWin
QLearningWin_SI = QLearningWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_SI = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_SI = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# First Move for Q-Learning Agent

games = 20000
SIAgentWin = QLearningWin = Draw = 0

qLearningPlayer = QLearning() 
qLearningPlayer.loadQLearningModel()

print(f"Current Q Learning model has {len(qLearningPlayer.QLearningStates)} states")

for _ in tqdm(range(games)):
    ttt_game = TicTacToe_Game()
    ttt_game.initialise_baord()
    
    SIAgent_plays_first = False

    winner = qLearningPlayer.play_tic_tac_toe(SIAgent_plays_first, ttt_game)

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
QLearningWin_Ql = QLearningWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_Ql = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_Ql = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


#  The Graphs comparing Training performance of Q-Learning vs. Semi-Intelligent player

def add_labels(data, shift):
    for i in range(len(data)):
        plt.text(i + shift, data[i], str(data[i]), ha = 'center', va = 'bottom')

labels = ['QLearning First move', 'Semi-Intelligent Agent First move']
QLearning_wins = [QLearningWin_Ql, SIAgentWin_Ql]  # QLearning wins when QLearning and SI agent move first respectively
SIAgent_wins = [QLearningWin_SI, SIAgentWin_SI]    # SIAgent wins when QLearning and SI agent move first respectively
draws = [Draw_Ql, Draw_SI]            # Draws when QLearning and SI agent move first respectively

x = np.arange(len(labels)) 
width = 0.25 

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(x - width, QLearning_wins, width, label='QLearning Wins', color='blue')
plt.bar(x, SIAgent_wins, width, label='SIAgent Wins', color='orange')
plt.bar(x + width, draws, width, label='Draws', color='green')
plt.xlabel('Game Type')
plt.ylabel('Number of Games Won')
plt.title('Comparison of Model Testing Performance Including Draws')
plt.xticks(x, labels)
plt.legend()

add_labels(QLearning_wins, -width)
add_labels(SIAgent_wins, 0)
add_labels(draws, width)

plt.tight_layout()
plt.show()

# Plot for the Testing part of the model using bar graph

players = ['QLearning', 'Semi-Intelligent']
games_won = [QLearningWin, SIAgentWin]
games_drawn = [Draw, Draw]

# Plotting the bar graph
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(players))
bar1 = plt.bar(index, games_drawn, bar_width, label='Games Drawn')
bar2 = plt.bar(index, games_won, bar_width, label='Games Won', bottom=games_drawn)
plt.xlabel('Player')
plt.ylabel('Number of Games')
plt.title('Performance Comparison: Q-Learning vs Semi-Intelligent Player')
plt.xticks(index, players)
plt.legend()

# Adding labels to the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.annotate('{}'.format(height),
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), 
                     textcoords="offset points",
                     ha='center', va='bottom')

add_labels(bar1)
add_labels(bar2)
plt.tight_layout()
plt.show()