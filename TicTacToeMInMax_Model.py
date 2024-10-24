#!/usr/bin/env python
# coding: utf-8

import random
import math
from IPython.display import display
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt

class TicTacToe_MinMax:
    
    def initialise_baord_set_letter(self) : 
        self.ttt_board = {
                            1: ' ', 2:' ', 3: ' ',
                            4: ' ', 5:' ', 6: ' ',
                            7: ' ', 8:' ', 9: ' '
                         }
        self.SI_Agent_Letter = 'X'
        self.MinMax_Letter = 'O'
    
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

    def get_random_generated_move(self):
        position = random.randint(1, 9)
        if self.validateMove(position):
            return position
        else:
            position = self.get_random_generated_move()
            return position
    
    def play_tic_tac_toe_with_alpha_beta_pruning(self, SIAgent_plays_first):
        while True:
            if SIAgent_plays_first:
                self.Semi_Intelligent_Agent_Move()

                if self.validateWinForLetter(self.SI_Agent_Letter) : 
                    return "SIAgentWon"

                if self.validateDraw():
                    return "Draw"

                self.Min_Max_Move_with_alpha_beta_pruning()

                if self.validateWinForLetter(self.MinMax_Letter) : 
                    return "MinMaxWon"


            else:

                self.Min_Max_Move_with_alpha_beta_pruning()

                if self.validateWinForLetter(self.MinMax_Letter) : 
                    return "MinMaxWon"

                self.Semi_Intelligent_Agent_Move()

                if self.validateWinForLetter(self.SI_Agent_Letter) :
                    return "SIAgentWon"
                
                if self.validateDraw():
                    return "Draw"



    def Semi_Intelligent_Agent_Move(self) : 
        for possible_position in self.ttt_board.keys():
            if self.ttt_board[possible_position] == ' ':
                
                self.ttt_board[possible_position] = self.SI_Agent_Letter
                
                if self.validateWin() : 
                    self.ttt_board[possible_position] = ' '
                    position = possible_position
                    break
                    
                elif self.validateDraw():
                    self.ttt_board[possible_position] = ' '
                    position = possible_position
                    break
                
                else:
                    self.ttt_board[possible_position] = ' '
                    position = self.get_random_generated_move()

        self.ttt_board[position] = self.SI_Agent_Letter
        return
        
    def Min_Max_Move_with_alpha_beta_pruning(self):
        optimised_score =  -math.inf
        optimised_position = self.get_random_generated_move()

        for possible_position in self.ttt_board.keys() : 

            if self.ttt_board[possible_position] == ' ' :
                self.ttt_board[possible_position] = self.MinMax_Letter
                current_score = self.evaluate_MinMax_score_with_alpha_beta_pruning(False, -math.inf, math.inf)
                self.ttt_board[possible_position] = ' '

                if current_score > optimised_score :
                    optimised_score = current_score
                    optimised_position = possible_position

        self.ttt_board[optimised_position] = self.MinMax_Letter
        return
    
    
    def evaluate_MinMax_score_with_alpha_beta_pruning(self, isMinMaxMove, alpha, beta):
        if self.validateWinForLetter(self.MinMax_Letter) :
            return 1
        elif self.validateWinForLetter(self.SI_Agent_Letter) :
            return -1
        elif self.validateDraw() :
            return 0

        if isMinMaxMove :
            optimisedScore = -math.inf

            for possible_position in self.ttt_board.keys():

                if self.ttt_board[possible_position] == ' ' :
                    self.ttt_board[possible_position] = self.MinMax_Letter
                    current_score = self.evaluate_MinMax_score_with_alpha_beta_pruning(False, alpha, beta)
                    self.ttt_board[possible_position] = ' '

                    optimisedScore = max(optimisedScore, current_score)
                    alpha = max(alpha, optimisedScore)

                    if alpha >= beta :
                        break

            return optimisedScore

        else:
            optimisedScore = math.inf
            for possible_position in self.ttt_board.keys():
                if self.ttt_board[possible_position] == ' ':
                    self.ttt_board[possible_position] = self.SI_Agent_Letter
                    current_score = self.evaluate_MinMax_score_with_alpha_beta_pruning(True, alpha, beta)
                    self.ttt_board[possible_position] = ' '

                    optimisedScore = min(optimisedScore, current_score)
                    beta = min(beta, optimisedScore)

                    if alpha >= beta :
                        break

            return optimisedScore
        
    def play_tic_tac_toe(self, SIAgent_plays_first):
        while True:
            if SIAgent_plays_first:
                self.Semi_Intelligent_Agent_Move()

                if self.validateWinForLetter(self.SI_Agent_Letter) : 
                    return "SIAgentWon"

                if self.validateDraw():
                    return "Draw"

                self.Min_Max_Move()

                if self.validateWinForLetter(self.MinMax_Letter) : 
                    return "MinMaxWon"


            else:

                self.Min_Max_Move()

                if self.validateWinForLetter(self.MinMax_Letter) : 
                    return "MinMaxWon"

                self.Semi_Intelligent_Agent_Move()

                if self.validateWinForLetter(self.SI_Agent_Letter) :
                    return "SIAgentWon"
                
                if self.validateDraw():
                    return "Draw"
                
    def Min_Max_Move(self):
        optimised_score =  -math.inf
        optimised_position = self.get_random_generated_move()

        for possible_position in self.ttt_board.keys() : 

            if self.ttt_board[possible_position] == ' ' :
                self.ttt_board[possible_position] = self.MinMax_Letter
                current_score = self.evaluate_MinMax_score(False)
                self.ttt_board[possible_position] = ' '

                if current_score > optimised_score :
                    optimised_score = current_score
                    optimised_position = possible_position

        self.ttt_board[optimised_position] = self.MinMax_Letter
        return
    
    def evaluate_MinMax_score(self, isMinMaxMove):
        if self.validateWinForLetter(self.MinMax_Letter) :
            return 1
        elif self.validateWinForLetter(self.SI_Agent_Letter) :
            return -1
        elif self.validateDraw() :
            return 0

        if isMinMaxMove :
            optimisedScore = -math.inf

            for possible_position in self.ttt_board.keys():

                if self.ttt_board[possible_position] == ' ' :
                    self.ttt_board[possible_position] = self.MinMax_Letter
                    current_score = self.evaluate_MinMax_score(False)
                    self.ttt_board[possible_position] = ' '

                    optimisedScore = max(optimisedScore, current_score)
                    
            return optimisedScore

        else:
            optimisedScore = math.inf
            for possible_position in self.ttt_board.keys():
                if self.ttt_board[possible_position] == ' ':
                    self.ttt_board[possible_position] = self.SI_Agent_Letter
                    current_score = self.evaluate_MinMax_score(True)
                    self.ttt_board[possible_position] = ' '

                    optimisedScore = min(optimisedScore, current_score)

            return optimisedScore
    


# First Move for Random, Algorithm: Min-Max without Alpha-Beta pruning

games = 100
SIAgentWin = MinMaxWin = Draw = 0

startTime = time.time()
for _ in tqdm(range(games)):
    ttt_min_max = TicTacToe_MinMax()
    ttt_min_max.initialise_baord_set_letter()
    
    SIAgent_plays_first = False
    if ttt_min_max.tossForFirstMove() == 1 :
        SIAgent_plays_first = True
    else : 
        SIAgent_plays_first = False
    try:
        winner = ttt_min_max.play_tic_tac_toe(SIAgent_plays_first)
    except:
        continue
        
    if winner == 'MinMaxWon':
        MinMaxWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1
totalTime = time.time()-startTime

# Display details

statistics_df = pd.DataFrame(columns=['Game Type', 'Time taken without Alpha Beta Pruning (in seconds)', 'Total Number of Games', 'Number of Games MinMax Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Random'
statistics_dict['Time taken without Alpha Beta Pruning (in seconds)'] = totalTime
totalTime_without_a_FM_Rand = totalTime
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_without_a_FM_Rand = MinMaxWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_without_a_FM_Rand = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_without_a_FM_Rand = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)

# First Move For Min-Max, Algorithm: Min-Max without Alpha-Beta pruning

games = 100
SIAgentWin = 0
MinMaxWin = 0
Draw = 0

startTime = time.time()
for _ in tqdm(range(games)):
    ttt_min_max = TicTacToe_MinMax()
    ttt_min_max.initialise_baord_set_letter()
    
    SIAgent_plays_first = False
    
    try:
        winner = ttt_min_max.play_tic_tac_toe(SIAgent_plays_first)
    except:
        continue
        
    if winner == 'MinMaxWon':
        MinMaxWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1

totalTime = time.time()-startTime

# Display details

statistics_df = pd.DataFrame(columns=['Game Type', 'Time taken without Alpha Beta Pruning (in seconds)', 'Total Number of Games', 'Number of Games MinMax Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: MinMax Player'
statistics_dict['Time taken without Alpha Beta Pruning (in seconds)'] = totalTime
totalTime_without_a_FM_MM = totalTime
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_without_a_FM_MM = MinMaxWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_without_a_FM_MM = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_without_a_FM_MM = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# First Move: Semi-Intelligent Player, Algorithm: Min-Max without Alpha-Beta pruning

games = 100
SIAgentWin = 0
MinMaxWin = 0
Draw = 0

startTime = time.time()
for _ in tqdm(range(games)):
    ttt_min_max = TicTacToe_MinMax()
    ttt_min_max.initialise_baord_set_letter()
    
    SIAgent_plays_first = True
    
    try:
        winner = ttt_min_max.play_tic_tac_toe(SIAgent_plays_first)
    except:
        continue
        
    if winner == 'MinMaxWon':
        MinMaxWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1

totalTime = time.time()-startTime

# Display details


statistics_df = pd.DataFrame(columns=['Game Type', 'Time taken without Alpha Beta Pruning (in seconds)', 'Total Number of Games', 'Number of Games MinMax Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Semi Intelligent Player'
statistics_dict['Time taken without Alpha Beta Pruning (in seconds)'] = totalTime
totalTime_without_a_FM_SI = totalTime
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_without_a_FM_SI = MinMaxWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_without_a_FM_SI = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_without_a_FM_SI = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# First Move: Random, Algorithm: Min-Max with Alpha-Beta pruning

games = 1000
SIAgentWin = MinMaxWin = Draw = 0

startTime = time.time()
for _ in tqdm(range(games)):
    ttt_min_max = TicTacToe_MinMax()
    ttt_min_max.initialise_baord_set_letter()
    
    SIAgent_plays_first = False
    if ttt_min_max.tossForFirstMove() == 1 :
        SIAgent_plays_first = True
    else : 
        SIAgent_plays_first = False
    
    try:
        winner = ttt_min_max.play_tic_tac_toe_with_alpha_beta_pruning(SIAgent_plays_first)
    except:
        continue
        
    if winner == 'MinMaxWon':
        MinMaxWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1
totalTime = time.time()-startTime

# Display details

statistics_df = pd.DataFrame(columns=['Game Type', 'Time taken with Alpha Beta Pruning (in seconds)', 'Total Number of Games', 'Number of Games MinMax Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Random Player'
statistics_dict['Time taken with Alpha Beta Pruning (in seconds)'] = totalTime
totalTime_with_a_FM_Rand = totalTime
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_with_a_FM_Rand = MinMaxWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_with_a_FM_Rand = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_with_a_FM_Rand = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# First Move: Min-Max, Algorithm: Min-Max with Alpha-Beta pruning

games = 1000
SIAgentWin = 0
MinMaxWin = 0
Draw = 0

startTime = time.time()
for _ in tqdm(range(games)):
    ttt_min_max = TicTacToe_MinMax()
    ttt_min_max.initialise_baord_set_letter()
    
    SIAgent_plays_first = False

    try:
        winner = ttt_min_max.play_tic_tac_toe_with_alpha_beta_pruning(SIAgent_plays_first)
    except:
        continue
        
    if winner == 'MinMaxWon':
        MinMaxWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1

totalTime = time.time()-startTime

# Display details

statistics_df = pd.DataFrame(columns=['Game Type', 'Time taken with Alpha Beta Pruning (in seconds)', 'Total Number of Games', 'Number of Games MinMax Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: MinMax Player'
statistics_dict['Time taken with Alpha Beta Pruning (in seconds)'] = totalTime
totalTime_with_a_FM_MM = totalTime
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_with_a_FM_MM = MinMaxWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_with_a_FM_MM = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_with_a_FM_MM = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# First Move: Semi-Intelligent Player, Algorithm: Min-Max with Alpha-Beta pruning

games = 1000
SIAgentWin = 0
MinMaxWin = 0
Draw = 0

startTime = time.time()
for _ in tqdm(range(games)):
    ttt_min_max = TicTacToe_MinMax()
    ttt_min_max.initialise_baord_set_letter()
    
    SIAgent_plays_first = True
    try:
        winner = ttt_min_max.play_tic_tac_toe_with_alpha_beta_pruning(SIAgent_plays_first)
    except:
        continue
        
    if winner == 'MinMaxWon':
        MinMaxWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1

totalTime = time.time()-startTime

# Display details

statistics_df = pd.DataFrame(columns=['Game Type', 'Time taken with Alpha Beta Pruning (in seconds)', 'Total Number of Games', 'Number of Games MinMax Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Semi Intelligent Player'
statistics_dict['Time taken with Alpha Beta Pruning (in seconds)'] = totalTime
totalTime_with_a_FM_SI = totalTime
statistics_dict['Total Number of Games'] = games
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_with_a_FM_SI = MinMaxWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_with_a_FM_SI = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_with_a_FM_SI = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)

# Plot for time comparison

game_types = ['Random', 'MinMax', 'Semi-Intelligent']
without_alpha_beta = [totalTime_without_a_FM_Rand, totalTime_without_a_FM_MM, totalTime_without_a_FM_SI]  # Time taken without Alpha Beta Pruning (in seconds)
with_alpha_beta = [totalTime_with_a_FM_Rand, totalTime_with_a_FM_MM, totalTime_with_a_FM_SI]  # Time taken with Alpha Beta Pruning (in seconds)

# Plotting the bar graph
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
opacity = 0.8
# Generating x values for the second set of bars
x = np.arange(len(game_types))
bar1 = plt.bar(x, without_alpha_beta, bar_width, alpha=opacity, color='b', label='Without Alpha-Beta Pruning')
bar2 = plt.bar(x + bar_width, with_alpha_beta, bar_width, alpha=opacity, color='r', label='With Alpha-Beta Pruning')
plt.xlabel('Game Type')
plt.ylabel('Time taken (seconds)')
plt.title('Comparison of Minimax Algorithm Implementations')
plt.xticks(x + bar_width/2, game_types)
plt.legend()
plt.tight_layout()
plt.show()

# Plot for comparison of Semi-intelligent player with Minmax(with alpha pruning)

MM_win_with_a = [MinMaxWin_with_a_FM_Rand, MinMaxWin_with_a_FM_MM, MinMaxWin_with_a_FM_SI]  
SI_win_with_a = [SIAgentWin_with_a_FM_Rand, SIAgentWin_with_a_FM_MM, SIAgentWin_with_a_FM_SI]  
Draw_with_a = [Draw_with_a_FM_Rand, Draw_with_a_FM_MM, Draw_with_a_FM_SI]

fig, ax = plt.subplots(figsize=(10, 6))
ind = np.arange(len(MM_win_with_a))
width = 0.25  
p1 = ax.bar(ind - width, MM_win_with_a, width, color='blue', label='Minmax Wins')
p2 = ax.bar(ind, SI_win_with_a, width, color='orange', label='Semi-Intelligent Wins')
p3 = ax.bar(ind + width, Draw_with_a, width, color='grey', label='Draws')
ax.set_xlabel('Game Type')
ax.set_ylabel('Number of Games')
ax.set_title('Minmax (with alpha pruning) Vs Semi-Intelligent player')
ax.set_xticks(ind)
ax.set_xticklabels(['Random First Move', 'Minmax First Move', 'Semi-Intelligent First Move'])
ax.legend()
def add_values(bar):
    for rect in bar:
        height = rect.get_height()
        if height != 0: 
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')
add_values(p1)
add_values(p2)
add_values(p3)
plt.show()

# Plot for comparison of Semi-intelligent player with Minmax(with alpha pruning)

MM_win_without_a = [MinMaxWin_without_a_FM_Rand, MinMaxWin_without_a_FM_MM, MinMaxWin_without_a_FM_SI]  
SI_win_without_a = [SIAgentWin_without_a_FM_Rand, SIAgentWin_without_a_FM_MM, SIAgentWin_without_a_FM_SI]  
Draw_without_a = [Draw_without_a_FM_Rand, Draw_without_a_FM_MM, Draw_without_a_FM_SI]

fig, ax = plt.subplots(figsize=(10, 6))
ind = np.arange(len(MM_win_with_a))
width = 0.25  
p1 = ax.bar(ind - width, MM_win_without_a, width, color='blue', label='Minmax Wins')
p2 = ax.bar(ind, SI_win_without_a, width, color='orange', label='Semi-Intelligent Wins')
p3 = ax.bar(ind + width, Draw_without_a, width, color='grey', label='Draws')
ax.set_xlabel('Game Type')
ax.set_ylabel('Number of Games')
ax.set_title('Minmax (without alpha pruning) Vs Semi-Intelligent player')
ax.set_xticks(ind)
ax.set_xticklabels(['Random First Move', 'Minmax First Move', 'Semi-Intelligent First Move'])
ax.legend()
def add_values(bar):
    for rect in bar:
        height = rect.get_height()
        if height != 0: 
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')
add_values(p1)
add_values(p2)
add_values(p3)
plt.show()


# Plot the bar graph for Semi-intelligent vs MinMax without pruning 

game_types = ['MinMax without A-B Pruning', 'Semi-Intelligent']
num_wins = [MinMaxWin_without_a_FM_MM, MinMaxWin_without_a_FM_SI]  # Number of games won by each player

# Plotting the bar graph
fig, ax = plt.subplots(figsize=(8, 6))
plt.bar(game_types, num_wins, color=['blue', 'green'])
plt.xlabel('Player Type')
plt.ylabel('Number of Wins')
plt.title('Performance Comparison: MinMax without A-B Pruning vs. Semi-Intelligent Player')
plt.ylim(0, 100)  # Set y-axis limit to 0-100 - no.of.games
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i in range(len(game_types)):
    plt.text(i, num_wins[i] + 1, str(num_wins[i]), ha='center', va='bottom')
plt.show()


# Plot the bar graph for Semi-intelligent vs MinMax without pruning 

game_types = ['MinMax with A-B Pruning', 'Semi-Intelligent']
num_wins = [MinMaxWin_with_a_FM_MM, MinMaxWin_with_a_FM_SI]  # Number of games won by each player

# Plotting the bar graph
fig, ax = plt.subplots(figsize=(8, 6))
plt.bar(game_types, num_wins, color=['blue', 'green'])
plt.xlabel('Player Type')
plt.ylabel('Number of Wins')
plt.title('Performance Comparison: MinMax with A-B Pruning vs. Semi-Intelligent Player')
plt.ylim(0, 1000)  # Set y-axis limit to 0-1000 - no.of.games
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i in range(len(game_types)):
    plt.text(i, num_wins[i] + 10, str(num_wins[i]), ha='center', va='bottom')
plt.show()