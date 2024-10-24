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

def play_connect4(SIAgent_plays_first, minmax_agent, si_agent, c4_game) : 
    MinMaxLetter = 1
    SIAgentLetter = 2
    isGameOver = False
    gameWinner = ''
    while not isGameOver : 
        
        if SIAgent_plays_first : 
            
            si_chosen_row, si_chosen_column = si_agent.Semi_Intelligent_Agent_Move(c4_game, SIAgentLetter, MinMaxLetter)
            
            if c4_game.validateMove(si_chosen_column-1):
                SIAgent_plays_first = False
                c4_game.connect4_board[si_chosen_row][si_chosen_column] = SIAgentLetter
                
                if c4_game.validateWin(SIAgentLetter):
                    isGameOver = True
                    gameWinner = 'SIAgentWon'
                    
            else:
                continue
        else:
            
            minmax_chosen_column, _ = minmax_agent.Min_Max_Move_with_alpha_beta_pruning_and_depth(c4_game, c4_game.connect4_board, 
                                        6, True, MinMaxLetter, SIAgentLetter, -math.inf, math.inf)


            if c4_game.validateMove(minmax_chosen_column):
                SIAgent_plays_first = True
                minmax_chosen_row = c4_game.getNextAvailableRow(minmax_chosen_column)
                c4_game.connect4_board[minmax_chosen_row][minmax_chosen_column] = MinMaxLetter

                if c4_game.validateWin(MinMaxLetter):
                    isGameOver = True
                    gameWinner = 'MinMaxWon'

            else:
                continue
    
    return gameWinner if gameWinner != '' else 'Draw'


# Min-Max Depth comparison Depth = 8 with the first move as Random

games = 100
SIAgentWin = 0
MinMaxWin = 0
Draw = 0

minmax_agent = MinMax()
si_agent = SI_Agent()

startTime = time.time()

for _ in tqdm(range(games)):
    c4_game = Connect4_Game()
    c4_game.initialise_board()
    
    SIAgent_plays_first = False
    if c4_game.tossForFirstMove() == 1 :
        SIAgent_plays_first = True
    else : 
        SIAgent_plays_first = False
    
    try:
        winner = play_connect4(SIAgent_plays_first, minmax_agent, si_agent, c4_game)
    except:
        continue
        
    if winner == 'MinMaxWon':
        MinMaxWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1
        
totalTime = time.time()-startTime


statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Time taken (in seconds) with Depth = 8', 'Number of Games MinMax Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Random'
statistics_dict['Total Number of Games'] = games
statistics_dict['Time taken (in seconds) with Depth = 8'] = totalTime
totalTime_d8_FM_Rand = totalTime
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_d8_FM_Rand = MinMaxWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_d8_FM_Rand = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_d8_FM_Rand = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# Min-Max Depth comparison Depth = 8 with the first move as always MinMax

games = 100
SIAgentWin = 0
MinMaxWin = 0
Draw = 0

minmax_agent = MinMax()
si_agent = SI_Agent()

startTime = time.time()

for _ in tqdm(range(games)):
    c4_game = Connect4_Game()
    c4_game.initialise_board()
    
    SIAgent_plays_first = False
    
    try:
        winner = play_connect4(SIAgent_plays_first, minmax_agent, si_agent, c4_game)
    except:
        continue
        
    if winner == 'MinMaxWon':
        MinMaxWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1
        
totalTime = time.time()-startTime

statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Time taken (in seconds) with Depth = 8', 'Number of Games MinMax Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: MinMax Player'
statistics_dict['Total Number of Games'] = games
statistics_dict['Time taken (in seconds) with Depth = 8'] = totalTime
totalTime_d8_FM_MM = totalTime
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_d8_FM_MM = MinMaxWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_d8_FM_MM = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_d8_FM_MM = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# Min-Max Depth comparison Depth = 8 with the first move as always Semi-Intelligent Agent

games = 100
SIAgentWin = 0
MinMaxWin = 0
Draw = 0

minmax_agent = MinMax()
si_agent = SI_Agent()

startTime = time.time()

for _ in tqdm(range(games)):
    c4_game = Connect4_Game()
    c4_game.initialise_board()
    
    SIAgent_plays_first = True
    
    try:
        winner = play_connect4(SIAgent_plays_first, minmax_agent, si_agent, c4_game)
    except:
        continue
        
    if winner == 'MinMaxWon':
        MinMaxWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1
        
totalTime = time.time()-startTime


statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Time taken (in seconds) with Depth = 8', 'Number of Games MinMax Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Semi Intelligent Player'
statistics_dict['Total Number of Games'] = games
statistics_dict['Time taken (in seconds) with Depth = 8'] = totalTime
totalTime_d8_FM_SI = totalTime
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_d8_FM_SI = MinMaxWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_d8_FM_SI = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_d8_FM_SI = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)



# Min-Max Depth comparison Depth = 6 with the first move as always Random

games = 100
SIAgentWin = 0
MinMaxWin = 0
Draw = 0

minmax_agent = MinMax()
si_agent = SI_Agent()

startTime = time.time()

for _ in tqdm(range(games)):
    c4_game = Connect4_Game()
    c4_game.initialise_board()
    
    SIAgent_plays_first = False
    if c4_game.tossForFirstMove() == 1 :
        SIAgent_plays_first = True
    else : 
        SIAgent_plays_first = False
    
    try:
        winner = play_connect4(SIAgent_plays_first, minmax_agent, si_agent, c4_game)
    except:
        continue
        
    if winner == 'MinMaxWon':
        MinMaxWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1
        
totalTime = time.time()-startTime


statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Time taken (in seconds) with Depth = 6', 'Number of Games MinMax Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Random'
statistics_dict['Total Number of Games'] = games
statistics_dict['Time taken (in seconds) with Depth = 6'] = totalTime
totalTime_d6_FM_Rand = totalTime
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_d6_FM_Rand = MinMaxWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_d6_FM_Rand = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_d6_FM_Rand = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# Min-Max Depth comparison Depth = 6 with the first move as always Min-Max

games = 100
SIAgentWin = 0
MinMaxWin = 0
Draw = 0

minmax_agent = MinMax()
si_agent = SI_Agent()

startTime = time.time()

for _ in tqdm(range(games)):
    c4_game = Connect4_Game()
    c4_game.initialise_board()
    
    SIAgent_plays_first = False
    
    try:
        winner = play_connect4(SIAgent_plays_first, minmax_agent, si_agent, c4_game)
    except:
        continue
        
    if winner == 'MinMaxWon':
        MinMaxWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1
        
totalTime = time.time()-startTime

statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Time taken (in seconds) with Depth = 6', 'Number of Games MinMax Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: MinMax Player'
statistics_dict['Total Number of Games'] = games
statistics_dict['Time taken (in seconds) with Depth = 6'] = totalTime
totalTime_d6_FM_MM = totalTime
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_d6_FM_MM = MinMaxWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_d6_FM_MM = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_d6_FM_MM = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# Min-Max Depth comparison Depth = 6 with the first move as always Semi-Intelligent Agent

games = 100
SIAgentWin = 0
MinMaxWin = 0
Draw = 0

minmax_agent = MinMax()
si_agent = SI_Agent()

startTime = time.time()

for _ in tqdm(range(games)):
    c4_game = Connect4_Game()
    c4_game.initialise_board()
    
    SIAgent_plays_first = True
    
    try:
        winner = play_connect4(SIAgent_plays_first, minmax_agent, si_agent, c4_game)
    except:
        continue
        
    if winner == 'MinMaxWon':
        MinMaxWin += 1
    elif winner == 'SIAgentWon':
        SIAgentWin += 1
    else:
        Draw += 1
        
totalTime = time.time()-startTime


statistics_df = pd.DataFrame(columns=['Game Type', 'Total Number of Games', 'Time taken (in seconds) with Depth = 6', 'Number of Games MinMax Won', 'Number of Games Semi-Intelligent player Won', 'Number of Games Drawn'])
statistics_dict = {}
statistics_dict['Game Type'] = 'First Move: Semi Intelligent Player'
statistics_dict['Total Number of Games'] = games
statistics_dict['Time taken (in seconds) with Depth = 6'] = totalTime
totalTime_d6_FM_SI = totalTime
statistics_dict['Number of Games MinMax Won'] = MinMaxWin
MinMaxWin_d6_FM_SI = MinMaxWin
statistics_dict['Number of Games Semi-Intelligent player Won'] = SIAgentWin
SIAgentWin_d6_FM_SI = SIAgentWin
statistics_dict['Number of Games Drawn'] = Draw
Draw_d6_FM_SI = Draw
statistics_df = statistics_df.append(statistics_dict, ignore_index = True)
statistics_df = statistics_df.style.applymap(lambda x:'white-space:nowrap')
display(statistics_df)


# Plot the performance of MinMax

game_types = ['Random', 'MinMax', 'Semi-Intelligent']
depth_6_times = [totalTime_d6_FM_Rand, totalTime_d6_FM_MM, totalTime_d6_FM_SI]  # Time taken with Depth = 6
depth_8_times = [totalTime_d8_FM_Rand, totalTime_d8_FM_MM, totalTime_d8_FM_SI]  # Time taken with Depth = 8

# Plotting
bar_width = 0.35
index = range(len(game_types))

fig, ax = plt.subplots()
bar1 = ax.bar(index, depth_6_times, bar_width, label='Depth = 6')
bar2 = ax.bar([i + bar_width for i in index], depth_8_times, bar_width, label='Depth = 8')

ax.set_xlabel('Game Types')
ax.set_ylabel('Time taken (in seconds)')
ax.set_title('Performance of MinMax Algorithm with Different Depths')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(game_types)
ax.legend()

plt.tight_layout()
plt.show()

# Plot the time taken for Depth 8 MinMax vs Semi-inelligent 

game_types = ['MinMax (Depth = 8)', 'Semi-Intelligent']
time_taken = [totalTime_d8_FM_MM, totalTime_d8_FM_SI] 

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(game_types, time_taken, color=['blue', 'orange'])
plt.xlabel('Game Types')
plt.ylabel('Time taken (in seconds)')
plt.title('Performance Comparison: MinMax (Depth = 8) vs. Semi-Intelligent Player')
plt.ylim(0, max(time_taken) * 1.2)  
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, value in enumerate(time_taken):
    plt.text(i, value + 5, f"{value:.2f}", ha='center')

plt.tight_layout()
plt.show()

# Plot the time taken for Depth 6 MinMax vs Semi-inelligent 

game_types = ['MinMax (Depth = 6)', 'Semi-Intelligent']
time_taken = [totalTime_d6_FM_MM, totalTime_d6_FM_SI]  

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(game_types, time_taken, color=['blue', 'orange'])
plt.xlabel('Game Types')
plt.ylabel('Time taken (in seconds)')
plt.title('Performance Comparison: MinMax (Depth = 6) vs. Semi-Intelligent Player')
plt.ylim(0, max(time_taken) * 1.2) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, value in enumerate(time_taken):
    plt.text(i, value + 2, f"{value:.2f}", ha='center')
plt.tight_layout()
plt.show()

# Plot for comparing number of wins for Depth 6 MinMax vs Semi-inelligent

MM_win_d6 = [MinMaxWin_d6_FM_Rand, MinMaxWin_d6_FM_MM, MinMaxWin_d6_FM_SI]  
SI_win_d6 = [SIAgentWin_d6_FM_Rand, SIAgentWin_d6_FM_MM, SIAgentWin_d6_FM_SI]  
Draw_d6 = [Draw_d6_FM_Rand, Draw_d6_FM_MM, Draw_d6_FM_SI]

fig, ax = plt.subplots(figsize=(10, 6))
ind = np.arange(len(MM_win_d6))
width = 0.25  
p1 = ax.bar(ind - width, MM_win_d6, width, color='blue', label='Minmax Wins')
p2 = ax.bar(ind, SI_win_d6, width, color='orange', label='Semi-Intelligent Wins')
p3 = ax.bar(ind + width, Draw_d6, width, color='grey', label='Draws')
ax.set_xlabel('Game Type')
ax.set_ylabel('Number of Games')
ax.set_title('Minmax (Depth = 6) Vs Semi-Intelligent player')
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

# Plot for comparing number of wins for Depth 8 MinMax vs Semi-inelligent

MM_win_d8 = [MinMaxWin_d8_FM_Rand, MinMaxWin_d8_FM_MM, MinMaxWin_d8_FM_SI]  
SI_win_d8 = [SIAgentWin_d8_FM_Rand, SIAgentWin_d8_FM_MM, SIAgentWin_d8_FM_SI]  
Draw_d8 = [Draw_d8_FM_Rand, Draw_d8_FM_MM, Draw_d8_FM_SI]

fig, ax = plt.subplots(figsize=(10, 6))
ind = np.arange(len(MM_win_d8))
width = 0.25  
p1 = ax.bar(ind - width, MM_win_d8, width, color='blue', label='Minmax Wins')
p2 = ax.bar(ind, SI_win_d8, width, color='orange', label='Semi-Intelligent Wins')
p3 = ax.bar(ind + width, Draw_d8, width, color='grey', label='Draws')
ax.set_xlabel('Game Type')
ax.set_ylabel('Number of Games')
ax.set_title('Minmax (Depth = 8) Vs Semi-Intelligent player')
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