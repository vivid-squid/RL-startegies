The brief overview of the things done are provided below based on Python files.

# Model generation for each game using Q-learning RL method:

Pickle File Generating code for TicTacToe - 
    def saveQLearningModel(self):
        with open("TicTacToeQLearningTraningModel.pickle", "wb") as file:
            pickle.dump(self.QLearningStates, file)
            
    def loadQLearningModel(self):
        with open("TicTacToeQLearningTraningModel.pickle", "rb") as file:
            self.QLearningStates = pickle.load(file)

Pickle File Generating code connect4 - 
def saveQLearningModel(self):
        with open("Connect4QLearningTraningModel.pickle", "wb") as file:
            pickle.dump(self.QLearningStates, file)
            
    def loadQLearningModel(self):
        with open("Connect4QLearningTraningModel.pickle", "rb") as file:
            self.QLearningStates = pickle.load(file)


# Tic-Tac-Toe Minimax Algorithm with Alpha-Beta Pruning

This repository contains Python code for implementing the Tic-Tac-Toe game with the Minimax algorithm, both with and without Alpha-Beta pruning, along with a Semi-Intelligent player as an opponent. The code provides a comparison of the performance of these algorithms in terms of wins and execution time.

## Files

- `TicTacToe_MinMax.py`: python file containing the implementation of the Tic-Tac-Toe game with Minimax algorithm and Alpha-Beta pruning.

## Import .py file in python file
To use the provided python file (`Connect4_MinMax_Model.py`), follow these steps:

1. Open a terminal and navigate to the directory where the `Connect4_MinMax_Model.py` file is located.
2. Run the following command to start python file:
   ```
   python file_name.py
   ```


# Tic-Tac-Toe with Q-Learning

## Overview
This project implements a Tic-Tac-Toe game using the Q-learning algorithm to train an AI agent to play against a semi-intelligent player. The code is provided in a python file format.

## Files
- `TicTacToeQLearning.py`: python file containing the Python code for the Tic-Tac-Toe game implementation with Q-learning.
- `TicTacToeQLearningTraningModel.pickle`: Pickle file containing the trained Q-learning model.

## Import .py file in Jupyter

## Run the Code


# Tic-Tac-Toe Q-Learning vs. MinMax Player

This repository contains Python code for simulating Tic-Tac-Toe games played between a Q-Learning agent and a MinMax player. The code evaluates the performance of both agents under various scenarios and provides statistics on game outcomes.

## Code Overview

The main components of the code include:

- `TicTacToe_Game`: Class handling the Tic-Tac-Toe game logic.
- `QLearning`: Class implementing the Q-learning algorithm for learning the game.
- `TicTacToe_MinMax`: Class implementing the Minimax algorithm with alpha-beta pruning for making moves.
- `play_tic_tac_toe`: Function simulating a game between the Q-learning agent and the Minimax agent.
- Experimentation: Conducts experiments to compare the performance of both agents under different initial move conditions.

## Run the python file


# Connect4QLearningModel.py

## Introduction
This Python file contains the code for training a Q-learning model to play Connect Four against a semi-intelligent agent. It also includes visualization of the performance comparison between the Q-learning agent and the semi-intelligent player.

## Generating the Pickle File
The pickle file, named Connect4QLearningTraningModel.pickle, is generated during the training process of the Q-learning model. Here's the relevant code snippet for generating the pickle file:
```python
# Saving the Q-learning model
qLearning = QLearning()
qLearning.saveQLearningModel()
```

## Run the python file


# Connect4_MinMax_Model

This repository contains Python code for playing Connect Four using MinMax algorithm with Alpha-Beta pruning and Depth = 8, along with a Semi-Intelligent player as opponent.

## Overview
Connect Four is a two-player connection board game, where the players choose a color and take turns dropping colored discs into a vertically suspended grid. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four discs of your color.

This repository provides implementations for two players:
1. **MinMax Player with Depth = 6/8:** The MinMax algorithm with Alpha-Beta pruning is implemented to make optimal moves. The depth of the search tree is set to balance between computational complexity and performance.
2. **Semi-Intelligent Player:** This player makes somewhat intelligent moves by checking for immediate threats and opportunities to form four-in-a-row horizontally, vertically, or diagonally.

## Run the python file


# Connect4 Game: Q-Learning vs MinMax

This repository contains code for simulating games of Connect Four between a Q-Learning player and a MinMax player. The code is written in Python and is executed within a python file environment.

## Run the python file


# Dependencies

The code relies on the following Python libraries:

- `matplotlib`: For plotting graphs and visualizations.
- `pandas`: For data manipulation and analysis.
- `tqdm`: For displaying progress bars during experimentation.
- `numpy`: For performing calculations.
- 'Jinja2': For handling templates.

Make sure to install these libraries before running the code.

These libraries can be installed using pip: 
```
pip install numpy pandas matplotlib tqdm Jinja2
```

# Visualizing the Plot

To visualize the plot generated by the code, follow these steps:

1. Ensure that the necessary Python libraries are installed.
2. Run the python code containing the plot generation code.
3. The plot should be displayed directly in a separate window.


