# `rlai` Examples
This repository provides examples of using the [`rlai`](https://github.com/MatthewGerber/rlai) package for 
reinforcement learning. 

# Installation
1. Clone the repository:
   ```
   git clone git@github.com:MatthewGerber/rlai-dependency-example.git
   ```
   Note the structure of the example project, particularly `pyproject.toml` (which brings in the `rlai` package).

2. Install to a new Python environment:
   ```
   cd rlai-dependency-example
   poetry env use 3.11
   poetry install
   ```

# Sticky Gridworld
This example modifies the probabilistic structure of a standard gridworld environment, specifically Example 4.1 in the 
Sutton and Barto (2018) text. The modification causes one of the goal states to be surrounded by states that are 
"sticky", in that they can be traversed in the usual way but incur a 10x larger negative reward for doing so. This can 
be interpreted as taking 10x longer to traverse than the other states (hence the nickname "sticky").

Run the [example code](src/examples/gridworld/sticky_gridworld.py):

```
python src/example/sticky_gridworld.py
```

The generated policy should be as follows.

```
State 0:
	Pr(u):  0.25
	Pr(d):  0.25
	Pr(l):  0.25
	Pr(r):  0.25
State 1:
	Pr(l):  1.0
State 2:
	Pr(l):  1.0
State 3:
	Pr(l):  1.0
State 4:
	Pr(u):  1.0
State 5:
	Pr(l):  1.0
State 6:
	Pr(l):  1.0
State 7:
	Pr(u):  1.0
State 8:
	Pr(u):  1.0
State 9:
	Pr(u):  1.0
State 10:
	Pr(l):  1.0
State 11:
	Pr(d):  1.0
State 12:
	Pr(u):  1.0
State 13:
	Pr(l):  1.0
State 14:
	Pr(r):  1.0
State 15:
	Pr(u):  0.25
	Pr(d):  0.25
	Pr(l):  0.25
	Pr(r):  0.25
```

In the above, note that states 0 (upper-left corner) and 15 (lower-right corner) are goal states in the 4x4 gridworld. 
The optimal action in state 14 is to move right, incur the -10 reward, and reach the goal state 15. By contrast, the 
optimal action in state 13 is to move left toward state 0 in the upper-left corner, ignoring the goal state 15 just two 
steps to the right owing to the -10 reward that would be received after entering state 14.

# OpenAI Gym Cartpole with Stochastic Gradient Descent
This example recreates the case study described 
[here](https://matthewgerber.github.io/rlai/case_studies/inverted_pendulum.html).

Run the [example code](src/examples/openai_gym/cartpole.py):

```
python src/examples/openai_gym/cartpole.py
```

Training plots will be displayed, and a video will be rendered every 100 episodes.