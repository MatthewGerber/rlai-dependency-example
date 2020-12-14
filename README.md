# `rlai` Dependency Example
This is an example of using the [`rlai`](https://github.com/MatthewGerber/rlai) package as a dependency for 
reinforcement learning. The goal here is to modify the probabilistic structure of a standard gridworld environment 
(specifically Example 4.1 in the Sutton and Barto (2018) text). The modification causes one of the goal states to be 
surrounded by states that are "sticky", in that they can be traversed in the usual way but incur a 10x larger negative 
reward for doing so. This can be interpreted as taking 10x longer to traverse than the other states (hence the nickname 
"sticky").

# Installing and Running the Example

1. Clone the repository.
```
git clone git@github.com:MatthewGerber/rlai-dependency-example.git
```
Note the structure of the example project, particularly `setup.py` (which brings in the `rlai` package) and the 
[sticky_gridworld.py](src/example/sticky_gridworld.py) script, which implements the example.

2.  Create and activate a fresh virtual environment, and install the example package.
```
cd rlai-dependency-example
virtualenv -p python3.8 venv
. venv/bin/activate
pip install -e .
```

3. Run the example.
```
python src/example/sticky_gridworld.py
```

4. Inspect the generated policy, which should be as follows.
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