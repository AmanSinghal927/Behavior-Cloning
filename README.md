# Behavior Cloning Report

## GIF Demonstrations

### Fixed Goal BC
![Fixed Goal BC](changing_goal_bc.gif)

### Changing Goal BC
![Changing Goal BC](changing_goal_gcbc.gif)

### Fixed Goal GCBC
![Fixed Goal GCBC](fixed_goal_bc.gif)

### Changing Goal GCBC
![Changing Goal GCBC](multimodal_bet.gif)

### Multimodal BET
![Multimodal BET](multimodal_gcbc.gif)

### Multimodal BC
![Multimodal BC](multimodal_vinn.gif)

### VI-NN
![VI-NN](conda_env.yml)

## Report Content

### 1. Fixed goal BC
For fixed goal BC, the goal is fixed, however the starting position of the agent keeps on changing. The BC agent is able to learn to reach the goal.

### 2. Changing goal BC
The agent moves slowly after training and does not reach the goal every time even after training for 8000 iterations. This is because the agent has no information of the goal and is hence randomly. If it reaches the goal it is due to chance.

### 3. Fixed goal GCBC
Having knowledge of the goal is unnecessary in this experiment as the goal is fixed across episodes. The agent is easily able to reach the goal.

### 4. Changing goal GCBC
The actor takes the goal as well as the state as input. Hence the agent is able to reach the goal even as it is changing across episodes.

### 5. Multimodal BET
The behavior transformer is able to follow trajectories as expected. The agent “appears” to move in only horizontal or only vertical directions at once. This is because the model outputs two heads – one responsible for offsets and the other is responsible for predicting the bins.

### 6. Multimodal BC
The agent is able to reach the goal, however it fails to adhere to only horizontal or only vertical trajectories (and moves diagonally). This is because of the MSE loss. BET solves the problem by sampling at a bin level and predicting offsets. Hence, even though both agents get a reward of 1, BET is taking trajectories per our expectation while BC is unable to follow to “only horizontal” or “only vertical” trajectories.

### 7. VI-NN
The vinn agent is able to make jagged progressions towards the goal. It is able to reach the goal ultimately. There is no training involved. For running replace the train.py file with the file in the train.py in the vinn folder, also include the configuration file for vinn.
