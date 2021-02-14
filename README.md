# Self-Balancing-Robot with Reinforcement Learning 
This project aims at developing a self balancing control system for a two-wheeled robot using Reinforcement Learning. 
The plan is to train a model in an emulator with simulated data, and transfer the trained model to a real-life two wheeled robot and continue training with real-life data.

## Agent models implemented
 1. Random search
 2. SARSA ([G. A. Rummery , M. Niranjan](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.2539))
 3. Q-learning ([C. J. C. H. Watkins](http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf))
<!-- 4. DQN ([DeepMind](https://www.nature.com/articles/nature14236))
 5. DDPG ([T. P. Lillicrap](https://arxiv.org/abs/1509.02971)) -->

## Directories
 - *Emulator*: contains the code to train agent in simulated environment
   - *Models*: contains a collection of agent models as listed above
<!--  - *RealEnv*: contains the code to run agent in real-life environment  -->

## To do:
 - modify emulator environment to become more realistic
 - re-train with the new environment
 - deploy trained model in real robot
