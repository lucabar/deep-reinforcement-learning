# Deep Reinforcement Learning - Assignment 2 - Group 13
Our main classes and functions are in the `dqn.py` file and depending on the experiment you want to recreate, use the following flags.

## Ablation study, double DQN, DDQN, no training
### Run
To run basic deep Q network without replay buffer or target network
```
python dqn.py
```

#### Add optional flags for:  
activating the target network or experience replay

```
python dqn.py --target_active --experience_replay
```

playing without learning, using pre-trained weights

```
python dqn.py --no_training
```

running Double DQN

```
python dqn.py --double
```

running Dueling DQN

```
python dqn.py --dueling
```

Please do not use other combinations than those mentioned above. We did not make it fully functional and self-consistent.

