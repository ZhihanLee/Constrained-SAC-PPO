This is a prototype of **Constrained Soft Actor Crirtic** or **Soft Actor Critic Lagrangian** (**CSAC** or **SAC-Lagrangian**)

The basic SAC algorithm comes form ElegantRL :

https://github.com/AI4Finance-Foundation/ElegantRL

I established a Constrained SAC algorithm to deal with CMDP problem.

See 'Class AgentConstrainedSAC' in "AgentSAC.py" for details.


log 22.7.8: Detach update lambda from update_net method

log 22.7.30: A pytorch implementation of Proximal Policy Optimization with Lagranian (PPO-L) will be released soon

log 23.1.3: Add a pytorch implementation for PPO-Lagrangian with LSTM, see details in LSTM-PPO_L.py
