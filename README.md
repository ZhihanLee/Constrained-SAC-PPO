This is a prototype of **Constrained Soft Actor Crirtic** or **Soft Actor Critic Lagrangian** (**CSAC** or **SAC-Lagrangian**)

The basic SAC algorithm comes form ElegantRL :

https://github.com/AI4Finance-Foundation/ElegantRL

I established a Constrained SAC algorithm to deal with CMDP problem.

See 'Class AgentConstrainedSAC' in "AgentSAC.py" for details.

However, I didn't use NN to update Lagrangian Multiplier λ.

In the future, I will add this method.


log 22.7.8: Detach update lambda from update_net method
