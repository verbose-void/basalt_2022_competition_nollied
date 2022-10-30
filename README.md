# BASALT 2022 Challenge Research

I began this challenge with a few self-imposed constraints to spice things up:
1. Don't use a value function in [the traditional sense](http://www.incompleteideas.net/book/ebook/node34.html). Temporal difference learning of value functions in deep-RL is notoriously hard and sensitive to hyperparameters. They're also highly dependent upon the ever-changing policy. I believe there should be a more robust solution, so I set out to not use them.
2. Design my solution to work for any arbitrary action/observation space with minimal architectural requirements.
3. Try not to train a policy function, instead, center the focus of my training on the composition of the reward function. This is how I justified being able to not have a value function on the side of the "generating" agent.

I was inspired by the work of [GAIL](https://arxiv.org/abs/1606.03476) and [AIRL](https://arxiv.org/abs/1710.11248) to use a discriminator model in my solution. However, it wouldn't exactly be a GAN setup, because of my 3rd constraint -- Try not to train a policy model.

## Rare Event Sampling (RES)
Rare event sampling (RES) is a branch of research centered around physics (thermodynamics, chemistry, biology, etc.). It has not yet made it into the deep reinforcement learning community, despite being highly applicable.

RES algorithms in essence are a sort of non-parametric tree search with the goal of efficiently sampling/measuring the probability of rare events without bias. We can draw many parallels between these algorithms, and the role that the [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) (MCTS) plays in DRL for methods like [AlphaZero](https://arxiv.org/abs/1712.01815), and more importantly for my work here, [MuZero](https://arxiv.org/abs/1911.08265). 

Importantly, MCTS uses a value function (violating my 1st constraint) to efficiently perform it's lookahead.

## MuZero + Rare Event Sampling
The reason why MuZero was chosen over AlphaZero is because it makes use of a "representation function" and a "dynamics function". These are important components for the application of a tree search/RES method to the MineRL environments because in order for these techniques to work, you must be able to extract the current state of the environment AND be able to copy + restore from an extracted state, which, due to the complexity of the MineRL environments, is infeasible with the current implementations.

MuZero's representation function takes the current observation from the environment and puts it into an embedding space. The following dynamics function then "unrolls" a simulated version of the actual environment's dynamics by taking in a sequence of actions and forwarding the embedding (provided originally from the representation function) along. So at each time step, the dynamics function takes in the corresponding action, and outputs the next state embedding.

This is analagous to how the pretrained MineRL models take the current observation and place them into an embedding space to be processed by the recurrent layer.  

With a perfect dynamics model (that's also feasible to run with low latency) and a perfect tree search/RES, you could theoretically find trajectories to terminal states assuming you have a good reward function to follow, regardless of the complexity of the original environment's implementation (ie. if it is copyable or not).



## Discriminator Dynamics Function


## XIRL/TCC Representation Function
TODO: note about how i observed that the pretrained image process models that were provided may have already had the properties that TCC trained embeddings had (structured embedding space w.r.t temporality/task completion)
