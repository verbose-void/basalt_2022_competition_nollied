# FractalZero

## Preface
I wasn't able to get a submission ready for evaluation in time for the deadline. I work full time, so during the majority of the competition I was only able to work on my free time, usually on weekends. However, at the end of the competition I took a full 2 weeks off work to focus 100% on my solution.

Though I feel I made tremendous progress in terms of code, I don't have any results to show and have limited time to write this document, so I'll detail the solution to the best of my abilities.

I observed signs of life from the technique described in the following sections, however most of my evidence (figures, plots, etc.) are scattered all over the place in many different wandb experiments/files on my harddrive and it would take a lot of time to piece them into a story that makes any amount of sense.

I plan on revisiting this project, and I'd like to eventually publish a paper on it when I have time. It would be nice to be considered for the research prize.

## Constraints
I began this challenge with a few self-imposed constraints to spice things up:
1. Don't use a value function in [the traditional sense](http://www.incompleteideas.net/book/ebook/node34.html). Temporal difference learning of value functions in deep-RL is notoriously hard and sensitive to hyperparameters. They're also highly dependent upon the ever-changing policy. I believe there should be a more robust solution, so I set out to not use them.
2. Design my solution to work for any arbitrary action/observation space with minimal architectural requirements.
3. Try not to train a policy function. Instead, center the focus of my training on the composition of the reward function. This is how I justified being able to not have a value function on the side of the "generating" agent.

I was inspired by the work of [GAIL](https://arxiv.org/abs/1606.03476) and [AIRL](https://arxiv.org/abs/1710.11248) to use a discriminator model in my solution. However, it wouldn't exactly be a GAN setup, because of my 3rd constraint -- Try not to train a policy model.

## Rare Event Sampling (RES)
Rare event sampling (RES) is a branch of research centered around physics (thermodynamics, chemistry, biology, etc.). It has not yet made it into the deep reinforcement learning community, despite it being highly applicable from my experience.

RES algorithms in essence are a sort of non-parametric tree search with the goal of efficiently sampling/measuring the probability of rare events leveraging uniformly random actions/perturbations. We can draw many parallels between these algorithms, and the role that the [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) (MCTS) plays in DRL for methods like [AlphaZero](https://arxiv.org/abs/1712.01815), and more importantly for my work here, [MuZero](https://arxiv.org/abs/1911.08265) which will be detailed in the FractalZero section. 

Importantly, MCTS uses a traditional value function (violating my 1st constraint) to efficiently perform it's lookahead.

Further, MCTS also is very hard to get to handle in domains of continuous action spaces, whereas MineRL has a mouse. Further, it's even more unstable in mixed-action space environments with discrete AND continuous. Minecraft is both discrete (keyboard) and continuous (mouse). So we need something better! 

## Fractal Monte Carlo (FMC)
Instead of MCTS/other RES algorithms, I chose to use [Fractal Monte Carlo (FMC)](https://arxiv.org/abs/1803.05049), a relatively unknown and forgotten RES algorithm.

Many RES methods use a population of coordinated random-acting agents called "walkers" to explore the environment. The different methods describe how these "walkers" coordinate, and FMC is one such method.

It's key distinction between FMC and other RES methods/MCTS is the usage of a contrastive orientation method. They calculate a "virtual rewards" vector which has 1 element for every walker. Each "virtual reward" value corresponds to the probability of a walker receiving reinforcements. The reinforcements come in the form of "clones", where a source walker copies and sets it's own state to some target walker's state, based on said probability.

The virtual reward is the exploration and exploitation balancing mechanism. 2 input vectors are first determined, relativized, then multiplied across:
1. Exploitation: The current "score" of a walker. For simpler cases like cartpole, this would be the cumulative rewards a walker has accumulated during it's trajectory.
2. Exploration: There are a couple steps to calculating the exploration vector:
   1. Assign every walker in the swarm a randomly selected partner walker. These will be the source and target walkers respectively.
   2. Calculate the distance between each pair of walkers. For the case of cartpole, you simply take the observation vector and calculate the distance between the two.

```python
# Both the explore and exploit vectors, before being multiplied together, are relativized independently:
def relativize_vector(vector: torch.Tensor):
    std = vector.std()
    if std == 0:
        return torch.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = torch.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return 
```

The source walker is then, based on the virtual rewards, told to clone or not clone to the target walker:

```python
# cloning (with virtual rewards)
def clone(self):
    # scores are either current rewards, average accumulated reward, accumulated reward, etc. The scale here is **invariant** because of the relitivization method.
    scores = self._get_scores()

    partners = # randomly assign partners
    walker_distances = 1 - F.cosine_similarity(self.states, self.states[partners], dim=-1)

    rel_scores = relativize_vector(scores)
    rel_distances = relativize_vector(walker_distances)

    # NOTE: if balance is 1.0, explore and exploit are equally weighted (generally gives good results)
    virtual_rewards = rel_scores ** self.balance * rel_distances

    # calculate the clone mask
    vr = virtual_rewards
    pair_vr = virtual_rewards[partners]
    value = (pair_vr - vr) / torch.where(vr > 0, vr, 1e-8)
    clone_mask = (value >= torch.rand(1)).bool()

    # execute cloning
    self.states[clone_mask] = self.states[partners[clone_mask]]
```

Since we sample actions uniformly, this satisfies my 2nd constraint, aiming to be applicable to arbitrary action spaces. Because gym environments always have a ".sample()" method to generate a uniformly random action.

Although this algorithm is quite simple, only having 2 hyperparameters (number of walkers/parallel states, and balance), it's very powerful for generating trajectories through the state space. For cartpole, 16 walkers and 200 steps of the simulation will always give you a winning trajectory with absolutely no neural networks or function approximators. Tons of atari games have also been solved this way. See a chart of scores [here](https://github.com/FragileTech/FractalAI#fractal-monte-carlo-agent-performance-table).

I've applied this technique to many real world problems and it's always been extremely useful.


## FractalZero (FMC + MuZero)
The reason why MuZero was chosen over AlphaZero is because it makes use of a "representation function" and a "dynamics function". These are important components for the application of a tree search/RES method **specific to the MineRL environment classes**, because in order for these techniques to work, you must be able to extract the current state of the environment AND be able to copy + restore from an extracted state. Due to the complexity of the MineRL environment class implementations, it is infeasible to copy and load states. Likely because it has a lot of expensive to run Java code.

MuZero's representation function takes the current observation from the environment and puts it into an embedding space. The following dynamics function then "unrolls" a simulated version of the actual environment's dynamics by taking in a sequence of actions and forwarding the embedding (provided originally from the representation function) along. So at each time step, the dynamics function takes in the corresponding action, and outputs the next state embedding alongside the reward it believes the original environment would have assigned. A tree search/RES method can then be applied to this.

Embeddings are trivial to copy and update, so our walkers can clone about as much as they'd like with very low latency.

This is analagous to how the pretrained MineRL models take the current observation and place them into an embedding space to be processed by the recurrent layer.  

With a perfect dynamics model (that's also feasible to run with low latency) and a perfect tree search/RES, you could theoretically find trajectories to terminal states assuming you have a good reward function to follow, regardless of the complexity of the original environment's implementation (ie. if it is copyable or not).

So, instead of using MuZero with a replay buffer and playing the game in the way MuZero did, I use the human demonstration data as the replay buffer (offline data).

But, there's one more piece missing that I haven't mentioned yet; and that's the "prediction function" from MuZero. The prediction function takes the current embedded state (either from the representation function OR the dynamics function) and outputs policy logits and the estimated value. However, because of my constraints, I decided to drop the "prediction function" entirely. I don't want to learn a policy, and I don't want to learn a value function.

Since MineRL does NOT provide a reward function... I had to get creative for how to train the dynamics/representation functions.

## Discriminator Dynamics Function
In GAIL and GANs in general, there are 2 competing "players" for some task. There's a discriminator, which is given some sample and should predict whether that sample came from an expert (in our case, human demonstrations of performing some specific minecraft task), OR the generator, which in the tradional case of GANs would be a basic neural network, and for GAIL, it would be something like a TRPO trained agent (or for more recent use cases, PPO).

They compete by both optimizing inverse versions of some loss function. For a binary classifying discriminator on a dataset of cat images, the discriminator's goal is to **minimize** binary cross entropy of "is cat" or "is fake cat". The generator's goal is to confuse the discriminator into thinking it's outputs are a real cat, and not a fake one. This is done by **maximizing** the discriminator's binary cross entropy.

This is quite an elegant setup, although they can be very sensitive to the balance of progress. If the discriminator gets too far ahead of the generator, it can de-stabilize the network and the generator may not generate anything useful within a reasonable amount of time.

This is where, instead of training a deep reinforcement learning agent to confuse the discriminator, I opted to use FMC as a policy improvement operator (the role that MCTS plays in deep reinforcement learning). Since it's fast to run and doesn't require a value function, I had FMC maximize the confusion of the discriminator.

What's also cool about this setup, is I can have the discriminator be generalized to all tasks by having it perform multi-class classificaiton, where the goal is to predict which task a trajectory belongs to (find cave, make waterfall, etc.) AND an extra class logit for the generator (FMC). Softmaxing this output, gets you a reward target for FMC to optimize in the dynamics environment...

**So, actually, the discriminator IS the MuZero dynamics function!** The dynamics function is trained to, at each time step, minimize the cross entropy loss for 5 classes:
1. Trajectory is task Find Cave
2. Trajectory is task Make Waterfall
3. Trajectory is task Build House
4. Trajectory is task Animal Pen
5. Trajectory was generated by **FMC**

Naturally, this is easy to optimize, because we already have the dataset and can train a video classifier (with an extra set of input features, being the actions taken).

For FMC, the environment that's being used is the dynamics function with the reward as being the target task logit (after all logits have been softmaxed). This reward function means we're kind of trying to generate adversarial examples, which, with the 5th classification label being FMC itself, the discriminator is hypothesized to become robust to these examples, and since FMC is a policy improvement operator much like MCTS, the idea would be that FMC never gets TOO good and the discriminator (so long as it's playing by the rules -- more on that in the "FMC Got Hacked" section) also never gets too good, since it's embeddings are directly being used by FMC as part of it's cellular automata rule set.

