# Markov Decision Processes

This is a package for solving MDPs. We leverage the existing MDPToolbox, but add a bit more functionality that we need.


In particular, we need to be able to:
- augment a state space with time so that we may have time-varying transition probabilities
- estimate transition probabilities with a ML model (e.g., catboost)


## Getting started

To run the example notebooks, run

```
uv sync --dev
```

Go to `src/examples/` and fire up a notebook.



## Generic Problem

Suppose we have a set of regularly departing services, such as a bus or ferry, that have a long selling window before departure. For simplicity, we'll consider a single service and suppress any indexing indentifying that service.

The selling period evolves in dicrete time steps: $t \in \{T, T-1, \dots, 2, 1, 0\}$. Time $t = T$ is when tickets for the service are open for sale; time $t=0$ is when the service departs.

At the beginning of each period, the service operator gets to choose from a discrete set of prices: $p_{t} \in \{P_{0}, P_{1}, \dots, P_{n}\}$

The service has a fixed capacity $C$. In each period, some of remaining capacity $c_{t}$ can be sold at the price $p_{t}$. For simplicity, we discretize the amount of capacity that can be sold into a grid with mesh $\Delta m$:

$$\mathcal{C} =  \{0,\, \Delta m,\, 2\Delta m,\, \dots,\, (m-1)\Delta m,\, C\}$$


The state space for this problem is 

$$ S = \mathcal{C} \times \{T, T-1, \dots, 2, 1, 0\} $$


We suppose that there are a set of static covariates, $X$, that affect the demand for the service. E.g., peak/off-peak season; time of day the service departs. We also allow demand to depend on the day in the selling window $t$.


For states $s_{t}, s_{t+1} \in S$, our Bellman equation is


$$
V(s_t) = \max_{p_t} \{
\sum_{s_{t+1}}
\mathbb{P}(s_{t+1} \mid s_t, p_t, t, X)
\left ( p_t (c_t - c_{t+1}) + \gamma V(s_{t+1}) \right)
\}
$$



Note that our period reward function $r(p_{t}, s_{t}, s_{t+1}) = \sum_{s_{t+1}} p_{t}\left(c_{t} - c_{t+1} \right) \mathbb{P}(s_{t+1}\, | \, s_{t}, p_{t}, t, X)$ is random because our action, to choose a particular price, does not give a deterministic outcome. The demand for capacity in this period is uncertain. This is handled natively by packages such as MDPtoolbox.


### Things we need to do to set up the MDP

1. Choose values for $m$ and $T$. The smaller $m$ and the bigger $T$, the bigger the state space becomes. Start coarse, then go finer until computation becomes intolerably slow.
2. If you are given a set of allowable prices to choose from, use that. If you are allowed to price continuously, create a discrete grid as for capacity.
3. Choose a value for $\gamma$. If we're in the finite horizon setting, use $\gamma = 1$.
3. Train a model for $\mathbb{P}(s_{t+1}\, | \, s_{t}, p_{t}, t, X)$. (discussion below)
4. Construct a set of transition matrices for each action (price). You should have a list: $[ T_{p_{0}}, T_{p_{1}}, \dots, T_{p_{n}}]$ of matrices that are square. The $i,j$ elements of $T_{p_{0}}$ are the probability of transitioning from state $i$ to state $j$ when price $p_{0}$ is chose.
5. Construct a set of reward matrices for each action. You should have a list: $R_{p_{0}}, R_{p_{1}}, \dots, R_{p_{n}}$. The $i,j$ elements of $R_{p_{0}}$ is the reward resulting from a transition from state $i$ to state $j$ when price $p_{0}$ was chosen (i.e., $p_{0}(c_{i} - c_{j})$)
6. Load these into the MDP Toolbox's FiniteHorizon class and use it to solve for the optimal policy and value functions. The policy function tells you what action to take in a given state. The value function tells you the expected discounted revenue resulting from following the optimal policy.



So the hard part there is that we need a model for the transition probabilities. To keep this Markovian, we need to ensure that the model only uses the current remaining capacity at the beginning of the period, the price chosen for today, the time of the selling window, and static covariates. The target variable is the remaining capacity tomorrow. Basically any regression model will give you $\mathbb{E}[c_{t+1}]$ as a function of those features. The wrinkle here is that we need not just the expected value of the target, but the whole probability distribution for it. Options there might be:

- estimate the model for the mean, then assume that the random disturbances are normally distributed. Estimate the standard deviation on a test set. Then use the normality assumption to compute how much probability mass goes to each possible $c_{t+1}$ in the grid.
- use a model that can give you many quantiles for the target. Estimate a model for each quantile in $\{.05, .1, 0.15, \dots, .95, .975\}$ and use that to back out the pdf.