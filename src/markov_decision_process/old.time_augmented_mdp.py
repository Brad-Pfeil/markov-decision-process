# ----------------------------------------------------------------------------#
#
#                              TimeAugmentedMDP
#
# ----------------------------------------------------------------------------#

import numpy as np
import pandas as pd

from typing import Any, Callable

from scipy.sparse import csr_matrix

from sklearn import preprocessing

from itertools import product

from mdptoolbox.mdp import ValueIteration, FiniteHorizon

import matplotlib.pyplot as plt
import seaborn as sns

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------#


class TimeAugmentedMDP:
    """
    A class to represent a Time-Augmented Markov Decision Process (MDP).

    This class provides methods to build transition and reward matrices from
    either a function or a given model, and to plot the transition
    probabilities at specific time steps for given actions.

    Attributes:
    -----------
    S : dict
        A dictionary representing the states.
    T : dict
        A dictionary representing the time steps.
    A : dict
        A dictionary representing the actions.
    S_augmented : dict
        A dictionary representing the time-augmented states.
    transitions : list
        A list to store the transition matrices.
    rewards : list
        A list to store the reward matrices.

    Methods:
    --------
    build_transitions_and_rewards_from_model(model: Callable, static_vars:
    dict[str, Any] | None = None) -> None:
        Builds the transition and reward matrices from the given model.

    get_transition_probabilties(s: int | str, t: int, a: int | str) -> dict:
        Returns the transition probabilities for a given state, time, and
        action.

    plot_transitions(t: int, a: int | str, ax: plt.Axes | None = None,
    annotate: bool = False, vmin: float = 0, vmax: float | None = None) ->
    plt.Figure:
        Plots the transition probabilities at a specific time step for a given
        action using a seaborn heatmap.
    """

    def __init__(
        self,
        S: dict[int, str | int | float],
        A: dict[int, str | int | float],
        T: dict[int, int],
        gamma: float = 1,
    ) -> None:
        # States ,actions, and time steps
        self.S = S
        self.A = A
        self.T = T
        self.gamma = gamma

        # We need to do a few checks.

        # At least one state, action and time.
        assert len(self.S) > 0, "State space must have at least one element"
        assert len(self.A) > 0, "Action space must have at least one element"
        assert len(self.T) > 0, "Time space must have at least one element"

        # Check that the gamma value is between 0 and 1
        assert 0 < self.gamma <= 1, "Gamma must be between 0 and 1"

        # Add a dummy time period to the time space
        self.T[len(self.T)] = max(self.T.values()) + 1

        # Inverse dictionaries
        self.inverse_S = {v: k for k, v in self.S.items()}
        self.inverse_A = {v: k for k, v in self.A.items()}
        self.inverse_T = {v: k for k, v in self.T.items()}

        self.policy_function = None
        self.value_function = None
        self.transitions = []
        self.rewards = []

        # Discretized state space (if applicable)
        self.s_grid = []

        # Payoff to assign infeasible transitions
        self.INFEASIBLE = -1e8

        self.S_augmented = self._augment_state_space_with_time()

        return None

    def _augment_state_space_with_time(self) -> dict[int, dict[str, Any]]:
        """
        Augments the state space by taking the cartesian product of the
        original state space with the time horizon.

        The returned dictionary has keys equal to the augmented state index.

        Returns:
        --------
        dict[int, dict[str, Any]]:
            A dictionary where each key is an augmented state index and each
            value is a dictionary containing:
                - "label": A string representing the state and time.
                - "s": The original state.
                - "t": The time step.
                - "s_index": The index of the original state.
                - "t_index": The index of the time step.
        """

        S_augmented = {
            l: {
                "label": f"(s={s}, t={t})",
                "s": s,
                "t": t,
                "s_index": i,
                "t_index": j,
            }
            for l, ((i, s), (j, t)) in enumerate(
                product(self.S.items(), self.T.items())
            )
        }

        logger.info("State space augmented with time")

        return S_augmented

    def _enforce_valid_matrices(self) -> None:
        """
        Some of the states are not accessible given certain actions, so the rows in
        the transition matrix will be all zeros. We need to enforce that the
        transition matrix is valid, i.e. that the rows sum to 1.

        Wherever we place the 1 in the transition matrix (on the  diagonal), we also
        need to place an infeasible reward in the reward matrix.
        """

        # If transitions or rewards are an empty list, throw an error
        if not self.transitions or not self.rewards:
            raise ValueError("Transitions and rewards must be non-empty lists")

        new_transitions = []
        new_rewards = []

        for P, R in zip(self.transitions, self.rewards):
            # Sum the rows of the transition matrix
            row_sums = P.sum(axis=1)

            # Find where the row sums are zero
            zero_rows = np.where(row_sums == 0)[0]

            # Construct a new csr matrix with the same shape as the original, but
            # with a 1 on the diagonal where the row sums are zero
            new_P = csr_matrix(
                (np.ones_like(zero_rows), (zero_rows, zero_rows)),
                shape=P.shape,
                dtype=float,
            )

            # Construct a new reward matrix with infeasible rewards where the row
            # sums are zero
            new_R = csr_matrix(
                (
                    np.ones_like(zero_rows) * self.INFEASIBLE,
                    (zero_rows, zero_rows),
                ),
                shape=P.shape,
                dtype=float,
            )

            new_P = P + new_P

            # Normalize so rows sum to 1
            new_P = preprocessing.normalize(new_P, norm="l1", axis=1)

            # Add the new matrices to the list
            new_transitions.append(new_P)
            new_rewards.append(R + new_R)

        self.transitions = new_transitions
        self.rewards = new_rewards

        return None

    def build_transitions_and_rewards_from_function(
        self,
        transition_function: Callable,
    ) -> None:
        """
        Build the transition and reward matrices from a transition function.

        :params

        transition_function: A function that takes arguments (s_prime,s, a, t)
        and returns the probability of transitioning to state s_prime from
        state s
        """

        # Construct a data frame with columns, s, t, a, s_prime, t_prime
        df = (
            pd.DataFrame.from_dict(
                self.S_augmented, orient="index"
            ).reset_index()  # Essential to get the indices as a column
        )

        # Create a copy of the data frame, but to every column name add the
        # suffix '_prime' We will use this to do a self-join to get transitions
        # from each state to each other state
        df_prime = df.copy()
        df_prime.columns = [f"{col}_prime" for col in df.columns]

        # Create a cartesian product of the two data frames
        # Additionally, we only want to compute the transition probabilities and payoffs
        # for feasible states. These are states where t_prime = t + 1
        state_combinations_df = df.merge(df_prime, how="cross").query(
            "t_prime == t + 1"
        )

        # --------------------------------------------------------------------#

        transitions = []
        rewards = []

        for a in self.A.values():
            state_combinations_df["a"] = a

            # Apply the transition function to every row in the data frame
            (
                state_combinations_df["reward"],
                state_combinations_df["probability"],
            ) = zip(
                *state_combinations_df.apply(
                    lambda row: transition_function(
                        row["s_prime"],
                        row["s"],
                        row["t"],
                        row["a"],
                    ),
                    axis=1,
                )
            )

            # Fill nas with infeasible values for rewards
            state_combinations_df["reward"] = state_combinations_df[
                "reward"
            ].fillna(self.INFEASIBLE)

            # Create a sparse matrix with dimension len(S_augmented) x len(S_augmented)
            P = csr_matrix(
                (
                    state_combinations_df["probability"],
                    (
                        state_combinations_df["index"],
                        state_combinations_df["index_prime"],
                    ),
                ),
                shape=(len(self.S_augmented), len(self.S_augmented)),
                dtype=float,
            )
            R = csr_matrix(
                (
                    state_combinations_df["reward"],
                    (
                        state_combinations_df["index"],
                        state_combinations_df["index_prime"],
                    ),
                ),
                shape=(len(self.S_augmented), len(self.S_augmented)),
                dtype=float,
            )

            transitions.append(P)
            rewards.append(R)

        self.transitions = transitions
        self.rewards = rewards

        self._enforce_valid_matrices()

        logger.info("Transition and reward matrices built")

        return None

    def solve(self, algorithm: str, **kwargs) -> None:
        """
        Solve the MDP using the specified algorithm.

        :params

        algorithm: The algorithm to use for solving the MDP. Can be either
        'value_iteration' or 'finite_horizon'
        """

        # If N is not in kwargs, set it to the maximum time period
        if "N" not in kwargs.keys():
            kwargs["N"] = len(self.T)
        if "discount" not in kwargs.keys():
            kwargs["discount"] = self.gamma

        if algorithm == "value_iteration":
            vi = ValueIteration(
                self.transitions,
                self.rewards,
                **kwargs,
            )
            vi.run()
            self.policy_function = vi.policy
            self.value_function = vi.V

        elif algorithm == "finite_horizon":
            fh = FiniteHorizon(
                self.transitions,
                self.rewards,
                **kwargs,
            )
            fh.run()
            self.policy_function = fh.policy
            self.value_function = fh.V

        else:
            raise ValueError(
                "Algorithm must be either 'value_iteration' or 'finite_horizon'"
            )

        logger.info(f"MDP solved using {algorithm}")

        self._extract_feasible_states()

        return None

    def _extract_feasible_states(self) -> None:
        """
        The policy/value function will be of shape (len(S_augmented), len(T) - 1).
        Note that the final time period is a dummy time period.

        Rows are states, columns are time periods.

        We want to create a dictionary where the keys are times, and the values are
        maps of states to actions. Only states where the time value equals the
        current time are feasible.
        """

        policy, value = self.policy_function, self.value_function

        # Correct the shape
        policy = policy[:, 1:]
        value = value[:, 1:-1]

        policy_dict = {}
        value_dict = {}

        # Check that at least one of policy or value is not None
        if policy is None and value is None:
            raise ValueError(
                "At least one of policy or value must be not None."
            )

        # Check that shapes are correct.
        if policy is not None:
            if (
                policy.shape[0] != len(self.S_augmented)
                or policy.shape[1] != len(self.T) - 1
            ):
                logger.error(
                    f"Policy shape is incorrect. Expected shape {(len(self.S_augmented), len(self.T) - 1)}, got {policy.shape}"
                )
                raise ValueError("Policy shape is incorrect.")

            for time_ix, time in self.T.items():
                # Skip the last time index
                if time_ix == max(self.T.keys()):
                    continue

                time_dict = {}
                for state_ix, state in self.S_augmented.items():
                    # If the time index is the same as the current time, add the state to the policy dict.
                    if state["t_index"] == time_ix:
                        time_dict[state["s"]] = {
                            "action": self.A[policy[state_ix, time_ix]]
                        }

                policy_dict[time] = time_dict

        if value is not None:
            if (
                value.shape[0] != len(self.S_augmented)
                or value.shape[1] != len(self.T) - 1
            ):
                logger.error(
                    f"Value shape is incorrect. Expected shape {(len(self.S_augmented), len(self.T) - 1)}, got {value.shape}"
                )
                raise ValueError("Value shape is incorrect.")

            for time_ix, time in self.T.items():
                # Skip the last time index
                if time_ix == max(self.T.keys()):
                    continue

                time_dict = {}
                for state_ix, state in self.S_augmented.items():
                    # If the time index is the same as the current time, add the state to the policy dict.
                    if state["t_index"] == time_ix:
                        time_dict[state["s"]] = float(value[state_ix, time_ix])

                value_dict[time] = time_dict

        self.policy_function, self.value_function = policy_dict, value_dict

        return None

    def discretize_state_space(
        self,
        num_points: int,
    ) -> tuple[list[float], dict[int, dict[str, float]]]:
        """
        If we need to discretize the interval [0, 10] inte 5 points, we get:
        [0, 2, 4, 6, 8, 10]
        """

        # assert that state space has more than one element
        assert len(self.S) > 1, "State space must have more than one element"

        # assert state space values are numeric
        assert all(isinstance(s, (int, float)) for s in self.S.values()), (
            "State space values must be numeric"
        )

        start = self.S[0]
        end = self.S[len(self.S) - 1]

        assert start < end, "Start must be less than end"

        assert num_points > 1, "Number of points must be greater than 1"

        grid = np.linspace(start, end, num_points).tolist()

        # Round these to 3 decimal places
        grid = [round(x, 3) for x in grid]

        self.S = {i: s for i, s in enumerate(grid)}
        self.inverse_S = {v: k for k, v in self.S.items()}

        logger.info("State space discretized.")
        logger.info(f"State space: {self.S}")

        # Generate boundaries for each pair of points Create a dictionary where
        # the key is the key of the state in self.S, and the value is (lower,
        # upper). E.g., if our grid is [0, 2, 4], then the boundaries are
        # {0: {'lower': -inf, 'upper': 1.5},
        #  1: {'lower': 1.5, 'upper': 3.5},
        #  2: {'lower': 3.5, 'upper': inf}}

        midpoints = [(a + b) / 2 for a, b in zip(grid, grid[1:])]
        midpoints = [-np.inf] + midpoints + [np.inf]
        boundaries = {
            i: {"lower": midpoints[i], "upper": midpoints[i + 1]}
            for i in range(len(midpoints) - 1)
        }

        # augment state space with time
        self.S_augmented = self._augment_state_space_with_time()

        return grid, boundaries

    def build_transitions_and_rewards_from_model(
        self,
        model: Callable,
        static_vars: dict[str, Any] | None = None,
    ) -> None:
        """
        Build the transition and reward matrices from a model. The model object
        should have  a '.predict()' method.

        The predict method will operate on a data frame that has the columns,
        s, t, a, and any vars in static_vars.

        Its output should be an array of probabilities for transitioning into
        the next state.
        """

        if static_vars is None:
            features = ["s", "t", "a"]
        else:
            features = ["s", "t", "a"] + list(static_vars.keys())

        transitions = []
        rewards = []

        # Data frame with the set of predictors we need for model.
        # Should have columns s, t, a.
        df = pd.DataFrame().from_dict(self.S_augmented, orient="index")

        # Well need to self-join to get all possible next states we can transition to.
        join_df = (
            pd.DataFrame()
            .from_dict(self.S_augmented, orient="index")[
                ["s", "s_index", "t_index"]
            ]
            .reset_index()
            .rename(
                columns={
                    "index": "index_prime",
                    "s_index": "s_prime_index",
                    "t_index": "t_prime_index",
                    "s": "s_prime",
                }
            )
        )

        # Add to this column the static variables
        if static_vars is not None:
            df = df.assign(**static_vars)

        for a in self.A.values():
            inference_df = df.copy()

            inference_df["a"] = a
            probabilities = model.predict_proba(inference_df[features])

            # Round these probabilities to 4 dp
            probabilities = np.round(probabilities, 4)

            P = pd.DataFrame(probabilities, index=self.S_augmented.keys())

            # Currently prob is a list. We wantt to make a list of tuples. Ie, if prob has [3, 4]
            # let's create [(0, 3), (1, 4)]
            P["prob"] = P.apply(lambda x: list(zip(range(len(x)), x)), axis=1)
            P = P[["prob"]]

            # Join this to inference_df
            P = P.join(inference_df, how="inner")
            P = P.reset_index().rename(columns={"index": "index"})

            # The only valid transitions are to the next time step.
            P["t_prime"] = P["t"] + 1
            P["t_prime_index"] = P["t_index"] + 1

            # Explode prob
            P = P.explode("prob")

            # prob is now a tuple. Let's make it two columns called s_prime_index and probability
            P["s_prime_index"] = P["prob"].apply(lambda x: x[0])
            P["probability"] = P["prob"].apply(lambda x: x[1])

            P = P.merge(join_df, on=["s_prime_index", "t_prime_index"])

            P["reward"] = P["a"] * (P["s"] - P["s_prime"])

            P = P[["index", "index_prime", "reward", "probability"]]

            # Build a transition matrix that is csr sparse and fills index, index_prime
            # with probability and has shape len(S_augmented)

            R = csr_matrix(
                (P["reward"], (P["index"], P["index_prime"])),
                shape=(len(self.S_augmented), len(self.S_augmented)),
            )
            P = csr_matrix(
                (P["probability"], (P["index"], P["index_prime"])),
                shape=(len(self.S_augmented), len(self.S_augmented)),
            )

            transitions.append(P)
            rewards.append(R)

        self.transitions = transitions
        self.rewards = rewards

        self._enforce_valid_matrices()

        logger.info("Transition and reward matrices built")

        return None

    def get_transition_probabilties(
        self,
        s: int | str,
        t: int,
        a: int | str,
    ) -> dict[str, Any]:
        """
        Plot the transition probabilities at time step t for action a.
        """

        # Checks on the inputs
        assert s in self.S.values(), "State must be a valid value in S"

        # Check t is a valid value for time
        assert t in self.T.values(), "Time must be a valid value in T"

        # Check that t is not the last time period
        assert t != max(self.T.values()), (
            "Cannot plot transition probabilities for the last time period"
        )

        # Check a is a valid action
        assert a in self.A.values(), "Action must be a valid action in A"

        # What are the indices of t and a?
        t_ix = self.inverse_T[t]
        a_ix = self.inverse_A[a]

        # Get the transition matrix for action a
        P = self.transitions[a_ix]
        R = self.rewards[a_ix]

        # Which row of the transition matrix corresponds to state s at time t?
        s_current = [
            k
            for k, v in self.S_augmented.items()
            if v["s"] == s and v["t"] == t
        ]

        # Check that this has exactly one element and extract it
        assert len(s_current) == 1, "State s at time t must be unique"
        s_current_ix = s_current[0]

        # For S_augmented, find which states have time index t_ix + 1
        s_prime_ix = [
            k for k, v in self.S_augmented.items() if v["t_index"] == t_ix + 1
        ]

        s_prime_labels = [self.S_augmented[ix]["s"] for ix in s_prime_ix]

        # Pull out the s_current_ix row of the transition matrix and the s_prime_ix columns
        probs = P[s_current_ix, :].toarray().flatten()[s_prime_ix]

        rewards = R[s_current_ix, :].toarray().flatten()[s_prime_ix]

        return {
            "current_state": s,
            "current_time": t,
            "action": a,
            "next_states": s_prime_labels,
            "probabilities": probs,
            "rewards": rewards,
        }

    def plot_matrix(
        self,
        matrix_type: str,
        t: int,
        a: int | str,
        ax: plt.Axes | None = None,
        annotate: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """
        Plot the transition probabilities at time step t for action a.

        Use a seaborn heatmap where the y axis is s and the x axis is s_prime.
        """

        assert matrix_type in ["transitions", "rewards"], (
            "Type must be either 'transitions' or 'rewards'"
        )
        assert t in self.T.values(), "Time must be a valid value in T"
        assert a in self.A.values(), "Action must be a valid action in A"

        # empty np array
        matrix = np.zeros((len(self.S), len(self.S)))

        y_labels = list(self.S.values())

        for i, s in enumerate(y_labels):
            data = self.get_transition_probabilties(s, t, a)
            if matrix_type == "transitions":
                matrix[i, :] = data["probabilities"]
            if matrix_type == "rewards":
                matrix[i, :] = data["rewards"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        # Plot the heatmap
        sns.heatmap(
            matrix,
            xticklabels=y_labels,
            yticklabels=y_labels,
            ax=ax,
            annot=annotate,
            linewidths=0.5,
            linecolor="black",
            cmap="coolwarm",
            alpha=0.7,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Next State")
        ax.set_ylabel("Current State")
        ax.set_title(f"{matrix_type.title()} at Time {t} for Action {a}")

        # Reduce the number of labels to at most 5
        max_labels = 5
        if max_labels is not None:
            x_labels = ax.get_xticks()
            y_labels = ax.get_yticks()
            ax.set_xticks(x_labels[:: max(1, len(x_labels) // max_labels)])
            ax.set_yticks(y_labels[:: max(1, len(y_labels) // max_labels)])

        return None

    def plot_policy(
        self,
        t: int,
        ax: plt.Axes | None = None,
    ) -> None:
        """
        Plot the policy at time t. The policy is a mapping from states to actions.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        policy = self.policy_function[t]

        # Get the states and actions
        states = list(policy.keys())
        actions = [policy[s]["action"] for s in states]

        # Plot the policy
        ax.bar(states, actions)
        ax.set_xlabel("State")
        ax.set_ylabel("Action")
        ax.set_title(f"Policy at Time {t}")

        fig.plot()

        return None


# ----------------------------------------------------------------------------#
