import numpy as np
import pandas as pd

from typing import Any

from scipy.sparse import csr_matrix

from sklearn import preprocessing

from itertools import product

from mdptoolbox.mdp import FiniteHorizon

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
    """

    def __init__(self):
        self.transitions: list[csr_matrix] = []
        self.rewards: list[csr_matrix] = []

        self.policy_function: list[np.ndarray] = []
        self.value_function: list[np.ndarray] = []

        # Payoff to assign infeasible transitions
        self.INFEASIBLE = -1e8

        return None

    def sanity_check(self):
        # Check self.S exists
        assert hasattr(self, "S"), "State space must be defined"
        assert hasattr(self, "A"), "Action space must be defined"
        assert hasattr(self, "T"), "Time space must be defined"
        assert hasattr(self, "reward"), "Reward function must be defined"

        assert len(self.S) > 0, "State space must have at least one element"
        assert len(self.A) > 0, "Action space must have at least one element"
        assert len(self.T) > 0, "Time space must have at least one element"

        # Check that S and A are sets, T is a list
        assert isinstance(self.S, list), "State space must be a list"
        assert isinstance(self.A, list), "Action space must be a list"
        assert isinstance(self.T, list), "Time space must be a list"

        return None

    def _augment_state_space_with_time(self) -> None:
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

        # Add a dummy time period to the time space
        self.T.append(max(self.T) + 1)

        self.S_augmented = [(s, t) for s, t in product(self.S, self.T)]

        # We need these for filling in the transition and reward matrices.
        self.augmented_state_to_index = {
            state_tuple: idx
            for idx, state_tuple in enumerate(self.S_augmented)
        }

        logger.info("State space augmented with time")

        return None

    def create_data_frame(self) -> None:
        """
        Creates an attribute called `data` that is a data frame containing
        that has columns s_prime, s, a, t, t_prime.

        """

        self.sanity_check()

        self._augment_state_space_with_time()

        # Create a data frame that is a cross product:
        # S x S x A x T
        df = pd.DataFrame(
            product(self.S, self.A, self.T), columns=["s", "a", "t"]
        )

        df["s_prime"] = [self.S] * len(df)
        df = df.explode("s_prime")
        df = df[["s_prime", "s", "a", "t"]]

        # Add a column called t_prime equal to t + 1. Filter down to rows
        df["t_prime"] = df["t"] + 1

        # Filter down to rows where t is strictly less than the maximum time period.
        df = df[df["t"] < max(self.T)]

        df = df.sort_values(by=["t", "a", "s", "s_prime"])

        # Set this as an attribute.
        self.data = df

        return None

    def compute_transitions_and_rewards(self):
        """
        Uses provided transition and reward functions to populate the
        data frame with transition probabilities and rewards.
        """

        # Create a column called 'probability' that is the result of applying
        # the transition function.
        self.data["probability"] = self.data.apply(
            lambda x: self.transition(x["s_prime"], x["s"], x["t"], x["a"]),
            axis=1,
        )

        # Create a column called 'reward' that is the result of applying the
        # reward function.
        self.data["reward"] = self.data.apply(
            lambda x: self.reward(x["s_prime"], x["s"], x["t"], x["a"]), axis=1
        )

        return None

    def compute_transitions_and_rewards_vectorized(self):
        """
        Uses provided transition and reward functions to populate the
        data frame with transition probabilities and rewards.
        """

        # Create a column called 'probability' that is the result of applying
        # the transition function.
        self.data["probability"] = self.transition(
            self.data["s_prime"],
            self.data["s"],
            self.data["t"],
            self.data["a"],
        )

        # Create a column called 'reward' that is the result of applying the
        # reward function.
        self.data["reward"] = self.reward(
            self.data["s_prime"],
            self.data["s"],
            self.data["t"],
            self.data["a"],
        )

        return None

    def _build_matrix(
        self,
        df: pd.DataFrame,
    ) -> csr_matrix:
        """
        Given a data frame with columns s_prime, s, t, t_prime, probability,
        and reward, build a transition or reward matrix.

        Parameters:
        -----------
        df: pd.DataFrame
            A data frame with columns s_prime, s, t, t_prime, probability, and
            reward.

        Returns:
        --------
        csr_matrix:
            A sparse matrix representing the transition or reward matrix.
        """

        # Each row has s_prime, s, t, t_prime. Add two new columnns,
        # s_augmented_index and s_prime_augmented_index. These are the indices
        # of the augmented state space for tuples (s,t) and (s_prime, t_prime).

        # Create new columns by applying the mapping
        df.loc[:, "s_augmented_index"] = df.apply(
            lambda row: self.augmented_state_to_index.get(
                (row["s"], row["t"]), -1
            ),
            axis=1,
        )
        df.loc[:, "s_prime_augmented_index"] = df.apply(
            lambda row: self.augmented_state_to_index.get(
                (row["s_prime"], row["t_prime"]), -1
            ),
            axis=1,
        )

        # Create a sparse matrix with dimension len(S_augmented) x
        # len(S_augmented). Note that we need the indices in S_augmented
        # corresponding to (s, t) and (s_prime, t_prime).
        # Create a column called 'index' for what the index of (s,t) is.

        P = csr_matrix(
            (
                df["probability"],
                (
                    df["s_augmented_index"],
                    df["s_prime_augmented_index"],
                ),
            ),
            shape=(len(self.S_augmented), len(self.S_augmented)),
            dtype=float,
        )

        R = csr_matrix(
            (
                df["reward"],
                (
                    df["s_augmented_index"],
                    df["s_prime_augmented_index"],
                ),
            ),
            shape=(len(self.S_augmented), len(self.S_augmented)),
            dtype=float,
        )

        return P, R

    def build_matrices(self):
        """
        Populate the transitions and rewards attributes with sparse arrays
        representing the transition probabilities and rewards for each action.
        """

        # Filter out rows where probability is zero
        self.data = self.data.query("probability > 0")

        # Each row has s_prime, s, t, t_prime. Add two new columnns,
        # s_augmented_index and s_prime_augmented_index. These are the indices
        # of the augmented state space for tuples (s,t) and (s_prime, t_prime).
        # We need these for filling in the transition and reward matrices.
        augmented_state_to_index = {
            state_tuple: idx
            for idx, state_tuple in enumerate(self.S_augmented)
        }

        # Create new columns by applying the mapping
        self.data["s_augmented_index"] = self.data.apply(
            lambda row: augmented_state_to_index.get((row["s"], row["t"]), -1),
            axis=1,
        )
        self.data["s_prime_augmented_index"] = self.data.apply(
            lambda row: augmented_state_to_index.get(
                (row["s_prime"], row["t_prime"]), -1
            ),
            axis=1,
        )

        for action in self.A:
            # Filter down to rows where the action is the current action
            df = self.data.query("a == @action")

            # Create a sparse matrix with dimension len(S_augmented) x
            # len(S_augmented). Note that we need the indices in S_augmented
            # corresponding to (s, t) and (s_prime, t_prime).
            # Create a column called 'index' for what the index of (s,t) is.

            P = csr_matrix(
                (
                    df["probability"],
                    (
                        df["s_augmented_index"],
                        df["s_prime_augmented_index"],
                    ),
                ),
                shape=(len(self.S_augmented), len(self.S_augmented)),
                dtype=float,
            )

            R = csr_matrix(
                (
                    df["reward"],
                    (
                        df["s_augmented_index"],
                        df["s_prime_augmented_index"],
                    ),
                ),
                shape=(len(self.S_augmented), len(self.S_augmented)),
                dtype=float,
            )

            self.transitions.append(P)
            self.rewards.append(R)

        return None

    def _enforce_valid_matrices(self):
        """
        Sometimes the transition matrices don't have rows summing to 1. This is
        either because there are inaccessible states, or there are rounding
        errors.

        First check for rows with all zeros. If there are any, set the diagonal
        to 1 and set the corresponding entry in rewards to INFEASIBLE.

        Then, check for rows that don't sum to 1. If there are any, normalize
        the row so that it sums to 1.
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
            adjustment_P = csr_matrix(
                (np.ones_like(zero_rows), (zero_rows, zero_rows)),
                shape=P.shape,
                dtype=float,
            )

            # Construct a new reward matrix with infeasible rewards where the row
            # sums are zero
            adjustment_R = csr_matrix(
                (
                    np.ones_like(zero_rows) * self.INFEASIBLE,
                    (zero_rows, zero_rows),
                ),
                shape=P.shape,
                dtype=float,
            )

            new_P = P + adjustment_P
            new_R = R + adjustment_R

            # Normalize so rows sum to 1
            new_P = preprocessing.normalize(new_P, norm="l1", axis=1)

            # Add the new matrices to the list
            new_transitions.append(new_P)
            new_rewards.append(new_R)

        self.transitions = new_transitions
        self.rewards = new_rewards

        return None

    def _solver(self):
        """
        This is the method that actually does the solving using PyMDPToolbox.
        """

        self._enforce_valid_matrices()

        n = len(self.T)

        # Solve the finite horizon MDP
        mdp = FiniteHorizon(
            self.transitions,
            self.rewards,
            1,
            n,
        )

        mdp.run()

        # ---------------------------------------------------------------------#
        # The value function has an entry for each (s,t) in S_augmented.
        # We need to filter down to the entries where t is not the dummy time
        # period

        self.value_function = mdp.V
        self.policy_function = mdp.policy

        self._extract_feasible_states()

        logger.info("MDP solved")

    def solve(self):
        """
        This method runs all the steps using the non-vectorized evaluations
        of the transition and reward functions.
        """
        self.create_data_frame()
        self.compute_transitions_and_rewards()
        self.build_matrices()
        self._solver()

        return None

    def solve_vectorized(self):
        """
        This method runs all the steps using the vectorized evaluations
        of the transition and reward functions.
        """
        self.create_data_frame()
        self.compute_transitions_and_rewards_vectorized()
        self.build_matrices()
        self._solver()

        return None

    def solve_model(self, model):
        """
        Use a model object to solve.
        This object should have a .predict_proba(df) method.
        """

        self.create_data_frame()

        # To preserve memory, remove the s_prime column and drop duplicates
        self.data = self.data.drop(columns=["s_prime"]).drop_duplicates()

        transitions = []
        rewards = []
        for a in self.A:
            # Filter down to the subset of df with action a
            df_a = self.data.query("a == @a").copy()

            assert len(df_a) > 0, f"No data for action {a}"

            X = df_a[model.feature_names_]
            probabilities = model.predict_proba(X)

            # Round the probabilities to 4 decimal places
            probabilities = np.round(probabilities, 4)

            # Add a column called s_prime
            df_a["s_prime"] = [self.S] * len(df_a)

            # Add the probabilities
            df_a["probability"] = probabilities.tolist()

            # Now we need to explode the probability column
            df_a = df_a.explode(["probability", "s_prime"])

            # Now filter out any rows where probability is within 1e-4 of 0
            df_a = df_a.query("probability > 1e-4")

            # Compute reward
            df_a.loc[:, "reward"] = self.reward(
                df_a["s_prime"], df_a["s"], df_a["t"], df_a["a"]
            )

            # Contruct the transition and reward matrices
            P, R = self._build_matrix(df_a)

            # Normalize these matrices
            P = preprocessing.normalize(P, norm="l1", axis=1)

            transitions.append(P)
            rewards.append(R)

        # Set the transitions and rewards attributes
        self.transitions = transitions
        self.rewards = rewards

        # Solve the MDP
        self._solver()

        return None

    def _extract_feasible_states(self) -> None:
        """
        The value function will be of shape (len(S_augmented), len(T)).

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

        # Policy first
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

        if (
            value.shape[0] != len(self.S_augmented)
            or value.shape[1] != len(self.T) - 1
        ):
            logger.error(
                f"Value shape is incorrect. Expected shape {(len(self.S_augmented), len(self.T) - 1)}, got {value.shape}"
            )
            raise ValueError("Value shape is incorrect.")

        for time_ix, time in enumerate(self.T[:-1]):
            time_dict_value = {}
            time_dict_policy = {}
            for state_ix, state in enumerate(self.S_augmented):
                # If the time index is the same as the current time, add the state to the policy dict.
                if state[1] == time:
                    time_dict_value[state[0]] = float(value[state_ix, time_ix])
                    time_dict_policy[state[0]] = self.A[
                        policy[state_ix, time_ix]
                    ]

            value_dict[time] = time_dict_value
            policy_dict[time] = time_dict_policy

        self.policy_function = policy_dict
        self.value_function = value_dict

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
        assert s in self.S, "State must be a valid value in S"

        # Check t is a valid value for time
        assert t in self.T, "Time must be a valid value in T"

        # Check that t is not the last time period
        assert t != max(self.T), (
            "Cannot plot transition probabilities for the last time period"
        )

        # Check a is a valid action
        assert a in self.A, "Action must be a valid action in A"

        # What is the index of a?
        a_ix = self.A.index(a)

        # Get the transition matrix for action a
        P = self.transitions[a_ix]
        R = self.rewards[a_ix]

        # Which row of the transition matrix corresponds to state s at time t?
        s_current = [
            k
            for k, v in enumerate(self.S_augmented)
            if v[0] == s and v[1] == t
        ]

        # Check that this has exactly one element and extract it
        assert len(s_current) == 1, "State s at time t must be unique"
        s_current_ix = s_current[0]

        # For S_augmented, find which states have time index t_ix + 1
        s_prime_ix = [
            k for k, v in enumerate(self.S_augmented) if v[1] == t + 1
        ]
        s_prime_labels = [self.S_augmented[ix][0] for ix in s_prime_ix]

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
        assert t in self.T, "Time must be a valid value in T"
        assert a in self.A, "Action must be a valid action in A"

        # empty np array
        matrix = np.zeros((len(self.S), len(self.S)))

        y_labels = self.S

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
