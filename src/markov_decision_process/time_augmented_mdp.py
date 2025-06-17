import numpy as np
import polars as pl
import pandas as pd
import os

from typing import Any, Callable, List, Tuple, Literal

from scipy.sparse import csr_matrix
from scipy.sparse import vstack

from sklearn import preprocessing
from sklearn.isotonic import IsotonicRegression

from itertools import product

from mdptoolbox.mdp import FiniteHorizon

import matplotlib.pyplot as plt
import seaborn as sns

from .utilities import (
    monotonic_discrete_fit,
    get_policy,
)

import inspect
import copy
import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class TimeAugmentedMDP:
    def __init__(
        self,
        states: List[int] | List[float] | List[str] | List[Tuple],
        actions: List[int] | List[float] | List[str] | List[Tuple],
        times: List[int],
        reward_function: Callable,
        transition_function: Callable | None = None,
        discount_factor: float = 1,
        mode: Literal["flexible", "vectorized", "model"] | None = None,
        model: Callable | None = None,
        state_space_data_path: str | None = None,
        force_overwrite: bool = False,
        static_features: dict[str, Any] | None = None,
    ):
        # ---------------------------------------------------------------------
        # Assignment and sanity checks

        # Assign these inputs to self
        for key, value in locals().items():
            if key != "self":
                # Set a copy, otherwise we'll mutate the original
                setattr(self, key, copy.deepcopy(value))

        # Transitions and rewards will be constructed by methods below
        self.transition_matrices: List[csr_matrix] = []
        self.reward_matrices: List[csr_matrix] = []
        self.INFEASIBLE = -1e8

        self.discount_factor = discount_factor

        # Value and policy functions
        self.value_function: dict[int, dict[int, float]] | None = None
        self.policy_function: dict[int, dict[int, int]] | None = None
        self.value_monotone: dict[Tuple, float] | None = None
        self.policy_monotone: dict[int, dict[int, int]] | None = None

        # The policy above will be filtered down to a dictionary of
        # feasible states and times. This will be the full policy
        # across augmented states.
        self.policy_function_augmented = None

        # Augmented states
        self.states_augmented: List[Tuple] = []
        self.augmented_state_to_index: dict = {}

        # If model is passed, set the model to be model
        if self.model is not None:
            logger.info("Model provided. Setting mode to 'model'")
            self.mode = "model"

        # Checks on inputs
        self.__check_inputs()

        # If the mode is not set, try to infer it.
        if self.mode is None:
            logger.info("Mode not set. Inferring mode...")
            self.__infer_mode()

        # ---------------------------------------------------------------------
        # Augmentation

        # Add a dummy time period to the time space
        self.times.append(max(self.times) + 1)

        # Create an list of augmented states that is a tuple of the
        # state and time period. Create an index mapping for these states.
        self.states_augmented, self.augmented_state_to_index = (
            self.__augment_state_space_with_time()
        )

        # ---------------------------------------------------------------------

        # Generate the state space data
        self.state_space_data = self.__generate_state_space_data()

        return None

    def __check_inputs(self):
        # States must be non-empty
        if not self.states:
            raise ValueError("States must be non-empty")
        # States must be a list where all elements are of the same type
        if not all(
            isinstance(state, type(self.states[0])) for state in self.states
        ):
            raise ValueError(
                "States must be a list where all elements are of the same type"
            )
        # States must be integers, floats, or strings
        if not all(
            isinstance(state, (int, float, str)) for state in self.states
        ):
            raise ValueError("States must be integers, floats, strings")

        # Same for actions
        if not self.actions:
            raise ValueError("Actions must be non-empty")
        if not all(
            isinstance(action, type(self.actions[0]))
            for action in self.actions
        ):
            raise ValueError(
                "Actions must be a list where all elements are of the same type"
            )
        if not all(
            isinstance(action, (int, float, str)) for action in self.actions
        ):
            raise ValueError("Actions must be integers, floats, strings")

        # Times must be integers
        if not all(isinstance(time, int) for time in self.times):
            raise ValueError("Times must be integers")
        # Times must be non-empty
        if not self.times:
            raise ValueError("Times must be non-empty")
        # Times must be consecutive
        if not all(
            self.times[i] == self.times[i - 1] + 1
            for i in range(1, len(self.times))
        ):
            raise ValueError("Times must be consecutive and increasing")

        # Mode must be one of the allowed values
        if self.mode is not None:
            if self.mode not in ["flexible", "vectorized", "model"]:
                raise ValueError(
                    'Mode must be one of "flexible", "vectorized", or "model"'
                )
        # If mode is flexible or vectorized, a transition function must be provided
        if self.mode in ["flexible", "vectorized"] and not callable(
            self.transition_function
        ):
            raise ValueError(
                'If mode is "flexible" or "vectorized", a transition function must be provided'
            )
        if self.mode is None and not callable(self.transition_function):
            raise ValueError(
                "Either a model or a transition function must be provided"
            )

        # Check that reward function has correct arguments
        reward_signature = list(
            inspect.signature(self.reward_function).parameters.keys()
        )
        if reward_signature != ["s_prime", "s", "a", "t"]:
            logger.warning(
                f"Reward function currently has arguments {reward_signature}. Should be (s_prime, s, a, t)"
            )
            raise ValueError(
                "Reward function should have arguments (s_prime, s, a, t)"
            )

        # If the transition_function is provided, check that it has the correct arguments
        if self.transition_function is not None:
            transition_signature = list(
                inspect.signature(self.transition_function).parameters.keys()
            )
            if transition_signature != ["s_prime", "s", "a", "t"]:
                logger.warning(
                    f"Transition function currently has arguments {transition_signature}. Should be (s_prime, s, a, t)"
                )
                raise ValueError(
                    "Transition function should have arguments (s_prime, s, a, t)"
                )

        return None

    def __is_func_vectorized(self, func):
        # Create test series inputs
        s_prime = pd.Series([1, 2, 3])
        s = pd.Series([1, 2, 3])
        a = pd.Series([1, 2, 3])
        t = pd.Series([1, 2, 3])

        try:
            # Try to apply the function to the series inputs
            result = func(s_prime, s, a, t)

            # Check if the result is a pl.Series
            if isinstance(result, pd.Series):
                return True
            else:
                return False
        except Exception:
            # If an exception occurs, it's likely not vectorized
            return False

    def __infer_mode(self):
        """
        Checks if the reward and transition functions are vectorized. If they
        are, sets mode to 'vectorized'. Otherwise, sets mode to 'flexible'.
        """

        # Is reward vectorized
        reward_vectorized = self.__is_func_vectorized(self.reward_function)
        logger.info(f"Reward function is vectorized: {reward_vectorized}")

        # Only inferring mode if we don't already have a model passed.
        # So transition function must exist.
        transition_vectorized = self.__is_func_vectorized(
            self.transition_function
        )
        logger.info(
            f"Transition function is vectorized: {transition_vectorized}"
        )

        if reward_vectorized and transition_vectorized:
            logger.info(
                'Reward and transition functions are vectorized. Setting mode to "vectorized"'
            )
            self.mode = "vectorized"
        else:
            logger.info(
                'Reward and transition functions are not vectorized. Setting mode to "flexible"'
            )
            self.mode = "flexible"

        return None

    def __augment_state_space_with_time(self) -> None:
        """ """

        states_augmented = [
            (s, t) for s, t in product(self.states, self.times)
        ]

        # We need these for filling in the transition and reward matrices.
        augmented_state_to_index = {
            state_tuple: idx
            for idx, state_tuple in enumerate(states_augmented)
        }

        logger.info("State space augmented with time")

        return states_augmented, augmented_state_to_index

    def __generate_state_space_data(self):
        """
        Need to create an outer product that depends on the mode.

        If we're in flexible or vectorized mode, we need to generate the
        full cross product:

        S x S x A x T

        If we're in model mode, we need to generate the cross product of
        S x A x T

        Save out this data to disk partitioned by the actions.

        Before we generate data, check if the data already exists on disk.
        If it does, load it in.
        """

        logger.info("Generating state space data...")

        # Check if a path has been passed
        if self.state_space_data_path is not None:
            state_space_path = self.state_space_data_path
        else:
            state_space_path = "/tmp/state_space_data/"
            logger.info("No path provided. Saving to /tmp/state_space_data/")

            # Check if the directory exists
            if os.path.exists(state_space_path) and not self.force_overwrite:
                raise ValueError(
                    f"Directory {state_space_path} already exists. Please provide a new path or delete the existing directory."
                )

        # If force_overwrite is True, delete the directory
        # and recreate it.
        if self.force_overwrite:
            logger.warning(
                "Force overwrite active. Deleting existing directory and recreating..."
            )
            os.system(f"rm -rf {state_space_path}")
            os.makedirs(state_space_path, exist_ok=True)

        # Check if the data already exists on disk
        try:
            data = pl.scan_parquet(state_space_path)
            data.head().collect()
            logger.info("Data found on disk. Loading in...")
            return data
        except:
            logger.info("No data found on disk. Generating...")

        # Create the data

        s_prime = pl.DataFrame({"s_prime": self.states})
        s = pl.DataFrame({"s": self.states})
        t = pl.DataFrame({"t": self.times})

        # Compute the cross join
        if self.mode in ["flexible", "vectorized"]:
            df = s_prime.join(s, how="cross").join(t, how="cross")
        elif self.mode == "model":
            df = s.join(t, how="cross")

        # add in the static features if they are provided
        if self.static_features is not None:
            df = df.with_columns(
                [
                    pl.lit(value).alias(key)
                    for key, value in self.static_features.items()
                ]
            )

        for action in self.actions:
            output_dir = f"{state_space_path}/a={action}"
            output_path = f"{output_dir}/partition.parquet"
            os.makedirs(output_dir, exist_ok=True)

            # Add the action to the DataFrame and save to disk partitioned
            # by the action.
            (
                df.with_columns(pl.lit(action).alias("a")).write_parquet(
                    output_path
                )
            )

        logger.info("State space data generated and saved to disk")

        # Now load in the data
        data = pl.scan_parquet(state_space_path)

        return data

    def __build_matrix(
        self,
        df: pl.DataFrame,
    ) -> csr_matrix:
        """
        Given a data frame with columns s_prime, s, t, t_prime, probability,
        and reward, build a transition or reward matrix.

        Parameters:
        -----------
        df: pl.DataFrame
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
        df = df.with_columns(
            [
                pl.struct(["s", "t"])
                .map_elements(
                    lambda row: self.augmented_state_to_index.get(
                        (row["s"], row["t"]), -1
                    ),
                    return_dtype=pl.UInt64,
                )
                .alias("s_augmented_index")
            ]
        )
        df = df.with_columns(
            [
                pl.struct(["s_prime", "t_prime"])
                .map_elements(
                    lambda row: self.augmented_state_to_index.get(
                        (row["s_prime"], row["t_prime"]), -1
                    ),
                    return_dtype=pl.UInt64,
                )
                .alias("s_prime_augmented_index")
            ]
        )

        # Create a sparse matrix with dimension len(S_augmented) x
        # len(S_augmented). Note that we need the indices in S_augmented
        # corresponding to (s, t) and (s_prime, t_prime).
        # Create a column called 'index' for what the index of (s,t) is.
        df = df.collect()

        P = csr_matrix(
            (
                df["probability"],
                (
                    df["s_augmented_index"],
                    df["s_prime_augmented_index"],
                ),
            ),
            shape=(len(self.states_augmented), len(self.states_augmented)),
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
            shape=(len(self.states_augmented), len(self.states_augmented)),
            dtype=float,
        )

        return P, R

    def build_rewards_and_transitions(self):
        """
        Generate the reward and transition matrices.

        If mode is 'flexible' or 'vectorized', we'll use the reward and
        transition functions to generate these matrices.

        If mode is 'model', we'll use the model to generate these matrices.
        """

        logger.info("Generating rewards and transitions...")

        transitions = []
        rewards = []

        for a in self.actions:
            # Filter down to the subset of df with action a
            df_a = self.state_space_data.filter(pl.col("a") == a)

            if self.mode == "flexible":
                # Use the apply method to apply the transition and reward
                # functions
                df_a = df_a.with_columns(
                    [
                        pl.struct("s_prime", "s", "a", "t")
                        .map_elements(
                            lambda row: self.transition_function(
                                row["s_prime"], row["s"], row["a"], row["t"]
                            ),
                            return_dtype=pl.Float32,
                        )
                        .alias("probability"),
                        pl.struct("s_prime", "s", "a", "t")
                        .map_elements(
                            lambda row: self.reward_function(
                                row["s_prime"], row["s"], row["a"], row["t"]
                            ),
                            return_dtype=pl.Float32,
                        )
                        .alias("reward"),
                    ]
                )

            if self.mode == "vectorized":
                # Use the apply method to apply the transition and reward
                # functions
                # TODO: How can we do this without a collect operation?
                df_a = df_a.collect().to_pandas()

                df_a["probability"] = self.transition_function(
                    df_a["s_prime"], df_a["s"], df_a["a"], df_a["t"]
                )

                df_a["reward"] = self.reward_function(
                    df_a["s_prime"], df_a["s"], df_a["a"], df_a["t"]
                )

                df_a = pl.DataFrame(df_a).lazy()

            # -----------------------------------------------------------------
            # Unclear if this works until tested
            if self.mode == "model":
                # Collect and convert to pandas
                df_a = df_a.collect().to_pandas().copy()

                X = df_a[self.model.feature_names_]
                probabilities = self.model.predict_proba(X)
                # probabilities = np.round(probabilities, 4)

                # Add a column called s_prime
                df_a.loc[:, "s_prime"] = [self.states] * len(df_a)

                # Add the probabilities
                df_a.loc[:, "probability"] = probabilities.tolist()

                # Now we need to explode the probability column
                df_a = df_a.explode(["probability", "s_prime"])

                # Now filter out any rows where probability is within 1e-4 of 0
                df_a = df_a.query("probability > 1e-4")

                # Compute reward
                df_a.loc[:, "reward"] = self.reward_function(
                    df_a["s_prime"], df_a["s"], df_a["a"], df_a["t"]
                )

                df_a = pl.DataFrame(df_a).lazy()
            # -----------------------------------------------------------------
            # Add a columns t_prime which is t + 1
            df_a = df_a.with_columns([(pl.col("t") + 1).alias("t_prime")])

            # Only retain columns where t_prime <= max(times)
            df_a = df_a.filter(pl.col("t_prime") <= max(self.times))

            # Contruct the transition and reward matrices
            P, R = self.__build_matrix(df_a)

            transitions.append(P)
            rewards.append(R)

        self.transition_matrices = transitions
        self.reward_matrices = rewards

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
        if not self.transition_matrices or not self.reward_matrices:
            raise ValueError("Transitions and rewards must be non-empty lists")

        new_transitions = []
        new_rewards = []

        for P, R in zip(self.transition_matrices, self.reward_matrices):
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

        self.transition_matrices = new_transitions
        self.reward_matrices = new_rewards

        return None

    def solve(self):
        self.build_rewards_and_transitions()
        self._enforce_valid_matrices()

        n = len(self.times)

        # Solve the finite horizon MDP
        mdp = FiniteHorizon(
            self.transition_matrices,
            self.reward_matrices,
            self.discount_factor,
            n,
        )

        mdp.run()

        # ---------------------------------------------------------------------#
        # The value function has an entry for each (s,t) in S_augmented.
        # We need to filter down to the entries where t is not the dummy time
        # period

        self.value_function = mdp.V
        self.policy_function = mdp.policy
        self.policy_function_augmented = mdp.policy

        self._extract_feasible_states()

        logger.info("MDP solved")

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
                policy.shape[0] != len(self.states_augmented)
                or policy.shape[1] != len(self.times) - 1
            ):
                logger.error(
                    f"Policy shape is incorrect. Expected shape {(len(self.states_augmented), len(self.times) - 1)}, got {policy.shape}"
                )
                raise ValueError("Policy shape is incorrect.")

        if (
            value.shape[0] != len(self.states_augmented)
            or value.shape[1] != len(self.times) - 1
        ):
            logger.error(
                f"Value shape is incorrect. Expected shape {(len(self.states_augmented), len(self.times) - 1)}, got {value.shape}"
            )
            raise ValueError("Value shape is incorrect.")

        for time_ix, time in enumerate(self.times[:-1]):
            time_dict_value = {}
            time_dict_policy = {}
            for state_ix, state in enumerate(self.states_augmented):
                # If the time index is the same as the current time, add the state to the policy dict.
                if state[1] == time:
                    time_dict_value[state[0]] = float(value[state_ix, time_ix])
                    time_dict_policy[state[0]] = self.actions[
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
        assert s in self.states, "State must be a valid value in S"

        # Check t is a valid value for time
        assert t in self.times, "Time must be a valid value in T"

        # Check that t is not the last time period
        assert t != max(self.times), (
            "Cannot plot transition probabilities for the last time period"
        )

        # Check a is a valid action
        assert a in self.actions, "Action must be a valid action in A"

        # What is the index of a?
        a_ix = self.actions.index(a)

        # Get the transition matrix for action a
        P = self.transition_matrices[a_ix]
        R = self.reward_matrices[a_ix]

        # Which row of the transition matrix corresponds to state s at time t?
        s_current = [
            k
            for k, v in enumerate(self.states_augmented)
            if v[0] == s and v[1] == t
        ]

        # Check that this has exactly one element and extract it
        assert len(s_current) == 1, "State s at time t must be unique"
        s_current_ix = s_current[0]

        # For S_augmented, find which states have time index t_ix + 1
        s_prime_ix = [
            k for k, v in enumerate(self.states_augmented) if v[1] == t + 1
        ]
        s_prime_labels = [self.states_augmented[ix][0] for ix in s_prime_ix]

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
        assert t in self.times, "Time must be a valid value in T"
        assert a in self.actions, "Action must be a valid action in A"

        # empty np array
        matrix = np.zeros((len(self.states), len(self.states)))

        y_labels = self.states

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
            cmap="inferno",
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

    def enforce_monotonicity(
        self,
        value_increasing: bool = True,
        policy_increasing: bool = True,
        allowed_actions: list[float] | None = None,
    ):
        """
        Enforce monotonicity on the policy and value functions
        """

        value_ir = IsotonicRegression(increasing=value_increasing)
        value_monotone = {}

        for t in self.value_function.keys():
            X = list(self.value_function[t].keys())
            y = list(self.value_function[t].values())
            y_monotone = value_ir.fit_transform(X, y)

            value_monotone[t] = dict(zip(X, y_monotone))

        self.value_monotone = value_monotone

        # Policy
        policy_ir = IsotonicRegression(increasing=policy_increasing)
        policy_monotone = {}

        for t in self.policy_function.keys():
            X = list(self.policy_function[t].keys())
            y = list(self.policy_function[t].values())

            # If allowed_actions is not None or empty, use it to enforce
            # monotonicity
            if allowed_actions is not None and len(allowed_actions) > 0:
                y_monotone = monotonic_discrete_fit(
                    y,
                    allowed_actions=allowed_actions,
                    increasing=policy_increasing,
                )
            else:
                y_monotone = policy_ir.fit_transform(X, y)

            policy_monotone[t] = dict(zip(X, y_monotone))

        self.policy_monotone = policy_monotone

        return None

    def forecast(
        self,
        current_state: int | str,
        current_time: int,
        rule: int
        | str
        | Callable = "optimal",  # Changed parameter name and type
    ):
        """
        Produce a forecast from the current state and time for the future
        states.

        Parameters:
        -----------
        current_state: int | str
            The starting state for the forecast.
        current_time: int
            The starting time for the forecast.
        rule: int | str | Callable, optional
            The rule to determine actions for constructing the transition matrix T.
            - If an action value (from self.actions or its index): T is the fixed transition matrix for that action.
            - If "optimal": T is constructed based on the optimal policy for the first time period (self.policy_function_augmented[:, 0]).
            - If a callable function: The function should take (original_state, time_of_augmented_state)
              and return an action value. T is constructed row by row based on this rule.
            Defaults to "optimal".

        Returns:
        --------
        tuple:
            - output_states_labels: List of original state labels.
            - output_times: List of future time steps in the forecast.
            - probabilities_matrix: NumPy array (num_states x num_times) with P(state | time).
            - expected_states: NumPy array (num_times,) with E[state | time].
        """

        # ---------------------------------------------------------------------#
        # To perform a forecast with a transition matrix T, we need an initias
        # state s0; where s0 is a vector of zeros with a 1 in the position of the
        # current state.
        # To perform an n-step-ahead forecast, we compute:
        # sn = s0 @ T^n
        # sn gives the probability distribution over the states at time t + n.

        # Construct the vector of current state

        # Check that current_state is a valid state
        max_time = max(self.times)  # This is the dummy terminal time

        # First check that current_time is less than the dummy terminal time
        assert current_time < max_time, (
            "Current time must be less than the maximum (dummy) terminal time."
        )

        # Find out which of the augmented states correspond to the initial state
        # and time.
        current_state_ix = [
            k
            for k, v in enumerate(self.states_augmented)
            if v[0] == current_state and v[1] == current_time
        ]
        if not current_state_ix:
            raise ValueError(
                f"Initial state ({current_state}, {current_time}) not found in augmented states."
            )

        current_state_vector = np.zeros((1, len(self.states_augmented)))
        current_state_vector[0, current_state_ix[0]] = (
            1  # Assuming unique initial augmented state
        )

        # ---------------------------------------------------------------------#
        # Construct the transition matrix T based on the rule provided. 'rule'
        # in this case means a policy. That is, a mapping from (state, time) to
        # action. Note that every row  of the augmented state space corresponds
        # to a unique (state, time) pair. To construct T, we need to
        # determine the action for each augmented state based on the rule.
        # We pull out the row from the transition matrix corresponding to
        # the action determined by the rule for that augmented state.

        # In the trivial case that the policy is to use the same action in
        # all states; then this reduces to just using the transition matrix
        # for that action.

        # In general, the transition matrix will be a combination of rows from
        # the transition matrices for each action.

        logger.info(f"Forecast called with rule: {rule}")

        if callable(rule):
            logger.info(
                "Constructing transition matrix T using custom rule function."
            )
            if not self.states_augmented:
                T = csr_matrix((0, 0), dtype=float)
            else:
                action_indices_for_T = np.zeros(
                    len(self.states_augmented), dtype=int
                )
                for i, (s_orig, t_curr_aug) in enumerate(
                    self.states_augmented
                ):
                    # t_curr_aug is the time component of the i-th augmented state.
                    # self.times[-1] is the dummy terminal time.
                    if (
                        t_curr_aug == self.times[-1]
                    ):  # If current augmented state is in the dummy terminal time
                        # For dummy terminal states, transitions are typically absorbing.
                        # The P_a[i,:] row should reflect this. Pick any valid action index (e.g., 0).
                        action_indices_for_T[i] = 0
                    else:
                        action_value_from_rule = rule(s_orig, t_curr_aug)
                        if action_value_from_rule not in self.actions:
                            raise ValueError(
                                f"Custom rule function returned action '{action_value_from_rule}' "
                                f"for state ({s_orig},{t_curr_aug}), which is not in self.actions: {self.actions}"
                            )
                        action_indices_for_T[i] = self.actions.index(
                            action_value_from_rule
                        )

                rows_for_T = []
                for state_idx, determined_action_idx in enumerate(
                    action_indices_for_T
                ):
                    row = self.transition_matrices[determined_action_idx][
                        state_idx, :
                    ]
                    rows_for_T.append(row)

                if (
                    not rows_for_T
                ):  # Should not happen if states_augmented is not empty
                    T = csr_matrix(
                        (
                            len(self.states_augmented),
                            len(self.states_augmented),
                        ),
                        dtype=float,
                    )
                else:
                    T = vstack(rows_for_T, format="csr")

        elif isinstance(rule, str) and rule == "optimal":
            logger.info(
                'Constructing transition matrix T using "optimal" policy (based on first time period).'
            )
            # self.policy_function_augmented is (N_aug, N_original_times)
            # Using policy_function_augmented[:, 0] means T is fixed based on the optimal policy for the first time period.
            if self.policy_function_augmented is None:
                raise ValueError(
                    "Optimal policy requested, but policy_function_augmented is not available. Run solve()."
                )

            optimal_action_indices = self.policy_function_augmented[:, 0]

            if (
                not optimal_action_indices.size
                and len(self.states_augmented) > 0
            ):
                # This case implies states_augmented exists but policy is empty, which is unusual.
                logger.warning(
                    "Optimal rule: states_augmented exist but optimal_action_indices is empty. Creating zero matrix T."
                )
                T = csr_matrix(
                    (len(self.states_augmented), len(self.states_augmented)),
                    dtype=float,
                )
            elif not self.states_augmented:
                T = csr_matrix((0, 0), dtype=float)
            else:
                rows_for_T = []
                for state_idx, determined_action_idx in enumerate(
                    optimal_action_indices
                ):
                    row = self.transition_matrices[determined_action_idx][
                        state_idx, :
                    ]
                    rows_for_T.append(row)

                if not rows_for_T and len(self.states_augmented) > 0:
                    T = csr_matrix(
                        (
                            len(self.states_augmented),
                            len(self.states_augmented),
                        ),
                        dtype=float,
                    )
                elif (
                    not rows_for_T and not len(self.states_augmented) > 0
                ):  # no states_augmented
                    T = csr_matrix((0, 0), dtype=float)
                else:
                    T = vstack(rows_for_T, format="csr")
        else:  # rule is a specific action value (label or index)
            action_to_use = None
            action_idx_to_use = -1
            try:
                action_idx_to_use = self.actions.index(rule)
                action_to_use = rule
            except ValueError:
                if isinstance(rule, int) and 0 <= rule < len(self.actions):
                    action_idx_to_use = rule
                    action_to_use = self.actions[action_idx_to_use]
                else:
                    raise ValueError(
                        f"Invalid rule: '{rule}'. Must be 'optimal', a valid action value "
                        f"(one of {self.actions}), an integer index for an action, or a callable."
                    )

            logger.info(
                f"Using fixed transition matrix T for action: {action_to_use} (index: {action_idx_to_use})"
            )
            if not self.transition_matrices:
                raise ValueError(
                    "Transition matrices are not built. Run solve() or build_rewards_and_transitions()."
                )
            T = self.transition_matrices[action_idx_to_use]

        # Prepare the output structures
        output_states_labels = list(self.states)
        output_states_values = np.array(self.states, dtype=float)
        # Forecast up to, but not including, the dummy terminal time max_time
        # output_times are the actual future times for which probabilities are calculated.
        output_times = list(
            range(current_time + 1, max_time)
        )  # Corrected range

        if (
            not output_times
        ):  # If current_time is the last actual decision epoch
            logger.info(
                "Current time is the last decision epoch. No future steps to forecast."
            )
            # Return empty probabilities and expected states, but valid labels and times list
            return (
                output_states_labels,
                [],
                np.array([]).reshape(len(output_states_labels), 0),
                np.array([]),
            )

        probabilities_matrix = np.zeros(
            (len(output_states_labels), len(output_times)), dtype=float
        )
        current_augmented_distribution = current_state_vector.copy()

        for col_idx, target_time in enumerate(output_times):
            current_augmented_distribution = current_augmented_distribution @ T

            if hasattr(current_augmented_distribution, "toarray"):
                dense_dist_at_target_time = (
                    current_augmented_distribution.toarray().flatten()
                )
            else:
                dense_dist_at_target_time = np.asarray(
                    current_augmented_distribution
                ).flatten()

            for row_idx, original_state_label in enumerate(
                output_states_labels
            ):
                augmented_state_tuple = (original_state_label, target_time)
                augmented_idx = self.augmented_state_to_index.get(
                    augmented_state_tuple
                )

                if augmented_idx is not None:
                    prob = dense_dist_at_target_time[augmented_idx]
                    probabilities_matrix[row_idx, col_idx] = prob
                else:
                    probabilities_matrix[row_idx, col_idx] = 0.0
                    logger.debug(
                        f"Augmented state {augmented_state_tuple} not found in index for forecast. Assigning P=0."
                    )

        expected_states = np.dot(output_states_values, probabilities_matrix)

        # ---------------------------------------------------------------------#
        expected_actions = []
        # If the rule is callable, we need to apply it to each state and time
        if callable(rule):
            logger.info(
                "Using custom rule function to determine expected actions."
            )
            for t, s in zip(output_times, expected_states):
                # Apply the rule function to get the expected action
                expected_action = rule(s, t)
                expected_actions.append(expected_action)

        elif isinstance(rule, str) and rule == "optimal":
            # Use the optimal policy

            if self.policy_monotone is not None:
                policy = self.policy_monotone
                logger.info(
                    "Using monotone policy for expected actions trace."
                )
            else:
                policy = self.policy_function

            # Apply the policy to get expected actions
            for t, s in zip(output_times, expected_states):
                # Find the closest state to s in self.states

                expected_actions.append(get_policy(t, s, policy))

        return (
            output_states_labels,
            output_times,
            probabilities_matrix,
            expected_states,
            expected_actions,
        )
