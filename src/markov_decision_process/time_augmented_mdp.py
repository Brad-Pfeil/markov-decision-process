import numpy as np
import pandas as pd

from typing import Any, Callable, List, Tuple, Literal

from scipy.sparse import csr_matrix

from sklearn import preprocessing

from itertools import product

from mdptoolbox.mdp import FiniteHorizon

import matplotlib.pyplot as plt
import seaborn as sns

import inspect
import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class TimeAugmentedMDP:

    def __init__(
        self,
        states: List[int] | List[float] | List[str],
        actions: List[int] | List[float] | List[str],
        times: List[int],
        reward_function: Callable,
        transition_function: Callable | None = None,
        discount_factor: float = 1,
        mode: Literal['flexible', 'vectorized', 'model'] = 'flexible',
        model: Callable | None = None,
    ):

        # Assign these inputs to self
        for key, value in locals().items():
            if key != 'self':
                setattr(self, key, value)

        # Transitions and rewards will be constructed by methods below
        self.transition_matrices: List[csr_matrix] = []
        self.reward_matrices: List[csr_matrix] = []
       

        # Checks on inputs
        self.__check_inputs()

        return None

    def __check_inputs(self):

        # States must be non-empty
        if not self.states:
            raise ValueError('States must be non-empty')
        # States must be a list where all elements are of the same type
        if not all(isinstance(state, type(self.states[0])) for state in self.states):
            raise ValueError('States must be a list where all elements are of the same type')
        # States must be integers, floats, or strings
        if not all(isinstance(state, (int, float, str)) for state in self.states):
            raise ValueError('States must be integers, floats, or strings')
        
        # Same for actions
        if not self.actions:
            raise ValueError('Actions must be non-empty')
        if not all(isinstance(action, type(self.actions[0])) for action in self.actions):
            raise ValueError('Actions must be a list where all elements are of the same type')
        if not all(isinstance(action, (int, float, str)) for action in self.actions):
            raise ValueError('Actions must be integers, floats, or strings')
        
        # Times must be integers
        if not all(isinstance(time, int) for time in self.times):
            raise ValueError('Times must be integers')
        # Times must be non-empty
        if not self.times:
            raise ValueError('Times must be non-empty')
        # Times must be consecutive
        if not all(self.times[i] == self.times[i-1] + 1 for i in range(1, len(self.times))):
            raise ValueError('Times must be consecutive and increasing')
        
        # Mode must be one of the allowed values
        if self.mode not in ['flexible', 'vectorized', 'model']:
            raise ValueError('Mode must be one of "flexible", "vectorized", or "model"')
        # If mode is 'model', model must be a function
        if self.mode == 'model' and not callable(self.model):
            raise ValueError('If mode is "model", model must be a function')
        # If mode is flexible or vectorized, a transition function must be provided
        if self.mode in ['flexible', 'vectorized'] and not callable(self.transition_function):
            raise ValueError('If mode is "flexible" or "vectorized", a transition function must be provided')
        
        # Check that reward function has correct arguments
        reward_signature = list(inspect.signature(self.reward_function).parameters.keys())
        if reward_signature != ['s_prime', 's', 'a', 't']:
            logger.warning(f'Reward function currently has arguments {reward_signature}. Should be (s_prime, s, a, t)')
            raise ValueError('Reward function should have arguments (s_prime, s, a, t)')

        # If the transition_function is provided, check that it has the correct arguments
        if self.transition_function is not None:
            transition_signature = list(inspect.signature(self.transition_function).parameters.keys())
            if transition_signature != ['s_prime', 's', 'a', 't']:
                logger.warning(f'Transition function currently has arguments {transition_signature}. Should be (s_prime, s, a, t)')
                raise ValueError('Transition function should have arguments (s_prime, s, a, t)')
        
        return None
