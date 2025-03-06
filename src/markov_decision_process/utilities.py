# ----------------------------------------------------------------------------#
#
#                                  Utilities
#
# ----------------------------------------------------------------------------#


import numpy as np


import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------#

def discretize_normal(
    grid: list[float] | np.ndarray,
    mean: float,
    std: float
) -> np.ndarray:
    """
    Discretize a normal distribution.
    
    For each grid point, computes the probability mass of a normal distribution
    with the given mean and standard deviation that falls within the region:
    - For the first grid point: from negative infinity to the midpoint between
      the first and second grid points
    - For interior grid points: from the midpoint between the current point and
      the previous point to the midpoint between the current point and the next point
    - For the last grid point: from the midpoint between the second-to-last and
      last grid points to positive infinity
    
    Parameters
    ----------
    grid : list[float] | np.ndarray
        A sorted list of grid points.
    mean : float
        Mean of the normal distribution.
    std : float
        Standard deviation of the normal distribution.
        
    Returns
    -------
    np.ndarray
        Array of same size as grid with the probability mass for each grid point.
    """
    from scipy.stats import norm
    
    grid = np.asarray(grid)
    n = len(grid)
    
    # Calculate the boundaries for integration
    boundaries = np.zeros(n + 1)
    # First boundary is -infinity
    boundaries[0] = -np.inf
    # Last boundary is infinity
    boundaries[n] = np.inf
    # Middle boundaries are midpoints between grid points
    for i in range(1, n):
        boundaries[i] = (grid[i-1] + grid[i]) / 2
    
    # Calculate the probability mass for each grid point
    probabilities = np.zeros(n)
    for i in range(n):
        # Use the CDF to calculate the probability mass between boundaries
        probabilities[i] = norm.cdf(boundaries[i+1], loc=mean, scale=std) - norm.cdf(boundaries[i], loc=mean, scale=std)
    
    return probabilities

