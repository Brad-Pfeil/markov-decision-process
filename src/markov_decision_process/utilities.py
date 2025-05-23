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
    grid: list[float] | np.ndarray, mean: float, std: float
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
        boundaries[i] = (grid[i - 1] + grid[i]) / 2

    # Calculate the probability mass for each grid point
    probabilities = np.zeros(n)
    for i in range(n):
        # Use the CDF to calculate the probability mass between boundaries
        probabilities[i] = norm.cdf(
            boundaries[i + 1], loc=mean, scale=std
        ) - norm.cdf(boundaries[i], loc=mean, scale=std)

    return probabilities


def monotonic_discrete_fit(
    x: np.ndarray,
    allowed_actions: list[float] | np.ndarray,
    increasing: bool = True,
):
    """
    Take a sequence of values and enforce monotonicity on them, either
    increasing or decreasing, choosing only from a set of allowed values.
    """

    # Allowed values must not be None or empty. If it is, throw an error and recommend
    # using the non-discrete fit.
    if allowed_actions is None or len(allowed_actions) == 0:
        raise ValueError("Allowed values must not be None or empty")

    n = len(x)
    allowed_actions = sorted(allowed_actions)
    k = len(allowed_actions)

    # Initialize DP tables
    dp = np.full((n, k), np.inf)
    path = np.zeros((n, k), dtype=int)

    # First row: cost of choosing any allowed value
    for j in range(k):
        dp[0][j] = (x[0] - allowed_actions[j]) ** 2

    # Fill DP table
    for i in range(1, n):
        for j in range(k):
            if increasing:
                # Ensure non-decreasing: allowed_actions[l] <= allowed_actions[j]
                for l in range(j + 1):
                    cost = dp[i - 1][l] + (x[i] - allowed_actions[j]) ** 2
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        path[i][j] = l
            else:
                # Ensure non-increasing: allowed_actions[l] >= allowed_actions[j]
                for l in range(j, k):
                    cost = dp[i - 1][l] + (x[i] - allowed_actions[j]) ** 2
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        path[i][j] = l

    # Find best endpoint
    end_idx = np.argmin(dp[-1])
    result = [0] * n
    result[-1] = allowed_actions[end_idx]

    # Backtrack
    for i in reversed(range(1, n)):
        end_idx = path[i][end_idx]
        result[i - 1] = allowed_actions[end_idx]

    return result
