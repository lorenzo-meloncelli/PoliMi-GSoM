"""
Efficient Frontier with Volatility Minimization and Linear Constraints
============================================================================

This module implements a robust optimization framework that constructs an efficient frontier
by minimizing portfolio volatility, with support for linear constraints.

The efficient frontier spans from the minimum volatility portfolio to the maximum
expected return portfolio, with each point on the frontier representing a portfolio that
minimizes volatility for a given target level of expected return.

Key Features:
- Minimize portfolio variance (w^T * Sigma * w)
- Flexible linear constraints (e.g., sector or asset class constraints)
- Bounds on individual asset weights
- Support for long-only and long-short strategies
- Numerical stability and convergence optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict, List, Optional
import warnings
import os
warnings.filterwarnings('ignore')


class MeanVarianceEfficientFrontier:
    """
    Constructs an efficient frontier that minimizes portfolio volatility
    while targeting different levels of expected return.

    Tracking error and other benchmark-relative metrics are still computed
    and reported in the output for reference.

    Parameters
    ----------
    returns : np.ndarray
        Historical asset returns of shape (n_periods, n_assets)
    benchmark_returns : np.ndarray
        Historical benchmark returns of shape (n_periods,)
    asset_names : List[str], optional
        Names of assets for better readability
    initial_benchmark_weights : np.ndarray, optional
        Initial benchmark weights (used to calculate relative weights)
    exp_returns : np.ndarray
        Expected returns vector
    covariance : np.ndarray
        Covariance matrix
    """

    def __init__(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
        initial_benchmark_weights: Optional[np.ndarray] = None,
        exp_returns=np.ndarray,
        covariance=np.ndarray,
        frequency: int = 4 
    ):
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.n_assets = returns.shape[1]
        self.n_periods = returns.shape[0]
        self.exp_returns = exp_returns
        self.covariance = covariance,
        self.frequency = frequency

        # Asset names
        if asset_names is None:
            self.asset_names = [f'Asset_{i}' for i in range(self.n_assets)]
        else:
            self.asset_names = asset_names

        # Benchmark properties
        self.benchmark_returns = benchmark_returns
        self.benchmark_mean_return = exp_returns.values @ initial_benchmark_weights
        self.benchmark_std = benchmark_returns.std() * np.sqrt(self.frequency)

        # Benchmark weights (for relative tracking error calculation)
        self.benchmark_weights = initial_benchmark_weights

        # Store optimal portfolios
        self.frontier_returns = []
        self.frontier_tracking_errors = []
        self.frontier_weights = []

    # -------------------------------------------------------------------------
    # Objective functions
    # -------------------------------------------------------------------------

    def _objective_minimize_vol(self, weights: np.ndarray) -> float:
        """
        Objective function: minimize portfolio variance.
        Variance = w^T * Sigma * w
        """
        return weights @ self.covariance @ weights

    def _ex_post_tracking_error_variance(self, weights: np.ndarray) -> float:
        """
        Calculate ex-post tracking error variance (used for output reporting only).
        TE^2 = Var(r_p - r_b)
        """
        portfolio_returns = self.returns @ weights
        tracking_errors = (
            np.array(portfolio_returns).reshape((len(portfolio_returns), 1))
            - np.array(self.benchmark_returns)
        )
        te_variance = np.var(tracking_errors, ddof=1)
        return te_variance

    def _ex_ante_tracking_error_variance(self, weights: np.ndarray) -> float:
        """
        Calculate ex-ante tracking error variance using relative weights.
        TE_variance = (w - w_b)^T * Cov * (w - w_b)
        """
        if self.benchmark_weights is None:
            raise ValueError("Benchmark weights required for ex-ante TE calculation")

        weight_diff = weights - self.benchmark_weights
        te_variance = weight_diff @ self.covariance @ weight_diff
        return te_variance

    def _ex_post_tracking_error_cvar(self, weights: np.ndarray, confidence=5) -> float:
        """
        Calculate ex-post tracking error CVaR (used for output reporting only).
        """
        portfolio_returns = self.returns @ weights
        tracking_errors = (
            np.array(portfolio_returns).reshape((len(portfolio_returns), 1))
            - np.array(self.benchmark_returns)
        )
        belowVaR = tracking_errors <= np.percentile(tracking_errors, confidence)
        return tracking_errors[belowVaR].mean()

    def _objective_minimize_te(self, weights: np.ndarray) -> float:
        """Kept for reference — minimize tracking error variance."""
        return self._ex_post_tracking_error_variance(weights)

    def _objective_minimize_te_cvar(self, weights: np.ndarray) -> float:
        """Kept for reference — minimize tracking error CVaR."""
        return self._ex_post_tracking_error_cvar(weights)

    # -------------------------------------------------------------------------
    # Helper functions
    # -------------------------------------------------------------------------

    def _portfolio_return(self, weights: np.ndarray, sign=1) -> float:
        """Calculate expected portfolio return."""
        weights = np.array(weights).reshape((self.n_assets, 1))
        return sign * self.exp_returns @ weights

    def _get_cvar(self, weights: np.ndarray, confidence=5):
        """Calculate the quarterly CVaR."""
        portfolio_returns = self.returns @ weights
        belowVaR = portfolio_returns <= np.percentile(portfolio_returns, confidence)
        return portfolio_returns[belowVaR].mean()

    def _constraint_target_return(self, weights: np.ndarray, target_return: float) -> float:
        """Constraint: portfolio return equals target return."""
        return_diff = (self._portfolio_return(weights) - target_return).values
        return return_diff.reshape((1))

    def _constraint_sum_to_one(self, weights: np.ndarray) -> float:
        """Constraint: weights sum to 1."""
        return np.sum(weights) - 1.0

    def _constraint_linear(self, weights: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Constraint: A @ weights <= b
        Returns b - A @ w >= 0 (scipy ineq convention).
        """
        return b - A @ weights

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------

    def optimize_portfolio(
        self,
        target_return: float,
        weight_bounds: Tuple[float, float] = (0.0, 1.0),
        constraints: Optional[Dict] = None,
        use_ex_ante: bool = False,
        initial_guess: Optional[np.ndarray] = None,
        ftol: float = 1e-20,
        maxiter: int = 1000
    ) -> Dict:
        """
        Optimize portfolio to minimize volatility for a given target return.

        Parameters
        ----------
        target_return : float
            Target portfolio return (annualized)
        weight_bounds : Tuple[float, float]
            (min_weight, max_weight) for each asset. Default: (0, 1) for long-only
        constraints : list, optional
            List of scipy constraint dicts (sum-to-one + any linear constraints)
        use_ex_ante : bool
            Unused — kept for API compatibility
        initial_guess : np.ndarray, optional
            Initial guess for optimization
        ftol : float
            Tolerance for convergence
        maxiter : int
            Maximum iterations

        Returns
        -------
        Dict
            Optimization results including weights, tracking error, and return
        """

        # Initial guess
        if initial_guess is None:
            x0 = np.array([1.0 / self.n_assets] * self.n_assets)
        else:
            x0 = initial_guess

        bounds = [weight_bounds] * self.n_assets

        # Add target return constraint
        new_con = {
            'type': 'eq',
            'fun': lambda w: self._constraint_target_return(w, target_return)
        }
        constraints.append(new_con)

        # Objective: minimize portfolio variance w^T * Sigma * w
        objective = self._objective_minimize_vol

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': ftol, 'maxiter': maxiter}
        )

        if not result.success:
            warnings.warn(
                f"Optimization did not converge for target return {target_return}: {result.message}"
            )

        weights = result.x

        return {
            'weights': weights,
            'target_return': target_return,
            'actual_return': self._portfolio_return(weights),
            'tracking_error': np.sqrt(self._ex_post_tracking_error_variance(weights) * self.frequency),
            'portfolio_std': np.sqrt(weights @ self.covariance @ weights) * np.sqrt(self.frequency),
            'portfolio_cvar': self._get_cvar(weights),
            'success': result.success,
            'message': result.message
        }

    def construct_efficient_frontier(
        self,
        n_portfolios: int = 50,
        weight_bounds: Tuple[float, float] = (0.0, 1.0),
        linear_constraints: Optional[Dict] = None,
        use_ex_ante: bool = False,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Construct the efficient frontier by optimizing portfolios at different return levels.

        The frontier spans from minimum volatility to maximum return.

        Parameters
        ----------
        n_portfolios : int
            Number of portfolios along the frontier
        weight_bounds : Tuple[float, float]
            (min_weight, max_weight) for each asset
        linear_constraints : Dict, optional
            Linear constraints {'A': ..., 'b': ...}
        use_ex_ante : bool
            Unused — kept for API compatibility
        min_return : float, optional
            Minimum target return. If None, calculated from minimum volatility portfolio
        max_return : float, optional
            Maximum target return. If None, uses maximum expected return

        Returns
        -------
        pd.DataFrame
            Efficient frontier data with columns: return, tracking_error,
            portfolio_std, portfolio_cvar, w_<asset_name> for each asset
        """

        constraints_general = [
            {'type': 'eq', 'fun': self._constraint_sum_to_one}
        ]
        if linear_constraints is not None:
            A = linear_constraints['A']
            b = linear_constraints['b']
            constraints_general.append({
                'type': 'ineq',
                'fun': lambda w: self._constraint_linear(w, A, b)
            })

        # Find minimum volatility portfolio to determine min_return
        if min_return is None:
            constraints_min_vol = constraints_general[:]

            result_min_vol = minimize(
                self._objective_minimize_vol,
                np.array([1.0 / self.n_assets] * self.n_assets),
                method='SLSQP',
                bounds=[weight_bounds] * self.n_assets,
                constraints=constraints_min_vol,
                options={'ftol': 1e-20, 'maxiter': 1000}
            )
            min_return = self._portfolio_return(result_min_vol.x)

        # Find maximum return portfolio
        if max_return is None:
            constraints_max_ret = constraints_general[:]

            result_max_ret = minimize(
                self._portfolio_return,
                np.array([1.0 / self.n_assets] * self.n_assets),
                method='SLSQP',
                bounds=[weight_bounds] * self.n_assets,
                constraints=constraints_max_ret,
                options={'ftol': 1e-20, 'maxiter': 1000},
                args=(-1)
            )
            max_return = self._portfolio_return(result_max_ret.x)

        # Generate target returns
        target_returns = np.linspace(min_return.values, max_return.values, n_portfolios)

        frontier_data = {
            'return': [],
            'tracking_error': [],
            'portfolio_std': [],
            'portfolio_cvar': [],
            'weights': []
        }

        # Optimize at each return level
        for i, target_ret in enumerate(target_returns):
            constraints_tgt_ret = constraints_general[:]
            result = self.optimize_portfolio(
                target_return=target_ret,
                weight_bounds=weight_bounds,
                constraints=constraints_tgt_ret,
                use_ex_ante=use_ex_ante
            )

            frontier_data['return'].append(result['actual_return'])
            frontier_data['tracking_error'].append(result['tracking_error'])
            frontier_data['portfolio_std'].append(result['portfolio_std'])
            frontier_data['portfolio_cvar'].append(result['portfolio_cvar'])
            frontier_data['weights'].append(result['weights'])

        return_df = pd.concat(frontier_data['return'], axis=0, ignore_index=True)
        tracking_error_df = pd.DataFrame(frontier_data['tracking_error'])
        portfolio_std_df = pd.DataFrame(frontier_data['portfolio_std'])
        portfolio_cvar_df = pd.DataFrame(frontier_data['portfolio_cvar'])

        df = pd.concat([return_df, tracking_error_df, portfolio_std_df, portfolio_cvar_df], axis=1)
        df.columns = ['return', 'tracking_error', 'portfolio_std', 'portfolio_cvar']

        weights_df = pd.DataFrame(
            frontier_data['weights'],
            columns=[f'w_{name}' for name in self.asset_names]
        )

        df = pd.concat([df, weights_df], axis=1)

        self.frontier_returns = return_df
        self.frontier_tracking_errors = tracking_error_df
        self.frontier_weights = weights_df

        return df

    # -------------------------------------------------------------------------
    # Constraint helpers
    # -------------------------------------------------------------------------

    def create_linear_constraint_matrix(
        self,
        constraints_dict: Dict[str, Dict[str, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create linear constraint matrices from a dictionary of constraints.

        Example:
            constraints = {
                'us_equity_max': {
                    'assets': {'Asset_1': 1.0, 'Asset_2': 1.0},
                    'bound': 0.6
                }
            }

        This creates: Asset_1 + Asset_2 <= 0.6

        Parameters
        ----------
        constraints_dict : Dict
            Dictionary where each constraint has:
            - 'assets': dict mapping asset_name -> coefficient
            - 'bound': upper bound for the constraint

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (A matrix, b vector) for constraints A @ w <= b
        """
        constraints_list = []
        bounds_list = []

        for constraint_name, constraint_info in constraints_dict.items():
            a_row = np.zeros(self.n_assets)

            for asset_name, coeff in constraint_info['assets'].items():
                asset_idx = self.asset_names.index(asset_name)
                a_row[asset_idx] = coeff

            constraints_list.append(a_row)
            bounds_list.append(constraint_info['bound'])

        A = np.array(constraints_list)
        b = np.array(bounds_list)

        return A, b

    def create_relative_weight_constraints(
        self,
        asset_names,
        max_pct_of_others=0.30,
        min_pct_of_others=0.10
    ):
        constraints_dict = {}
        for target in asset_names:
            c_max = {target: 1.0, **{a: -max_pct_of_others for a in asset_names if a != target}}
            constraints_dict[f'{target}_vs_others_max'] = {'assets': c_max, 'bound': 0.0}
            c_min = {target: -1.0, **{a: min_pct_of_others for a in asset_names if a != target}}
            constraints_dict[f'{target}_vs_others_min'] = {'assets': c_min, 'bound': 0.0}
        return constraints_dict

    def get_frontier_summary(self) -> pd.DataFrame:
        """Get summary statistics of the efficient frontier."""
        if not self.frontier_returns:
            raise ValueError("Frontier not yet constructed. Call construct_efficient_frontier first.")

        return pd.DataFrame({
            'Portfolio': range(len(self.frontier_returns)),
            'Return': self.frontier_returns,
            'Tracking_Error': self.frontier_tracking_errors
        })
