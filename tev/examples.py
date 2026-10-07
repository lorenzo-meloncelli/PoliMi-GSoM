"""
Practical Examples for Tracking Error Efficient Frontier
=========================================================

This file demonstrates various real-world scenarios and constraint configurations.
"""

import numpy as np
import pandas as pd
from tracking_error_efficient_frontier import TrackingErrorEfficientFrontier
from mean_variance_efficient_frontier import MeanVarianceEfficientFrontier
# from mean_cvar_efficient_frontier import MeanCVaREfficientFrontier
# from entropy_efficient_frontier import EntropyEfficientFrontier
import os
import datetime as dt
# import riskfolio as rp
# ============================================================================
# EXAMPLE 1: Basic Usage with Sector Constraints
# ============================================================================

def example_1_sector_constraints():
    """
    Build efficient frontier with sector allocation constraints.
    Scenario: Global equity portfolio with regions and equity/bond split.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Relative to a bmk")
    print("="*80)

    print_to_csv = True
    exp_ret_type = '10Y_ARI'
    cons = 'mod'
    model_direc = os.getcwd()
    # filename = r'C:\Users\zanetti\dev\qs\sandbox\PortOpt\Data.xlsx'
    rundate = '28112025'
    output_path = r''
    input_filename = 'Data4Optimisation.xlsx'
    input_full_filename = os.path.join(output_path, input_filename)
    df = pd.read_excel(input_full_filename, sheet_name='timeseries', header=0, parse_dates=True,
                       index_col=0)  # skiprows=range(1, 3)

    instruments = pd.read_excel(input_full_filename, sheet_name='asset_config', header=0)
    instruments.index = instruments.OPT_NAME

    asset_selection = instruments.OPT_NAME.tolist()

    rf_asset = 'EUR Cash'
    n_assets = len(asset_selection)
    start_date = dt.datetime(2003, 6, 30)
    frequency = 1
    resamp_freq = 'QE'
    risk_method = 'hist'  # 'ledoit_wolf' exp_cov semicovariance sample_cov
    min_weight = 0.0

    # Based on asset selection, reduce the instruments dataframe
    instruments_sub = instruments.loc[asset_selection]

    rets = df.dropna()
    sigma = pd.read_excel(input_full_filename, sheet_name='cov', header=0, index_col=0)
    sigma.columns = asset_selection
    sigma.index = asset_selection
    vol = pd.Series(np.sqrt(np.diag(sigma.values)), index=asset_selection)
    mu = np.array(instruments_sub[
                      exp_ret_type])  # sample_cov, semicovariance, exp_cov, lodoit_wolf, lodoit_wolf_constant_variance, ledoit_wolf_single_factor, ledoit_wolf_constant_correlation. oracle_approximating

    # if GEO convert to ARIT
    horz = int(exp_ret_type[:exp_ret_type.find('Y')])
    apprx = 1 - 1 / (frequency * horz)
    if exp_ret_type.find("GEO") != -1:
        mu += np.array(apprx / 2 * vol ** 2)
        mu = pd.Series(mu)

    mu = pd.DataFrame([mu.tolist()], columns=asset_selection)

    # Benchmark: 60% equities, 30% bonds, 10% commodities
    bmk_w = instruments_sub.loc[:, ['BMK_W']]
    benchmark_weights = bmk_w.values
    

    

    benchmark_returns = rets @ benchmark_weights

    # Initialize
    frontier = TrackingErrorEfficientFrontier(
        returns = rets.values,
        benchmark_returns=benchmark_returns,
        asset_names=asset_selection,
        initial_benchmark_weights=benchmark_weights,
        exp_returns = mu,
        covariance = sigma,
    #    frequency=frequency
    )

    # Define constraints
    # 1. Total equities ≤ 65%
    # 2. Total bonds ≥ 20%
    # 3. Commodities ≤ 15%
    # 4. Single equity asset ≤ 30%

    constraints_dict = {

    'equity_cap': {
        'assets': {
            'World EquityU': 1.0,
            'Barclays Global TreasuryH': 1.0,
        },
        'bound': 0.2
    }
        # 'relative': {
        #     'assets': {
        #         'EMU Bond All MaturityU':  1.0,
        #         'Barclays Global TreasuryH': -0.5
        #     },
        #     'bound': 0.0
        # },
    }
    
    # constraints_dict = {}

    A, b = frontier.create_linear_constraint_matrix(constraints_dict)
    linear_constraints = {'A': A, 'b': b}
    # linear_constraints = None
    # Build frontier
    frontier_df = frontier.construct_efficient_frontier(
        n_portfolios=50,
        weight_bounds=(0, 1),
        linear_constraints=linear_constraints,
        use_ex_ante=False
    )


    # Find portfolio with best risk-adjusted return
    #frontier_df['excess_return'] = frontier_df['return'] - benchmark_returns.mean() * 252
    #frontier_df['information_ratio'] = frontier_df['excess_return'] / frontier_df['tracking_error']

    #best_ir_idx = frontier_df['information_ratio'].idxmax()

    #weights = frontier_df.loc[best_ir_idx][[f'w_{n}' for n in asset_selection]].values

    return frontier_df


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRACKING ERROR EFFICIENT FRONTIER - AGD")
    print("="*80)

    df1 = example_1_sector_constraints()
    #df2 = example_2_relative_constraints()
    #df3 = example_3_diversification()
    #dfs_unc, dfs_tac = example_4_tactical_vs_strategic()
    output_path = r''
    output_full_filename = os.path.join(output_path, 'constrained_tev.csv')
    df1.to_csv(output_full_filename)
    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80)
 
