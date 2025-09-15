import solver
import copy
import numpy as np
import pandas as pd
import math
from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from SALib.plotting.morris import covariance_plot

"""
Purpose: Performs sensitivity analysis
Result:  Generates Morris Effects (EE) Table and corresponding covariance plot
"""

def apply_fluctuations_to_params(params_base, X):
    """
    Input:  params_base (dict) baseline parameters; X (array-like) Morris factors in fixed order.
    Output: New params dict with adjusted prices, costs, times, and demand
    """
    # Sets
    I = params_base['I']
    L = params_base['L']

    # Unpack factor vector
    idx = 0
    price_factor            = X[idx]; idx += 1
    travel_cost_factor      = X[idx]; idx += 1
    time_factor             = X[idx]; idx += 1
    location_cost_factor    = X[idx]; idx += 1
    delta_deviation         = X[idx]; idx += 1

    ## Prevent changes in params_base for consistency 
    Params = copy.deepcopy(params_base)

    # Scaled prices
    p_base = params_base['p']
    Params['p'] = [v * price_factor for v in p_base]

    # Scaled transportation costs
    c_travel_new = {l: {} for l in L}
    for l in L:
        for l_ in L:
            c_travel_new[l][l_] = params_base['c_travel'][l][l_] * travel_cost_factor
    Params['c_travel'] = c_travel_new
    Params['c_wh'] = {l: params_base['c_wh'][l] * travel_cost_factor for l in L}

    # Scaled travel times
    tau_travel_new = {l: {} for l in L}
    for l in L:
        for l_ in L:
            tau_travel_new[l][l_] = params_base['tau_travel'][l][l_] * time_factor
    Params['tau_travel'] = tau_travel_new
    Params['tau_wh'] = {l: params_base['tau_wh'][l] * time_factor for l in L}
    # Scaled location costs
    Params['F_sales']    = {l: params_base['F_sales'][l]    * location_cost_factor    for l in L}
    Params['F_resupply'] = {l: params_base['F_resupply'][l] * location_cost_factor for l in L}

    # Scaled demand
    sigma = params_base['sigma']  
    k_sales = params_base['k_sales'] 
    delta_new = {i: {} for i in I}
    Delta_new = {i: {} for i in I}
    for i in I:
        for l in L:
            proposal = params_base['delta'][i][l] + delta_deviation * sigma[i]
            proposal = max(0.0, proposal)
            delta_new[i][l] = math.floor(proposal) # stick to integer value
            Delta_new[i][l] = math.floor(max(0.0, proposal - k_sales[i])) # stick to integer value
    Params['delta'] = delta_new
    Params['Delta'] = Delta_new

    return Params



def build_morris_problem():
    """
    Input:  None
    Output: dict with 'num_vars', 'names', 'bounds' matching apply_fluctuations_to_params()
    """
    # Factor names
    keys = [
        'Gross profit',
        'Transportation cost',
        'Travel time',
        'Location cost',
        'Demand',
    ]

    # Define fluctuation bounds
    bounds = [
        [0.95, 1.05],   # Prices ±5%
        [0.90, 1.10],   # Transportation costs ±10%
        [0.75, 1.25],   # Travel times ±25%
        [0.90, 1.10],   # Location costs ±10%
        [-1.0, 1.0]]    # Demand ± 1(*std)

    return {'num_vars': len(keys), 'names': keys, 'bounds': bounds}


def evaluate_model_once(params_variant):
    """
    Input:  params_variant (dict) varied parameters
    Output: Objective value (float) from the solver
    """
    # Return model objective
    model = solver.solve(params_variant)

    return model.ObjVal


def run_morris_SA(params,r = 10,num_levels = 4,seed = 4):
    """
    Input:  params (dict) of baseline; r (int) trajectories; num_levels (int) levels p_tilde; seed (int)
    Output: (df, Si) where df has columns ['factor','mu_star','sigma','mu'] and Si is SALib result
    """
    problem = build_morris_problem()

    # Sampling
    X = morris_sample(problem,N=r,num_levels=num_levels,seed=seed)

    # Evaluate model for each sampled vector
    Y = np.empty(X.shape[0], dtype=float)
    for i, row in enumerate(X):
        Pvar = apply_fluctuations_to_params(params, row)
        Y[i] = evaluate_model_once(Pvar)

    # Analyze
    Si = morris_analyze(problem, X, Y,conf_level=0.95,num_levels=num_levels)

    # Create DataFrame
    df = pd.DataFrame({
        'factor': problem['names'],
        'mu_star': Si['mu_star'],
        'sigma': Si['sigma'],
        'mu': Si['mu']
    }).sort_values('mu_star', ascending=False).reset_index(drop=True)

    return df,Si


def plot_morris_covariance(Si):
    """
    Input:  Si (dict) from SALib.morris_analyze; show (bool); savepath (str|None).
    Output: (fig, ax) matplotlib figure and axes; file saved if savepath is given.
    """
    # Extract arrays
    mu_star = np.asarray(Si["mu_star"], dtype=float)
    sigma   = np.asarray(Si["sigma"],   dtype=float)
    names   = list(Si.get("names", [f"f{i}" for i in range(len(mu_star))]))

    # Base figure
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=False)
    covariance_plot(ax, Si) 

    # Set colored points per factor
    cmap = plt.cm.get_cmap("tab10", len(mu_star))
    scatter_handles = []
    for i, (x, y, label) in enumerate(zip(mu_star, sigma, names)):
        h = ax.scatter(x, y, s=70, color=cmap(i), edgecolor="none", label=label, zorder=3)
        scatter_handles.append(h)

    # Axes format
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
    ax.set_xlabel("μ*") 
    ax.set_ylabel("σ")

    # Reference legend
    ref_handles = [
        plt.Line2D([0],[0], color="black", lw=1, ls="-",  label="σ/μ* = 1.0"),
        plt.Line2D([0],[0], color="black", lw=1, ls="--", label="σ/μ* = 0.5"),
        plt.Line2D([0],[0], color="black", lw=1, ls="-.", label="σ/μ* = 0.1"),
    ]
    leg_ref = ax.legend(handles=ref_handles, title="Reference",
                      loc="upper left", bbox_to_anchor=(1.02, 1.00), frameon=False,
                      alignment='left')
    ax.add_artist(leg_ref)

    # Factor Legend
    ax.legend(handles=scatter_handles, title="Factors",
          loc="center left", bbox_to_anchor=(1.02, 0.58), frameon=False,
          alignment='left')

    # Inset zoom
    axins = ax.inset_axes([0.08, 0.705, 0.30, 0.26], zorder=10)
    x0, x1 = 0, 180
    y0, y1 = 0, 120
    xx = np.linspace(x0, x1, 50)
    for slope, style in [(1.0, "-"), (0.5, "--"), (0.1, "-.")]:
        axins.plot(xx, slope*xx, style, color="black", lw=1)
    for i, (x, y) in enumerate(zip(mu_star, sigma)):
        if x0 <= x <= x1 and y0 <= y <= y1:
            axins.scatter(x, y, s=40, color=cmap(i), edgecolor="none", zorder=3)
    axins.set_xlim(x0, x1); axins.set_ylim(y0, y1)
    axins.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
    axins.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
    axins.tick_params(labelsize=8)
    axins.set_facecolor("white")
    for sp in axins.spines.values():
        sp.set_edgecolor("0.25"); sp.set_linewidth(1.0)

    # Placeholder for legends
    plt.tight_layout(rect=(0.00, 0.00, 0.80, 1.00))

    # Open plot
    plt.show()

    return