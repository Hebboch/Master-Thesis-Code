import solver
import generate_instances as gen
import sensitivity_analysis as sa

"""
Purpose: Run performance and sensitivity analyses
Result:  Terminal logs (solver progress, solution, Morris summary) and a plot
"""

def run_performance(params_triplet):
    """
    Purpose: Solve scenarios S1–S3 and show computatinal log
    Input:  params_triplet = (params1, params2, params3)
    """
    # Unpack scenarios
    ## Order: S1, S2, S3
    params1, params2, params3 = params_triplet

    # Scenario S1
    print("Solution process of Scenario 1:\n")
    solver.solve(params1, display_processing=True, stop=True)

    input("Press Enter to start Scenario 2...")
    # Scenario S2
    print("Solution process of Scenario 2:\n")
    solver.solve(params2, display_processing=True, stop=True)

    input("Press Enter to start Scenario 3...")
    # Scenario S3
    print("Solution process of Scenario 3:\n")
    solver.solve(params3, display_processing=True, stop=True)

    return 


def run_sensitivity_s1(params1, r=10, num_levels=4, seed=4):
    """
    Purpose: Present baseline S1 solution and sensitivty analysis
    Input:  params1; trajectories r, levels p_tilde, seed
    """
    # Baseline solution
    print("Solution of Scenario 1:\n")
    print("Indexed shifted with -1 compared to established notation:\n")
    solver.solve(params=params1, display_solution=True, display_processing=False)

    input("Press Enter to start Morris analysis...")
    # Morris analysis
    df_morris, Si = sa.run_morris_SA(params=params1, r=r, num_levels=num_levels, seed=seed)
    print("\nMorris elementary effects (EE): Table 15")
    ## Displays Table 15
    print(df_morris)
    ## Shows Figure 8
    sa.plot_morris_covariance(Si)

    return


if __name__ == "__main__":
    # Generate instances
    params_triplet = gen.generate_instance()

    # Performance analysis
    run_performance(params_triplet)

    input("Press Enter to start sensitivity analysis...")
    # Sensitivity analysis for S1
    run_sensitivity_s1(params_triplet[0], r=10, num_levels=4, seed=4) # [~ 1 min]