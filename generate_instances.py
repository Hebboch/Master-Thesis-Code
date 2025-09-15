import random
import numpy as np
import math
import GBT

"""
Purpose: Bundle all required sets and parameters
Result:  Create parameter dict for each scenario for solver
"""

def generate_unique_points(n_points, grid_length):
    """
    Input:  n_points (int), grid_length (int) for a grid of size grid_length × grid_length.
    Output: List of (x, y) coordinate tuples, length = n_points.
    """
    # Seed for reproducibility
    random.seed(2)

    # Generate all grid points
    all_points = [(i, j) for i in range(grid_length) for j in range(grid_length)]

    # Sample random n_points
    selected = random.sample(all_points, n_points)

    return selected


def compute_manhattan_distances(points1,points2):
    """
    Input:  points1, points2 (lists; of (x, y) tuples)
    Output: dist (dict; calculated distances between all locations)
    """
    # Distance dict
    dist = {}
    for i in range(len(points1)):
        dist[i] = {}
        for j in range(len(points2)):
            point1 = points1[i]
            point2 = points2[j]
            ## Manhattan metric
            d = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
            dist[i][j] = d

    return dist


def generate_instance():
    """
    Input:  None
    Output: (params1, params2, params3) dicts for solver.solve(...)
    """
    # Seed for reproducibility
    rng = np.random.default_rng(4)  

    # General framework - consitent across all scenarios
    ## Sets
    I = [0,1,2] # (Burgers,Drinks,Sides)
    B = [0,1]

    ## Parameters 
    ### Time
    tau_start = 90
    tau_end = 390
    tau_svc = 15
    tau_work = 360
    tau_cont = 270
    Omega = [30,45]   
    M__ = tau_end

    ### Capacity
    k_resupply = [3000,2000,3000]
    k_sales = [800,600,800]

    ### Economic
    F_staff_sales = 10*8
    F_staff_driver = 15*8     
    p = [4.2,1.75,2.1]  

    ### #Sales Staff
    epsilon = 2

    ### Demand via GBT
    df = GBT.load_clean_rows('the_burger_spot.csv')
    model = GBT.fit_on_rows(df)
    pivot = GBT.prediction_pivot_from_rows(model,df)
    findings = GBT.random_day_minmax_over_areas(pivot)
    sigma = {
        0: float(findings.iloc[0,2]),
        1:  float(findings.iloc[1,2]),
        2:   float(findings.iloc[2,2]),
    }
    kappa = 4000      


    # Case specific
    ## S1 - Small-scale urban areas 
    ### Sets
    D1 = list(range(2))
    L1 = list(range(9)) 
    R1 = list(range(2))

    ### Parameters
    #### Sales trucks
    n1 = 7

    #### Coordinates + distance calculation
    coor1 = generate_unique_points(len(L1),4) 
    dist_L1 = compute_manhattan_distances(coor1,coor1)
    dist_wh1 = compute_manhattan_distances([(-1,1.5)],coor1)

    #### Time
    tau_travel1 = {}
    max_tau_travel1 = 0
    for i in range(len(L1)):
        tau_travel1[i] = {}
        for j in range(len(L1)):
            h = 1.6*dist_L1[i][j]/23.3*60 # 1.6* cause one mile difference between points
            tau_travel1[i][j] = h 
            if h > max_tau_travel1:
                max_tau_travel1 = h
    M1 = max(max_tau_travel1,(len(L1))*tau_svc)
    tau_wh1 = {}
    for i in range(len(L1)):
        tau_wh1[i] = 1.6*dist_wh1[0][i]/23.3*60 # 1.6* cause one mile difference between points

    #### Demand
    delta1 = {}
    delta1[0] = {}
    delta1[1] = {}
    delta1[2] = {}
    for i in range(len(L1)):
        delta1[0][i] = rng.integers(findings.iloc[0,0],findings.iloc[0,1]+1) # Random number between min/max of category burger
        percent = (delta1[0][i] - findings.iloc[0,0]) / (findings.iloc[0,1] - findings.iloc[0,0]) # Drawn burger share of its intervall
        delta1[1][i] = math.floor((findings.iloc[1,1] - findings.iloc[1,0]) * percent + findings.iloc[1,0]) # category drinks
        delta1[2][i] = math.floor((findings.iloc[2,1] - findings.iloc[2,0]) * percent + findings.iloc[2,0]) # category drinks
    Delta1 = {}
    for j in range(len(I)):
        Delta1[j] = {}
        for i in range(len(L1)):
            Delta1[j][i] = max(delta1[j][i] - k_sales[j], 0)
    M_1 = findings.iloc[0,1]+findings.iloc[1,1]+findings.iloc[2,1] # Sum over max-values is also suitable

    #### Economic
    F_resupply1 = {}
    for i in range(len(L1)):
        F_resupply1[i] = rng.integers(20,141)
    F_sales1 = {}
    for i in range(len(L1)):
        F_sales1[i] = rng.integers(30,151)  
    c_wh1 = {}
    for i in range(len(L1)):
        c_wh1[i] = 1.6*dist_wh1[0][i]*1.19 # 1.6* cause one mile difference between points
    c_travel1 = {}
    for i in range(len(L1)):
        c_travel1[i] = {}
        for j in range(len(L1)):
            c_travel1[i][j] = 1.6*dist_L1[i][j]*1.19 # 1.6* cause one mile difference between points

    ### Output scenario S1
    params1 = {
    'n': n1,
    'p': p,
    'F_resupply': F_resupply1,
    'F_sales': F_sales1,
    'tau_start': tau_start,
    'tau_end': tau_end,
    'tau_svc': tau_svc,
    'k_resupply': k_resupply,
    'k_sales': k_sales,
    'F_staff_sales': F_staff_sales,
    'F_staff_driver': F_staff_driver,
    'kappa': kappa,
    'epsilon': epsilon,
    'Omega': Omega,
    'tau_work': tau_work,
    'tau_cont': tau_cont,
    'M__': M__,
    'L': L1,
    'R': R1,
    'I': I,
    'B': B,
    'D': D1,
    'delta': delta1,
    'Delta': Delta1,
    'c_travel': c_travel1,
    'c_wh': c_wh1,
    'tau_travel': tau_travel1,
    'tau_wh': tau_wh1,
    'M': M1,
    'M_': M_1,
    'sigma' : sigma
    }


    ## S2 - Medium-scale urban areas 
    #### Sets
    D2 = list(range(4))
    L2 = list(range(19))
    R2 = list(range(4))

    ### Parameters
    ### #Sales trucks
    n2 = 15

    #### Coordinates + distance calculation
    coor2 = generate_unique_points(len(L2),6)
    dist2_L = compute_manhattan_distances(coor2,coor2)
    dist2_wh = compute_manhattan_distances([(-1,2.5)],coor2)

    #### Time
    tau_travel2 = {}
    max_tau_travel2 = 0
    for i in range(len(L2)):
        tau_travel2[i] = {}
        for j in range(len(L2)):
            h = 1.6*dist2_L[i][j]/23.3*60 
            tau_travel2[i][j] = h 
            if h > max_tau_travel2:
                max_tau_travel2 = h
    M2 = max(max_tau_travel2,(len(L2))*tau_svc)
    tau_wh2 = {}
    for i in range(len(L2)):
        tau_wh2[i] = 1.6*dist2_wh[0][i]/23.3*60 

    #### Demand
    delta2 = {}
    delta2[0] = {}
    delta2[1] = {}
    delta2[2] = {}
    for i in range(len(L2)):
        delta2[0][i] = rng.integers(findings.iloc[0,0],findings.iloc[0,1]+1) 
        percent = (delta2[0][i] - findings.iloc[0,0]) / (findings.iloc[0,1] - findings.iloc[0,0]) 
        delta2[1][i] = math.floor((findings.iloc[1,1] - findings.iloc[1,0]) * percent + findings.iloc[1,0]) 
        delta2[2][i] = math.floor((findings.iloc[2,1] - findings.iloc[2,0]) * percent + findings.iloc[2,0]) 
    Delta2 = {}
    for j in range(len(I)):
        Delta2[j] = {}
        for i in range(len(L2)):
            Delta2[j][i] = max(delta2[j][i] - k_sales[j], 0)
    M_2 = findings.iloc[0,1]+findings.iloc[1,1]+findings.iloc[2,1]

    #### Economic
    F_resupply2 = {}
    for i in range(len(L2)):
        F_resupply2[i] = rng.integers(20,141)
    F_sales2 = {}
    for i in range(len(L2)):
        F_sales2[i] = rng.integers(30,151)  
    c_travel2 = {}
    for i in range(len(L2)):
        c_travel2[i] = {}
        for j in range(len(L2)):
            c_travel2[i][j] = 1.6*dist2_L[i][j]*1.19 
    c_wh2 = {}
    for i in range(len(L2)):
        c_wh2[i] = 1.6*dist2_wh[0][i]*1.19 

    ## Output scenario S2
    params2 = {
    'n': n2,
    'p': p,
    'F_resupply': F_resupply2,
    'F_sales': F_sales2,
    'tau_start': tau_start,
    'tau_end': tau_end,
    'tau_svc': tau_svc,
    'k_resupply': k_resupply,
    'k_sales': k_sales,
    'F_staff_sales': F_staff_sales,
    'F_staff_driver': F_staff_driver,
    'kappa': kappa,
    'epsilon': epsilon,
    'Omega': Omega,
    'tau_work': tau_work,
    'tau_cont': tau_cont,
    'M__': M__,
    'L': L2,
    'R': R2,
    'I': I,
    'B': B,
    'D': D2,
    'delta': delta2,
    'Delta': Delta2,
    'c_travel': c_travel2,
    'c_wh': c_wh2,
    'tau_travel': tau_travel2,
    'tau_wh': tau_wh2,
    'M': M2,
    'M_': M_2,
    'sigma' : sigma
    }


    ## S3 - Large-scale urban areas 
    ### Sets
    D3 = list(range(20))
    L3 = list(range(100))
    R3 = list(range(20))

    ### Parameters
    ### #Sales trucks
    n3 = 15

    #### Coordinates + distance calculation
    coor3 = generate_unique_points(len(L3),13)
    dist3_L = compute_manhattan_distances(coor3,coor3)
    dist3_wh = compute_manhattan_distances([(-1,6)],coor3)

    #### Time
    tau_travel3 = {}
    max_tau_travel3 = 0
    for i in range(len(L3)):
        tau_travel3[i] = {}
        for j in range(len(L3)):
            h = 1.6*dist3_L[i][j]/23.3*60
            tau_travel3[i][j] = h 
            if h > max_tau_travel3:
                max_tau_travel3 = h
    M3 = max(max_tau_travel3,(len(L3))*tau_svc)
    tau_wh3 = {}
    for i in range(len(L3)):
        tau_wh3[i] = 1.6*dist3_wh[0][i]/23.3*60 

    #### Demand
    delta3 = {}
    delta3[0] = {}
    delta3[1] = {}
    delta3[2] = {}
    for i in range(len(L3)):
        delta3[0][i] = rng.integers(findings.iloc[0,0],findings.iloc[0,1]+1) 
        percent = (delta3[0][i] - findings.iloc[0,0]) / (findings.iloc[0,1] - findings.iloc[0,0]) 
        delta3[1][i] = math.floor((findings.iloc[1,1] - findings.iloc[1,0]) * percent + findings.iloc[1,0]) 
        delta3[2][i] = math.floor((findings.iloc[2,1] - findings.iloc[2,0]) * percent + findings.iloc[2,0]) 
    Delta3 = {}
    for j in range(len(I)):
        Delta3[j] = {}
        for i in range(len(L3)):
            Delta3[j][i] = max(delta3[j][i] - k_sales[j], 0)
    M_3 = findings.iloc[0,1]+findings.iloc[1,1]+findings.iloc[2,1] 

    #### Economic
    F_resupply3 = {}
    for i in range(len(L3)):
        F_resupply3[i] = rng.integers(20,141)
    F_sales3 = {}
    for i in range(len(L3)):
        F_sales3[i] = rng.integers(30,151)  
    c_travel3 = {}
    for i in range(len(L3)):
        c_travel3[i] = {}
        for j in range(len(L3)):
            c_travel3[i][j] = 1.6*dist3_L[i][j]*1.19 
    c_wh3 = {}
    for i in range(len(L3)):
        c_wh3[i] = 1.6*dist3_wh[0][i]*1.19 

    ## Output scenario S3
    params3 = {
    'n': n3,
    'p': p,
    'F_resupply': F_resupply3,
    'F_sales': F_sales3,
    'tau_start': tau_start,
    'tau_end': tau_end,
    'tau_svc': tau_svc,
    'k_resupply': k_resupply,
    'k_sales': k_sales,
    'F_staff_sales': F_staff_sales,
    'F_staff_driver': F_staff_driver,
    'kappa': kappa,
    'epsilon': epsilon,
    'Omega': Omega,
    'tau_work': tau_work,
    'tau_cont': tau_cont,
    'M__': M__,
    'L': L3,
    'R': R3,
    'I': I,
    'B': B,
    'D': D3,
    'delta': delta3,
    'Delta': Delta3,
    'c_travel': c_travel3,
    'c_wh': c_wh3,
    'tau_travel': tau_travel3,
    'tau_wh': tau_wh3,
    'M': M3,
    'M_': M_3,
    'sigma' : sigma
    }

    return params1,params2,params3