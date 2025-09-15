from gurobipy import Model, GRB, quicksum
import os
import psutil

"""
Purpose: Build and solve the integrated model and optionally print solution details
Result:  Model solution and computation log
"""

def solve(params,display_solution = False, display_processing = False, stop = False):
    """
    Input:
        - params (dict): Model data
        - display_solution (bool, default False): If True, prints nonzero variable values and a compact summary
        - display_processing (bool, default False): If True, shows solver progress logs
        - stop (bool, default False): If True, applies TimeLimit=12h and MIPGap=1% for controlled runs
    Output:
        - gurobipy.Model
    """


    # Initialise model
    model = Model("TruckNetworkModel")
    # Give no processing output
    if not display_processing: 
        model.Params.OutputFlag = 0
    # Solver settings
    if stop:
        model.setParam("TimeLimit", 12*3600)    ## 12h 
        model.setParam("MIPGap", 0.01)          ## Gap = 1% 
    model.setParam("Seed", 4)                   ## Reproducibility

    # Sets
    ## Truck Network Model
    L = params['L']
    R = params['R']
    I = params["I"] 
    ## Integrated Model 
    D = params["D"]   
    B = params["B"]              
    
    # Parameter
    ## Truck Network Model
    n = params['n'] 
    F_resupply = params['F_resupply'] 
    F_sales = params['F_sales']       
    c_travel = params['c_travel']  
    c_wh = params['c_wh']   
    tau_start = params['tau_start']
    tau_end = params['tau_end']
    tau_travel = params['tau_travel']                
    tau_svc = params['tau_svc']
    k_resupply = params['k_resupply']
    k_sales = params['k_sales']
    delta = params['delta']
    Delta = params['Delta']
    p = params['p']
    M = params['M']
    ## Integrated Model +
    F_staff_sales = params["F_staff_sales"]            
    F_staff_driver = params["F_staff_driver"]          
    kappa = params["kappa"]   
    epsilon = params["epsilon"]                           
    tau_wh = params["tau_wh"]    
    Omega = params["Omega"]       
    tau_cont = params["tau_cont"]  
    tau_work = params["tau_work"]  
    M_ = params["M_"]           
    M__ = params["M__"]             

    # Variables
    ## Truck Network Model
    z = model.addVars(L, vtype=GRB.BINARY, name="z")
    y = model.addVars(L, vtype=GRB.BINARY, name="y")
    q = model.addVars(R, L, vtype=GRB.BINARY, name="q") 
    x = model.addVars(R, L, L, vtype=GRB.BINARY, name="x") 
    h = model.addVars(R, L, vtype=GRB.INTEGER, lb=0, ub= len(L), name="h")
    f = model.addVars(I, R, L, L, vtype=GRB.INTEGER, lb=0, name="f")
    theta = model.addVars(R, L, vtype=GRB.INTEGER, lb=0, name="theta")
    psi = model.addVars(R, L, vtype=GRB.INTEGER, lb=0, name="psi")
    ## Integrated Model +
    mu = model.addVars(L, vtype=GRB.BINARY, name="mu")              
    w = model.addVars(D, vtype=GRB.BINARY, name="w")       
    sigma = model.addVars(R, D, vtype=GRB.BINARY, name="sigma")   
    Gamma = model.addVars(R, vtype=GRB.INTEGER, lb=0, name="Gamma") 
    pi = model.addVars(R, vtype=GRB.INTEGER, lb=0, name="pi")  
    u_shift = model.addVars(R, R, D, vtype=GRB.BINARY, name="u_shift") 
    u_loc = model.addVars(R, R, L, vtype=GRB.BINARY, name="u_loc")    
    Lambda = model.addVars(D, vtype=GRB.BINARY, name="lambda")
    beta = model.addVars(R, B, D, vtype=GRB.BINARY, name="beta")
    Theta = model.addVars(R, D, vtype=GRB.INTEGER, lb=0, name="Theta")  
    Xi    = model.addVars(R, D, vtype=GRB.INTEGER, lb=0, name="Xi")    
    Theta_tilde = model.addVars(R, R, D, vtype=GRB.INTEGER, lb=0, name="Theta_tilde")
    Xi_tilde    = model.addVars(R, R, D, vtype=GRB.INTEGER, lb=0, name="Xi_tilde")
    a_short = model.addVars(R, D, vtype=GRB.BINARY, name="a_short")
    a_long  = model.addVars(R, D, vtype=GRB.BINARY, name="a_long")

    # Objective function
    model.setObjective(
        quicksum(p[i] * delta[i][l] * y[l] for i in I for l in L)
    - (quicksum(F_resupply[l] * z[l] + 2 * c_wh[l] * quicksum(q[r, l] for r in R) for l in L)
        + quicksum((F_sales[l] + 2 * c_wh[l]) * y[l] for l in L)
        + quicksum(c_travel[l][l_] * x[r, l, l_] for r in R for l in L for l_ in L if l_ != l)
        + F_staff_sales * quicksum(epsilon * y[l] + mu[l] for l in L)
        + F_staff_driver * quicksum(w[d]+ Lambda[d] for d in D)),
        GRB.MAXIMIZE
    )

    # Constraints
    ## Truck Network Model
    ### (4.2)
    for r in R:
        for l in L:
            model.addConstr(q[r, l] <= z[l], name=f"c42_{r}_{l}")

    ### (4.3)
    for r in R:
        model.addConstr(quicksum(q[r, l] for l in L) <= 1, name=f"c43_{r}")

    ### (4.4)
    for r in R:
        for l in L:
            model.addConstr(quicksum(x[r, l, l_] for l_ in L if l != l_) >= q[r, l], name=f"c44_{r}_{l}")

    ### (4.5)
    for r in R:
        for l in L:
            for l_ in L:
                if l == l_:
                    continue
                model.addConstr(x[r, l, l_] <= quicksum(k_resupply[i] for i in I) * ( y[l_] + z[l_]), name=f"c45_{r}_{l}_{l_}")

    ### (4.6)
    for r in R:
        for l_ in L:
            model.addConstr(quicksum(x[r, l, l_] for l in L if l != l_) - quicksum(x[r, l_, l__] for l__ in L if l__ != l_) == 0, name=f"c46_{r}_{l_}")

    ### (4.7) 
    for r in R:
        for l in L:
            for l_ in L:
                if l == l_:
                    continue
                model.addConstr(h[r, l] - h[r, l_] + len(L)  * x[r, l, l_] <= (len(L) - 1) * ( 1 + q[r,l_]), name=f"c47_{r}_{l}_{l_}")

    ### (4.8)
    for r in R:
        for l_ in L:
            model.addConstr(h[r, l_] <= len(L)  * quicksum(x[r, l, l_] for l in L if l != l_), name=f"c48_{r}_{l_}")

    ### (4.9) 
    for r in R:
        for l_ in L:
            model.addConstr(h[r, l_] >= quicksum(x[r, l, l_] for l in L if l != l_), name=f"c49_{r}_{l_}")

    ### (4.10) 
    for i in I:
        for l_ in L:
            model.addConstr(quicksum(f[i, r, l, l_] for r in R for l in L if l != l_) == Delta[i][l_] * y[l_] , name=f"c410_{i}_{l_}")

    ### (4.11) 
    for i in I:
        for r in R:
            for l_ in L:
                model.addConstr(quicksum(f[i, r, l, l_] for l in L if l != l_) <= y[l_] * k_sales[i], name=f"c411_{i}_{r}_{l_}")

    ### (4.12) 
    for i in I:
        for r in R:
            for l in L:
                model.addConstr(quicksum(f[i, r, l, l_] for l_ in L if l_ != l) <= k_resupply[i] * q[r, l], name=f"c412_{i}_{r}_{l}")

    ### (4.13) 
    for i in I:
        for r in R:
            for l in L:
                for l_ in L:
                    if l == l_:
                        continue
                    model.addConstr(f[i, r, l, l_] <= k_resupply[i] * quicksum(x[r, l__, l_] for l__ in L if l__ != l_), name=f"c413_{i}_{r}_{l}_{l_}")

    ### (4.14) 
    for l in L:
        model.addConstr(quicksum(theta[r, l] + psi[r, l] for r in R) <= tau_end - tau_start, name=f"c414_{l}")

    ### (4.15)
    for r in R:
        for l__ in L:
            expr = quicksum(tau_travel[l][l_] * x[r, l, l_] for l in L for l_ in L if l != l_)
            model.addConstr(expr - M * (1 - q[r, l__]) <= theta[r, l__], name=f"c415_{r}_{l__}")

    ### (4.16)
    for r in R:
        for l__ in L:
            expr = quicksum(x[r, l, l_] for l in L if l != l__ for l_ in L if l != l_)
            model.addConstr(tau_svc * expr - M * (1 - q[r, l__]) <= psi[r, l__], name=f"c416_{r}_{l__}")

    ### (4.17)
    model.addConstr(quicksum(y[l] for l in L) <= n, name=f"c417")

    ### (4.18)
    for l in L:
        model.addConstr(z[l] + y[l] <= 1, name=f"c418_{l}")

    ## Integrated Model completion
    ### (4.26) 
    for l in L:
        model.addConstr(M_ * mu[l] >= (quicksum(delta[i][l] for i in I) - kappa) * y[l] , name=f"c426_{l}")

    ### (4.27)
    for r in R:
        model.addConstr(quicksum(q[r, l] for l in L) == quicksum(sigma[r, d] for d in D), name=f"c427_{r}")

    ### (4.28)
    for d in D:
        model.addConstr(len(R) * w[d] >= quicksum(sigma[r, d] for r in R), name=f"c428_{d}")

    ### (4.29)
    for r in R:
        for l in L:
            model.addConstr(Gamma[r] >= theta[r, l] + psi[r, l] + 2 * tau_wh[l] * q[r, l], name=f"c429_{r}_{l}")

    ### (4.30) 
    for d in D:
        for r in R:
            for r_ in R:
                if r == r_:
                    continue
                model.addConstr(pi[r] + Gamma[r] + quicksum(Omega[b] * beta[r_, b, d] for b in B) <= pi[r_] - 1 + M__ * (1 - u_shift[r, r_, d]),name=f"c430_{r}_{r_}_{d}")

    ### (4.31)
    for d in D:
        for r in R:
            for r_ in R:
                if r == r_:
                    continue
                model.addConstr(u_shift[r, r_, d] + u_shift[r_, r, d] >= sigma[r, d] + sigma[r_, d] - 1, name=f"c431_{r}_{r_}_{d}")

    ### (4.32)
    for d in D:
        for r in R:
            for r_ in R:
                if r == r_:
                    continue
                model.addConstr(u_shift[r, r_, d] + u_shift[r_, r, d] <= 0.5 * (sigma[r, d] + sigma[r_, d]), name=f"c432_{r}_{r_}_{d}")

    ### (4.33)
    for r in R:
        model.addConstr(pi[r] >= tau_start * quicksum(sigma[r, d] for d in D), name=f"c433_{r}")

    ### (4.34)
    for r in R:
        model.addConstr(pi[r] + Gamma[r] <= tau_end * quicksum(sigma[r, d] for d in D), name=f"c434_{r}")

    ### (4.35)
    for l in L:
        for r in R:
            for r_ in R:
                if r == r_:
                    continue
                model.addConstr(pi[r] + theta[r, l] + psi[r, l] <= pi[r_] - 1 + M__ * (1 - u_loc[r, r_, l]), name=f"c435_{r}_{r_}_{l}")

    ### (4.36)
    for l in L:
        for r in R:
            for r_ in R:
                if r == r_:
                    continue
                model.addConstr(u_loc[r, r_, l] + u_loc[r_, r, l] >= q[r, l] + q[r_, l] - 1, name=f"c436_{r}_{r_}_{l}")

    ### (4.37)
    for l in L:
        for r in R:
            for r_ in R:
                if r == r_:
                    continue
                model.addConstr(u_loc[r, r_, l] + u_loc[r_, r, l] <= 0.5 * (q[r, l] + q[r_, l]), name=f"c437_{r}_{r_}_{l}")

    ### (4.38) 
    for d in D:
        for r in R:
            model.addConstr(Theta[r, d] == quicksum(Theta_tilde[r_, r, d] for r_ in R if r_ != r),name=f"c438_{r}_{d}")

    ### (4.39) 
    for d in D:
        for r in R:
            model.addConstr(Xi[r, d] == quicksum(Xi_tilde[r_, r, d] for r_ in R if r_ != r),name=f"c439_{r}_{d}")

    ### (4.40) 
    for d in D:
        for r in R:
            for r_ in R:
                if r_ == r:
                    continue
                model.addConstr(Theta_tilde[r_, r, d] >= Gamma[r_] - quicksum(psi[r_, l] for l in L) - M__ * (1 - u_shift[r_, r, d]),name=f"c440_{r_}_{r}_{d}")

    ### (4.41) 
    for d in D:
        for r in R:
            for r_ in R:
                if r_ == r:
                    continue
                model.addConstr(Xi_tilde[r_, r, d] >= Gamma[r_] - M__ * (1 - u_shift[r_, r, d]),name=f"c441_{r_}_{r}_{d}")

    ### (4.42) 
    for d in D:
        for r in R:
            for r_ in R:
                if r_ == r:
                    continue
                model.addConstr(a_short[r, d] <= 0.5 * (beta[r_, B[0], d] + u_shift[r_, r, d]),name=f"c442_{r_}_{r}_{d}")

    ### (4.43) 
    for d in D:
        for r in R:
            for r_ in R:
                if r_ == r:
                    continue
                model.addConstr(
                    a_long[r, d] <= 0.5 * (beta[r_, B[1], d] + u_shift[r_, r, d]),name=f"c443_{r_}_{r}_{d}")

    ### (4.44) 
    for d in D:
        for r in R:
            model.addConstr(Theta[r, d] + Gamma[r] - quicksum(psi[r, l] for l in L) <= tau_cont + M__ * (beta[r, B[1], d] + Lambda[d] + a_long[r, d]),name=f"c444_{r}_{d}")

    ### (4.45) 
    for d in D:
        for r in R:
            model.addConstr(Xi[r, d] + Gamma[r] <= tau_work + M__ * (quicksum(beta[r, b, d] for b in B) + Lambda[d] + a_short[r, d] + a_long[r, d]),name=f"c445_{r}_{d}")

    ### (4.46)
    for d in D:
        for r in R:
            model.addConstr(quicksum(beta[r, b, d] for b in B) <= Xi[r, d] / 165,name=f"c446_{r}_{d}")

    ### (4.47) 
    for d in D:
        for r in R:
            model.addConstr(Gamma[r] - quicksum(psi[r, l] for l in L) <= tau_cont + M__ * (1 - sigma[r, d] + Lambda[d]),name=f"c447_{r}_{d}")

    ### (4.48) 
    for d in D:
        for r in R:
            model.addConstr(Gamma[r] <= tau_work + M__ * (1 - sigma[r, d] + Lambda[d]),name=f"c448_{r}_{d}")

    ### (4.49) 
    for d in D:
        model.addConstr(quicksum(beta[r, b, d] for r in R for b in B) + Lambda[d] <= 1,name=f"c449_{d}")


    # Execute
    model.optimize()


    # Display solution
    if display_solution or display_processing:
        ## Peak Working Set (Bytes) → GB
        proc = psutil.Process(os.getpid())
        mi = proc.memory_info()
        peak_bytes = getattr(mi, "peak_wset", mi.rss)   
        print(f"Peak RAM: {peak_bytes/1024/1024/1024:.4f} GB")
    
        ## Display output
        runtime  = model.Runtime
        nodes    = model.NodeCount
        inc      = model.ObjVal if model.SolCount > 0 else None
        bound    = getattr(model, "ObjBound", None)
        gap = (model.MIPGap if model.SolCount > 0 else None)
        print(f"Time: {runtime:.1f}s | Nodes: {nodes} | Inc: {inc} | Bound: {bound} | Gap: {None if gap is None else 100*gap:.4f}%")

    # Display variable values
    if display_solution:
        ## Check solution values
        if model.status == GRB.OPTIMAL:
            ### Print values
            for var in model.getVars():
                if var.X > 1e-6:
                    print(f"{var.VarName} = {var.X}")
            return model
        elif model.status == GRB.TIME_LIMIT:
            print(f"Time limit reached.")
            ### Print values
            for var in model.getVars():
                if var.X > 1e-6:
                    print(f"{var.VarName} = {var.X}")
            return model
        else:
            print("No feasible solution found.")
            return None
    else:
        return model