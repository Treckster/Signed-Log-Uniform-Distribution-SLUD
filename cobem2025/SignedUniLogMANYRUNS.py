import time
import matplotlib.pyplot as plt
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.callback import Callback
from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.moo.unsga3 import UNSGA3
import ray
from pymoo.core.termination import TerminateIfAny       # logical OR combiner
from pymoo.termination.fmin import MinimumFunctionValueTermination
import csv
import os
import funcs  # Import the functions from funcs.py

def SLUD(xis, bounds):

    X = np.zeros_like(xis)  # Initialize the output array with the same shape as xis

    # explicitly define lb and ub, can be done in a more elegant way
    lbb = np.array([d[0] for d in bounds]) # obtains the lower bounds of magnitude (MIN)
    ubb = np.array([d[1] for d in bounds]) # obtains the upper bounds of magnitude (MAX)
    sbb = np.array([d[2] for d in bounds]) # obtains the sign bounds, 0= both, 1=positive only, -1=negative only
    

    for i in range(len(xis)):
        if sbb[i] == 1:  # positive only
            sign = 1 # DEPRECATED
        elif sbb[i] == -1:  # negative only
            sign = -1 # DEPRECATED
        else:  # both signs
            xix=xis[i]
            logub = np.log10(ubb[i])*2          # upper bound in log10
            loglb = np.log10(lbb[i])*2          # lower bound in log10
            halfrange = (logub - loglb)/2       # half the range (ABS)
            halfmark = (logub + loglb)/2        # mid value (SIGNED)
            dist = xix-halfmark                 # distance from the mid value
            sign = np.sign(dist)                # negative half turns Negative, positive half turns positive
            unit_x = np.abs(dist)/halfrange     # unit distance from the mid value, normalized to [0, 1]
        
        # This is the signed log-uniform distribution   
        #      sign * lower_bound    *         (upper_bound/lower_bound) ^ abs(xi)
        X[i] = sign *    lbb[i]      *    np.power(  ubb[i]/lbb[i] , np.abs(unit_x) ) 

    return X

def LUD(xis, bounds):
    X = np.zeros_like(xis)  # Initialize the output array with the same shape as xis

    # explicitly define lb and ub, can be done in a more elegant way
    lbb = np.array([d[0] for d in bounds]) # obtains the lower bounds of magnitude (MIN)
    ubb = np.array([d[1] for d in bounds]) # obtains the upper bounds of magnitude (MAX)

    for i in range(len(xis)//2):
        X[i] = 10**xis[i] # Convert the first half of the input from log10 to linear scale

    for i in range(len(xis)//2, len(xis)):
        X[i] = xis[i]      # Keep the second half of the input as is to get the signed values later 

    return X

def LIN(xis, bounds):
    
    return xis



# Initialize Ray for parallel computing, before entering loop
debug=False                                     # True for Debugger execution, False for normal execution
n_threads   = 16                                # Number of threads for parallelization
ray.init(num_cpus=n_threads,local_mode=debug)  # Specify number of CPUs to use


# User initialization
n_pop       = 100             # Population size
n_gen       = 500             # Number of generations


# for functoeval in ['rosen', 'brown', 'powell', 'poly7']:  # Loop through the functions to evaluate
for functoeval in [  'brown']:  # Loop through the functions to evaluate
    for decade_selector in [LUD, SLUD, LIN]:  # Loop through the decade selectors
    # for decade_selector in [SLUD]:  # Loop through the decade selectors
        # Loop to run optimization 200 times
        for iteration in range(1000):
            seed = iteration  # Set seed equal to iteration number
            
            # Set variables based on the selected function
            # if using LUD, the number of variables is doubled
            if functoeval == 'rosen': # x = [unconstrained]   ;  fobjval = 0   ;   xopt = (1)^n
                evalfunc = funcs.rosen  # Import the Rosenbrock function from funcs.py
                fobjmin     = 1.0e-8        # Objective function minimum value
                if decade_selector == LUD:
                    # Number of variables
                    n_vars = 4

                    # Define bounds for the variables
                    ub = [1E2, 1E2, 1, 1]  # Biggest value to search in modulus
                    lb = [1E-4, 1E-4, -1, -1]  # Lowest value to search in modulus

                    # Create bounds array with size n_vars, where each element is [lb, ub, sign]
                    bounds = np.array([[lb[i], ub[i], 0] for i in range(n_vars)])  # Create bounds array

                    # Convert bounds to log10 scale for SLUD
                    xl=np.concatenate([np.log10(lb[0:2]), lb[2:4]])
                    xu=np.concatenate([np.log10(ub[0:2]), ub[2:4]])

                elif decade_selector == SLUD:
                    # Number of variables
                    n_vars = 2

                    # Define bounds for the variables
                    ub = [1E2, 1E2]  # Biggest value to search in modulus
                    lb = [1E-4, 1E-4]  # Lowest value to search in modulus

                    # Since we are using SLUD, we need to define the sign of each variable for the search space
                    sign = [0, 0]  # Sign bounds, 0= both, 1=positive only, -1=negative only (CHANGE AS NEEDED)
                    for i in range(len(sign)):
                        if sign[i] == 0:
                            sgn = 2  # Both signs allowed, so we double the range
                        else:
                            sgn = 1  # Single sign allowed, so we keep the range as is

                    # Create bounds array with size n_vars, where each element is [lb, ub, sign]
                    bounds = np.array([[lb[i], ub[i], sign[i]] for i in range(n_vars)])  # Create bounds array

                    # Convert bounds to log10 scale for SLUD
                    xl=np.log10(lb)*sgn
                    xu=np.log10(ub)*sgn
                else: # use linear sampling
                    decade_selector = LIN # linear distribution
                    # Number of variables
                    n_vars = 4

                    # Define bounds for the variables
                    ub = [1E2, 1E2, 1, 1]  # Biggest value to search in modulus
                    lb = [1E-4, 1E-4, -1 ,-1]  # Lowest value to search in modulus

                    # Create bounds array with size n_vars, where each element is [lb, ub, sign]
                    bounds = np.array([[lb[i], ub[i], 0] for i in range(n_vars)])  # Create bounds array

                    # dont convert bounds to log10, this is uniform linear
                    xl=np.array(lb)  # Lower bounds in linear scale
                    xu=np.array(ub)  # Upper bounds in linear scale

            elif functoeval == 'brown': # "badly scaled" x = [unconstrained?]   ;  fobjval = 0   ;   xopt = (1e6, 2e-6)
                evalfunc = funcs.brown  # Import the Brown function from funcs.py
                fobjmin     = 1.0e-10        # Objective function minimum value
                if decade_selector == LUD:
                    # Number of variables
                    n_vars = 4

                    # Define bounds for the variables
                    ub = [1E8, 1E8, 1, 1]  # Biggest value to search in modulus
                    lb = [1E-8, 1E-8, -1, -1]  # Lowest value to search in modulus

                    # Create bounds array with size n_vars, where each element is [lb, ub, sign]
                    bounds = np.array([[lb[i], ub[i], 0] for i in range(n_vars)])  # Create bounds array

                    # Convert bounds to log10 scale for SLUD
                    xl=np.concatenate([np.log10(lb[0:2]), lb[2:4]])
                    xu=np.concatenate([np.log10(ub[0:2]), ub[2:4]])

                elif decade_selector == SLUD:
                    # Number of variables
                    n_vars = 2

                    # Define bounds for the variables
                    ub = [1E8, 1E8]  # Biggest value to search in modulus
                    lb = [1E-8, 1E-8]  # Lowest value to search in modulus

                    # Since we are using SLUD, we need to define the sign of each variable for the search space
                    sign = [0, 0]  # Sign bounds, 0= both, 1=positive only, -1=negative only (CHANGE AS NEEDED)
                    for i in range(len(sign)):
                        if sign[i] == 0:
                            sgn = 2  # Both signs allowed, so we double the range
                        else:
                            sgn = 1  # Single sign allowed, so we keep the range as is

                    # Create bounds array with size n_vars, where each element is [lb, ub, sign]
                    bounds = np.array([[lb[i], ub[i], sign[i]] for i in range(n_vars)])  # Create bounds array

                    # Convert bounds to log10 scale for SLUD
                    xl=np.log10(lb)*sgn
                    xu=np.log10(ub)*sgn
                else: # use linear sampling
                    decade_selector = LIN # linear distribution
                    # Number of variables
                    n_vars = 4

                    # Define bounds for the variables
                    ub = [1E8, 1E8, 1, 1]  # Biggest value to search in modulus
                    lb = [1E-8, 1E-8, -1 ,-1]  # Lowest value to search in modulus

                    # Create bounds array with size n_vars, where each element is [lb, ub, sign]
                    bounds = np.array([[lb[i], ub[i], 0] for i in range(n_vars)])  # Create bounds array

                    # dont convert bounds to log10, this is uniform linear
                    xl=np.array(lb)  # Lower bounds in linear scale
                    xu=np.array(ub)  # Upper bounds in linear scale

            elif functoeval == 'powell': # "badly scaled" x = [-10, 10]   ;  fobjval = 0   ;   xopt = (1.098...×10−5 , 9.106...)
                evalfunc = funcs.powell  # Import the Powell function from funcs.py
                fobjmin     = 1.0e-10        # Objective function minimum value

                if decade_selector == LUD:
                    # Number of variables
                    n_vars = 4

                    # Define bounds for the variables
                    ub = [1E2, 1E2, 1, 1]  # Biggest value to search in modulus
                    lb = [1E-6, 1E-6, -1, -1]  # Lowest value to search in modulus

                    # Create bounds array with size n_vars, where each element is [lb, ub, sign]
                    bounds = np.array([[lb[i], ub[i], 0] for i in range(n_vars)])  # Create bounds array

                    # Convert bounds to log10 scale for SLUD
                    xl=np.concatenate([np.log10(lb[0:2]), lb[2:4]])
                    xu=np.concatenate([np.log10(ub[0:2]), ub[2:4]])

                elif decade_selector == SLUD:
                    # Number of variables
                    n_vars = 2

                    # Define bounds for the variables
                    ub = [1E2, 1E2]  # Biggest value to search in modulus
                    lb = [1E-6, 1E-6]  # Lowest value to search in modulus

                    # Since we are using SLUD, we need to define the sign of each variable for the search space
                    sign = [0, 0]  # Sign bounds, 0= both, 1=positive only, -1=negative only (CHANGE AS NEEDED)
                    for i in range(len(sign)):
                        if sign[i] == 0:
                            sgn = 2  # Both signs allowed, so we double the range
                        else:
                            sgn = 1  # Single sign allowed, so we keep the range as is

                    # Create bounds array with size n_vars, where each element is [lb, ub, sign]
                    bounds = np.array([[lb[i], ub[i], sign[i]] for i in range(n_vars)])  # Create bounds array

                    # Convert bounds to log10 scale for SLUD
                    xl=np.log10(lb)*sgn
                    xu=np.log10(ub)*sgn
                else: # use linear sampling
                    decade_selector = LIN # linear distribution
                    # Number of variables
                    n_vars = 4

                    # Define bounds for the variables
                    ub = [1E2, 1E2, 1, 1]  # Biggest value to search in modulus
                    lb = [1E-6, 1E-6, -1 ,-1]  # Lowest value to search in modulus

                    # Create bounds array with size n_vars, where each element is [lb, ub, sign]
                    bounds = np.array([[lb[i], ub[i], 0] for i in range(n_vars)])  # Create bounds array

                    # dont convert bounds to log10, this is uniform linear
                    xl=np.array(lb)  # Lower bounds in linear scale
                    xu=np.array(ub)  # Upper bounds in linear scale

            elif functoeval == 'poly7': # fobjt=0 ; 
                evalfunc = funcs.poly7  # Import the Polynomial function from funcs.py
                fobjmin = 1e-5  # Objective function minimum value
                if decade_selector == LUD:
                    # Number of variables
                    n_vars = 10

                    # Define bounds for the variables
                    ub = [1E2, 1E-1, 1E-4, 1E-7, 1E-10, 1, 1, 1, 1, 1]  # Biggest value to search in modulus
                    lb = [1E-2, 1E-5, 1E-8, 1E-11, 1E-14, -1, -1, -1, -1, -1]  # Lowest value to search in modulus

                    # Create bounds array with size n_vars, where each element is [lb, ub, sign]
                    bounds = np.array([[lb[i], ub[i], 0] for i in range(n_vars)])  # Create bounds array

                    # Convert bounds to log10 scale for SLUD
                    xl=np.concatenate([np.log10(lb[0:5]), lb[5:10]])
                    xu=np.concatenate([np.log10(ub[0:5]), ub[5:10]])
                elif decade_selector == SLUD:
                    n_vars = 5
                            # Define bounds for the variables
                    ub = [1E2, 1E-1, 1E-4, 1E-7, 1E-10]  # Biggest value to search in modulus
                    lb = [1E-2, 1E-5, 1E-8, 1E-11, 1E-14]  # Lowest value to search in modulus

                    # Since we are using SLUD, we need to define the sign of each variable for the search space
                    sign = [0, 0, 0, 0, 0]  # Sign bounds, 0= both, 1=positive only, -1=negative only (CHANGE AS NEEDED)
                    for i in range(len(sign)):
                        if sign[i] == 0:
                            sgn = 2  # Both signs allowed, so we double the range
                        else:
                            sgn = 1  # Single sign allowed, so we keep the range as is

                    # Create bounds array with size n_vars, where each element is [lb, ub, sign]
                    bounds = np.array([[lb[i], ub[i], sign[i]] for i in range(n_vars)])  # Create bounds array

                    # Convert bounds to log10 scale for SLUD
                    xl=np.log10(lb)*sgn
                    xu=np.log10(ub)*sgn
                else: # use linear sampling
                    decade_selector = LIN # linear distribution
                    # Number of variables
                    n_vars = 10

                    # Define bounds for the variables
                    ub = [1E2, 1E-1, 1E-4, 1E-7, 1E-10, 1, 1, 1, 1, 1]  # Biggest value to search in modulus
                    lb = [1E-2, 1E-5, 1E-8, 1E-11, 1E-14, -1, -1, -1, -1, -1]  # Lowest value to search in modulus

                    # Create bounds array with size n_vars, where each element is [lb, ub, sign]
                    bounds = np.array([[lb[i], ub[i], 0] for i in range(n_vars)])  # Create bounds array

                    # dont convert bounds to log10, this is uniform linear
                    xl=np.array(lb)  # Lower bounds in linear scale
                    xu=np.array(ub)  # Upper bounds in linear scale

            else:
                raise ValueError(f"Unknown function: {functoeval}")
                

            @ray.remote
            def evaluate_batch(batch):
                return [func(x) for x in batch]


            def func(x):
                # Transforms
                xis = decade_selector(x, bounds)  # Apply the decade selector to the input

                yis= evalfunc(xis)  

                return yis


            # class MyProblem(ElementwiseProblem):
            class MyProblem(Problem):
                def __init__(self, *args, **kwargs):
                    # Define your optimization problem
                    # n_var: number of variables
                    # n_obj: number of objectives (1 for single-objective)

                    super().__init__(n_var=n_vars, 
                                    n_obj=1, 
                                    n_ieq_constr=0,
                                    xl=xl,   # Provide the Lowest Modulus Bounds in log10 scale, doubling the range for both if both postive and negative values are allowed
                                    xu=xu    # Provide the Highest Modulus Bounds in log10 scale, doubling the range for both if both postive and negative values are allowed
                                    
                                    )
                
                def _evaluate(self, X, out, *args, **kwargs):
                    # Your objective function here
                    # x is a 2D array with shape (n_population, n_var)
                            # Submit tasks to Ray
                    batch_size = 10  # Adjust based on your problem
                    batches = [X[i:i+batch_size] for i in range(0, len(X), batch_size)]
                    
                    futures = [evaluate_batch.remote(batch) for batch in batches]
                    results = [item for sublist in ray.get(futures) for item in sublist]
                    
                    out["F"] = np.array(results).reshape(-1, 1)

            # Callback to track optimization progress
            class MyCallback(Callback):
                def __init__(self) -> None:
                    super().__init__()
                    self.data["best"] = []
                    
                def notify(self, algorithm):
                    best_value = algorithm.pop.get("F").min()
                    self.data["best"].append(best_value)

            # Create the problem instance
            problem = MyProblem()

            # Algorithm setup
            algorithm = PSO(
                pop_size=n_pop,
                sampling=LHS(), # Since we are providing log10 bounds, we can use LHS sampling to have equal probability per stratum per decade
            )

            # Create a termination criterion
            term_by_gen  = get_termination("n_gen",  n_gen)               # counts generations  
            term_by_fmin = MinimumFunctionValueTermination(fobjmin)    # single-obj only  

            # --- 2) combine them with "OR" ---------------------------------------------
            termination  = TerminateIfAny(term_by_gen, term_by_fmin)


            # Setup callback
            callback = MyCallback()

            # Run the optimization
            res = minimize(
                problem=problem,
                algorithm=algorithm,
                termination=termination,
                seed=seed,
                callback=callback,
                verbose=False,
                save_history=False,
                display=None,
            )


            # Nowwe can save the results of this iteration in a CSV 
            # Create directory structure and CSV filename
            stats_dir = "Stats"
            func_dir = os.path.join(stats_dir, functoeval)
            os.makedirs(func_dir, exist_ok=True)
            csv_filename = os.path.join(func_dir, f"{decade_selector.__name__}.csv")

            if iteration == 0:
                # Create CSV file with headers for the first iteration
                with open(csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header row
                    header = ['iteration', 'seed', 'final_objective_value', 'n_iter_opt'] + [f'x{i}' for i in range(n_vars)]
                    writer.writerow(header)

            # Append results for current iteration
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Transform the optimal solution back to original space
                X_transformed = decade_selector(res.X, bounds)
                # Write data row
                row = [iteration, seed, res.F[0], res.algorithm.n_gen] + list(X_transformed)
                writer.writerow(row)



# Shut down Ray when done
ray.shutdown()

# Get the optimal solution and objective value
X_opt = res.X
F_opt = res.F

print(f"Optimal solution (raw): {X_opt}")
print(f"Best solution found: {decade_selector(X_opt,bounds)}")
print(f"Objective value: {evalfunc(decade_selector(X_opt,bounds))}")

# print(f"Best solution found: {X_opt}")
# print(f"Objective value: {F_opt}")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(callback.data["best"], '-o')
plt.xlabel('Generation')
plt.ylabel('Best Objective Value')
plt.title('Convergence History')
plt.grid(True)
plt.savefig('convergence_plot.png')
