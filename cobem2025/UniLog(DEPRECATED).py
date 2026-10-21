from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os, csv, time, sys
import logging

# ESTEVAN
import dill
from pymoo.optimize import minimize
from pymoo.core.termination import NoTermination
from pymoo.util.display.output import Output
from pymoo.core.problem import Problem
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.operators.crossover.sbx import SBX
from pymoo.core.callback import Callback
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.core.repair import Repair
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.decomposition.aasf import AASF
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.unsga3 import UNSGA3
import ray
from pymoo.core.termination import TerminateIfAny       # logical OR combiner
from pymoo.termination.fmin import MinimumFunctionValueTermination

import funcs  # Import the functions from funcs.py

class LogUniformSampling(Sampling):
    """
    Produces an (n_samples, 20) matrix where

        X[:, 0:10]  = log-uniform magnitudes  |x_i|   (positive, in real space)
        X[:, 10:20] = sign coordinates s_i ∈ [-1, +1]

    The magnitude bounds are taken from problem.xl[0:10] / problem.xu[0:10]
    and must be strictly positive.  Sign bounds are assumed to be [-1, 1].
    """

    def _do(self, problem, n_samples, **kwargs):

        n_vars2 = int(problem.n_var/2)
        print(f"n_vars2: {n_vars2}")
        # -- split the bound vectors ---------------------------------------
        xl, xu = problem.bounds()                     # shape (20,)
        mag_lo, mag_hi = np.abs(xl[:n_vars2]), np.abs(xu[:n_vars2])   # |x| bounds
        # (sign bounds are typically [-1,1] so we don’t really need them)

        # -- sample log-uniform magnitudes ---------------------------------
        #   Draw log10(|x|) uniformly, then exponentiate back to real space
        log10_lo, log10_hi = np.log10(mag_lo), np.log10(mag_hi)
        mags = 10.0 ** np.random.uniform(log10_lo, log10_hi,
                                         size=(n_samples, n_vars2))

        # -- sample signs  --------------------------------------------------
        #   Start the run at the poles (+1 or −1) so the prior is symmetric
        signs = np.random.choice([-1.0, 1.0], size=(n_samples, n_vars2))

        # -- assemble the population matrix --------------------------------
        X = np.empty((n_samples, n_vars2*2), dtype=float)
        X[:, :n_vars2]  = mags                   # magnitudes in columns 0‥9
        X[:, n_vars2:]  = signs                  # signs       in columns 10‥19
        return X


# User initialization
n_pop = 100             # Population size
n_gen = 200             # Number of generations
n_threads = 14           # Number of threads for parallelization
seed= 4817                # Random seed for reproducibility
# simp = LHS()  # Sampling method for magnitudes and signs
simp = LogUniformSampling()  # Custom sampling method for log-uniform distribution
fobjmin = 1.0e-8  # Objective function minimum value

# 10 magnitudes and 10 signs
n_vars = 4  # Number of variables

uub=1E12  # Upper bounds true value for x
llb=1E-12  # Lower bounds true value for x

# Transformation used





@ray.remote
def func(x):
    yis = funcs.brown(x)  # Call the Powell function from funcs.py
    # yis = funcs.powell(x)  # Call the Powell function from funcs.py
    return yis

# prepare the bounds for the variables
unitary_bounds = [(llb, uub) for _ in range(n_vars//2)]  # Magnitude bounds
unitary_bounds += [(-1, 1) for _ in range(n_vars//2)]  # Sign bounds
print(unitary_bounds)

# Create bounds array with size n_vars, where each element is [lb, ub]
bounds = np.array([[llb, uub] for _ in range(n_vars)])

# Initialize Ray for parallel computing
ray.init(num_cpus=n_threads,local_mode=True)  # Specify number of CPUs to use


class MyProblem(Problem):
    def __init__(self, *args, **kwargs):
        # Define your optimization problem
        # n_var: number of variables
        # n_obj: number of objectives (1 for single-objective)
        lb = np.array([b[0] for b in unitary_bounds])
        ub = np.array([b[1] for b in unitary_bounds])
        print(lb,ub)
        super().__init__(n_var=n_vars, # !!!!!!!!!!!!!!!!
                         n_obj=1, 
                         n_ieq_constr=0,
                         xl=lb,
                         xu=ub)
    
    def _evaluate(self, X, out, *args, **kwargs):
        # Your objective function here
        # x is a 2D array with shape (n_population, n_var)
                # Submit tasks to Ray
        futures = [
            func.remote(x) for x in X
        ]

        result_array= ray.get(futures) # this is a blocking operation that waits for the results



        out["F"] = np.vstack(result_array)


# Callback to track optimization progress
class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []
        
    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F").min())



# Create the problem instance
problem = MyProblem()

# Algorithm setup
algorithm = PSO(
    pop_size=n_pop,
    sampling=simp,
    adaptive=True,
    pertube_best=True,
    output=SingleObjectiveOutput(),
)


# Create a termination criterion
term_by_gen  = get_termination("n_gen",  n_gen)               # counts generations  :contentReference[oaicite:0]{index=0}
term_by_fmin = MinimumFunctionValueTermination(fobjmin)    # single-obj only   :contentReference[oaicite:1]{index=1}

# --- 2) combine them with “OR” ---------------------------------------------
termination  = TerminateIfAny(term_by_gen, term_by_fmin)

# Setup callback
callback = MyCallback()

# Start timing the execution
start_time = time.time()

# Run the optimization
res = minimize(
    problem=problem,
    algorithm=algorithm,
    termination=termination,
    seed=seed,
    callback=callback,
    verbose=True,
    save_history=False,
    elementwise_evaluation=True,  # Use Ray for parallelization when True
)

# Calculate and print execution time
execution_time = time.time() - start_time
print(f"Optimization completed in {execution_time:.2f} seconds")

# Shut down Ray when done
ray.shutdown()

# Get the optimal solution and objective value
X_opt = res.X
F_opt = res.F

# Get the magnitudes (first half) and signs (second half)
n_vars_half = n_vars // 2
magnitudes = X_opt[:n_vars_half]
signs = np.sign(X_opt[n_vars_half:])

# Display the results in a readable format
print(f"Best solution found:")
print(f"  Magnitudes: {magnitudes}")
print(f"  Signs: {signs}")
# Display the actual values (magnitude * sign)
actual_values = magnitudes * signs
print(f"  Actual values: {actual_values}")
print(f"Objective value: {F_opt}")

# calculate the best objective value
xis = actual_values[0]
yis = actual_values[1]
f1 = 1.0e4 * xis * yis - 1.0
f2 = np.exp(-xis) + np.exp(-yis) - 1.0001
yiss= f1**2 + f2**2

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(callback.data["best"], '-o')
plt.xlabel('Generation')
plt.ylabel('Best Objective Value')
plt.title('Convergence History')
plt.grid(True)
plt.savefig('convergence_plot.png')
# plt.show()
