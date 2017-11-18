#!/usr/bin/python
"""

In this example we evolve running soft robots in a terrestrial environment using a standard version of the physics
engine (_voxcad). After running this program for some time, you can start having a look at some of the evolved
morphologies and behaviors by opening up some of the generated .vxa (e.g. those in
evosoro/evosoro/basic_data/bestSoFar/fitOnly) with ./evosoro/evosoro/_voxcad/release/VoxCad
(then selecting the desired .vxa file from "File -> Import -> Simulation")

The phenotype is here based on a discrete, predefined palette of materials, which are visualized with different colors
when robots are simulated in the GUI.

Materials are identified through a material ID:
0: empty voxel, 1: passiveSoft (light blue), 2: passiveHard (blue), 3: active+ (red), 4:active- (green)

Active+ and Active- voxels are in counter-phase.


Additional References
---------------------

This setup is similar to the one described in:

    Cheney, N., MacCurdy, R., Clune, J., & Lipson, H. (2013).
    Unshackling evolution: evolving soft robots with multiple materials and a powerful generative encoding.
    In Proceedings of the 15th annual conference on Genetic and evolutionary computation (pp. 167-174). ACM.

    Related video: https://youtu.be/EXuR_soDnFo

"""
#%%
import random
import numpy as np
from scipy import stats
import subprocess as sub
from functools import partial
import os
import sys

# Appending repo's root dir in the python path to enable subsequent imports
sys.path.append(os.getcwd() + "/../..")

from evosoro.progetto.shapes.base_mat import *
from evosoro.progetto.shapes.pyramid_5 import *
from evosoro.progetto.shapes.pyramid_7 import *
from evosoro.progetto.shapes.cube_3 import *
from evosoro.progetto.shapes.cube_5 import *

from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.networks import CPPN
from evosoro.softbot import Genotype, Phenotype, Population
from evosoro.tools.algorithms import ParetoOptimization
from evosoro.tools.utils import SetMatStiffness
from evosoro.tools.checkpointing import continue_from_checkpoint
from evosoro.tools.evaluation import JustSimulateDontEvaluate
from evosoro.tools.evaluation import justEvaluateDontSimulate


VOXELYZE_VERSION = '_voxcad_mod'
sub.call("cp ../" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)  # Making sure to have the most up-to-date version of the Voxelyze physics engine

        
        
#%% # -------------------------------------- Inizializzazione variabili simulazione fisica ----------------------------------------

#
NUM_RANDOM_INDS = 0  # Number of random individuals to insert each generation
MAX_GENS = 0  # Number of generations
POPSIZE = 1  # Population size (number of individuals in the population)
IND_SIZE = (21, 21, 1)  # Bounding box dimensions (x,y,z). e.g. IND_SIZE = (6, 6, 6) -> workspace is a cube of 6x6x6 voxels
SIM_TIME = 3  # (seconds), including INIT_TIME!
INIT_TIME = 1
DT_FRAC = 0.9  # Fraction of the optimal integration step. The lower, the more stable (and slower) the simulation.

TIME_TO_TRY_AGAIN = 54000  # (seconds) wait this long before assuming simulation crashed and resending
MAX_EVAL_TIME = 54000  # (seconds) wait this long before giving up on evaluating this individual
SAVE_LINEAGES = False
MAX_TIME = 8  # (hours) how long to wait before autosuspending
EXTRA_GENS = 0  # extra gens to run when continuing from checkpoint

CHECKPOINT_EVERY = 1  # How often to save an snapshot of the execution state to later resume the algorithm
SAVE_POPULATION_EVERY = 1  # How often (every x generations) we save a snapshot of the evolving population

SEED = 1
random.seed(SEED)  # Initializing the random number generator for reproducibility
np.random.seed(SEED)
#
#
# # Defining a custom genotype, inheriting from base class Genotype
class MyGenotype(Genotype):
    def __init__(self):
#         # We instantiate a new genotype for each individual which must have the following properties
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        self.add_network(CPPN(output_node_names=["weight"]))

        self.to_phenotype_mapping.add_map(name="weight", tag="<ClassWeight>",
                                          func=np.abs)
#
# # Define a custom phenotype, inheriting from the Phenotype class
class MyPhenotype(Phenotype):
    def is_valid(self):
        return True  
    
class AdditionalData(object):
    labels = []
    pressures = []
  
labels = [1, 1, 2, 2]
scenarios = {'pyramid_5' : pyramid_5, 'pyramid_7' : pyramid_7, 'cube_3' : cube_3, 'cube_5' : cube_5}
 
 
#%% # -------------------------------------- Generazione matrice delle pressioni 1 ----------------------------------------
  
RUN_DIR = "mat_soft"  # Subdirectory where results are going to be generated
RUN_NAME = "mat_soft"

additionalData_soft = AdditionalData()
additionalData_soft.labels = labels
try:
    loadedPressures_soft = np.loadtxt("./"+RUN_DIR+"/SoftMatNormPy.txt",delimiter = ',')
    additionalData_soft.pressures = loadedPressures_soft
except:

    # Setting up the simulation object
    my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)
    
    # Setting up the environment objects
    base_mat_soft = base_mat
    my_env_soft = Env(sticky_floor=0, time_between_traces=0, floor_enabled=0, softest_material=1, fixed_shape=base_mat_soft, scenarios=scenarios)
    
    # Now specifying the objectives for the optimization.
    # Creating an objectives dictionary
    my_objective_dict = ObjectiveDict()
    my_objective_dict.add_objective(name="fitness", maximize=None, tag="<ClassValue>", isArray=True)
    
    # Initializing a population of SoftBots
    pressure_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE)
    
     # Setting up our optimization
    my_optimization_soft = ParetoOptimization(my_sim, my_env_soft, pressure_pop)
    my_optimization_soft.evaluate = JustSimulateDontEvaluate
    #
    # # And, finally, our main
    if __name__ == "__main__":
        my_optimization_soft.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                            directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                            time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                            save_vxa_every=SAVE_POPULATION_EVERY, save_lineages=SAVE_LINEAGES)

    simPressures_soft = pressure_pop[0].fitness
    np.savetxt("./"+RUN_DIR+"/SoftMatPressures.txt",simPressures_soft,delimiter = ',')
    
    vpress = np.hstack(simPressures_soft)
    zpress = stats.zscore(vpress)+(np.mean(vpress)/np.std(vpress))
    simPressures_soft = np.reshape(-zpress,[len(labels),441])
    np.savetxt("./"+RUN_DIR+"/SoftMatNormPy.txt",simPressures_soft,delimiter = ',')
        
    additionalData_soft.pressures = simPressures_soft

#import matplotlib.image as mimg
#import matplotlib.pyplot as plt
#imTry = np.array(my_pop.individuals[1].pressione)
#imTry = np.reshape(imTry,[21,21])
#plt.imshow(-imTry)


#%% # -------------------------------------- Generazione matrice delle pressioni 2 ----------------------------------------
#
#
#RUN_DIR = "mat_hard"  # Subdirectory where results are going to be generated
#RUN_NAME = "mat_hard"
#
#additionalData_hard = AdditionalData()
#additionalData_hard.labels = labels
#
#try:
#    loadedPressures_hard = np.loadtxt("./"+RUN_DIR+"/HardMatNorm.txt",delimiter = ',')
#    additionalData_hard.pressures = loadedPressures_hard
#except:
#    # Setting up the simulation object
#    my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)
#        
#    # Setting up the environment objects
#    # cambio stiffness tappetino con funzione ad hoc "SetMatStiffness", importata da "Utils"
#    base_mat_hard = SetMatStiffness(base_mat,2) #PROBLEMA: SetMatStiffness modifica sia base_mat_hard che base_mat che base_mat_soft
#    my_env_hard = Env(sticky_floor=0, time_between_traces=0, floor_enabled=0, softest_material=1, fixed_shape=base_mat_hard, scenarios=scenarios)
#    
#    # Now specifying the objectives for the optimization.
#    # Creating an objectives dictionary
#    my_objective_dict = ObjectiveDict()
#    my_objective_dict.add_objective(name="fitness", maximize=None, tag="<ClassValue>", isArray=True)
#    
#    # Initializing a population of SoftBots
#    pressure_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE)
#    
#    my_optimization_hard = ParetoOptimization(my_sim, my_env_hard, pressure_pop)
#    my_optimization_hard.evaluate = JustSimulateDontEvaluate
#    
#    if __name__ == "__main__":
#        my_optimization_hard.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
#                            directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
#                            time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
#                            save_vxa_every=SAVE_POPULATION_EVERY, save_lineages=SAVE_LINEAGES)
#    
#    
#    simPressures_hard = pressure_pop[0].fitness
#    np.savetxt("./HardMatPressuresNew.txt",simPressures_hard,delimiter = ',')
#    
#    vpress = np.hstack(simPressures_hard)
#    zpress = stats.zscore(vpress)+(np.mean(vpress)/np.std(vpress))
#    simPressures_hard = np.reshape(-zpress,[len(labels),441])
#    np.savetxt("./HardMatNormPy.txt",simPressures_hard,delimiter = ',')
#    
#    additionalData_hard.pressures = simPressures_hard
#

#%% -------------------------- Esecuzione GA ---------------------------

NUM_RANDOM_INDS = 50  # Number of random individuals to insert each generation
MAX_GENS = 20  # Number of generations
POPSIZE = 500  # Population size (number of individuals in the population)
IND_SIZE = (21, 21, 1)  # Bounding box dimensions (x,y,z). e.g. IND_SIZE = (6, 6, 6) -> workspace is a cube of 6x6x6 voxels
SIM_TIME = 5  # (seconds), including INIT_TIME!
INIT_TIME = 1
DT_FRAC = 0.9  # Fraction of the optimal integration step. The lower, the more stable (and slower) the simulation.

TIME_TO_TRY_AGAIN = 180  # (seconds) wait this long before assuming simulation crashed and resending
MAX_EVAL_TIME = 180  # (seconds) wait this long before giving up on evaluating this individual
SAVE_LINEAGES = False
MAX_TIME = 8  # (hours) how long to wait before autosuspending
EXTRA_GENS = 0  # extra gens to run when continuing from checkpoint

CHECKPOINT_EVERY = 1  # How often to save an snapshot of the execution state to later resume the algorithm
SAVE_POPULATION_EVERY = 1  # How often (every x generations) we save a snapshot of the evolving population

def phenoClip(elements):
    return elements
    # return np.clip(elements, 0, 1000)

# Defining a custom genotype, inheriting from base class Genotype
class MyGenotype(Genotype):
    def __init__(self):
        # We instantiate a new genotype for each individual which must have the following properties
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        self.add_network(CPPN(output_node_names=["weight"]))

        self.to_phenotype_mapping.add_map(name="weight", tag="<ClassWeight>",
                                          func=phenoClip)

# Define a custom phenotype, inheriting from the Phenotype class
class MyPhenotype(Phenotype):
    def is_valid(self):
        return True


#Declare list of seeds
#SEEDS = list(range(1,76,5))
SEEDSLIST = list(range(1,7,5))

if os.path.exists("./smartMat") == False:
    os.mkdir("./smartMat")

for SEED in SEEDSLIST:
    
    RUN_DIR = "smartMat/seed_"+str(SEED)  # Subdirectory where results are going to be generated
    RUN_NAME = "smartMat_seed_"+str(SEED)

    
    
    # Setting up the simulation object
    my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)
    
    # Setting up the environment object
    my_env = Env(sticky_floor=0, time_between_traces=0, floor_enabled=0, softest_material=1)
    
    # Now specifying the objectives for the optimization.
    # Creating an objectives dictionary
    my_objective_dict = ObjectiveDict()
    my_objective_dict.add_objective(name="fitness", maximize=True, tag=None)
    # my_objective_dict.add_objective(name="intraDistance", maximize=False, tag=None)
    my_objective_dict.add_objective(name="sumOfAllWeights", maximize=False, tag=None)
    
    # Initializing a population of SoftBots
    my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE)
    
    # Setting up our optimization
    my_optimization = ParetoOptimization(my_sim, my_env, my_pop)
    my_optimization.evaluate = justEvaluateDontSimulate
    
    # And, finally, our main
    if __name__ == "__main__":
        my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                            directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                            time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,save_pareto = True,
                            save_vxa_every=SAVE_POPULATION_EVERY, save_lineages=SAVE_LINEAGES, additionalData=additionalData_soft)