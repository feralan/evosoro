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
import random
import numpy as np
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
from evosoro.tools.utils import count_occurrences, make_material_tree
from evosoro.tools.checkpointing import continue_from_checkpoint


VOXELYZE_VERSION = '_voxcad_mod'
# sub.call("rm ./voxelyze", shell=True)
sub.call("cp ../" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)  # Making sure to have the most up-to-date version of the Voxelyze physics engine
# sub.call("chmod 755 ./voxelyze", shell=True)
# sub.call("cp ../" + VOXELYZE_VERISON + "/qhull .", shell=True)  # Auxiliary qhull executable, used in some experiments to compute the convex hull of the robot
# sub.call("chmod 755 ./qhull", shell=True)  # Execution right for qhull


NUM_RANDOM_INDS = 1  # Number of random individuals to insert each generation
MAX_GENS = 1000  # Number of generations
POPSIZE = 15  # Population size (number of individuals in the population)
IND_SIZE = (21, 21, 1)  # Bounding box dimensions (x,y,z). e.g. IND_SIZE = (6, 6, 6) -> workspace is a cube of 6x6x6 voxels
SIM_TIME = 5  # (seconds), including INIT_TIME!
INIT_TIME = 1
DT_FRAC = 0.9  # Fraction of the optimal integration step. The lower, the more stable (and slower) the simulation.

TIME_TO_TRY_AGAIN = 30  # (seconds) wait this long before assuming simulation crashed and resending
MAX_EVAL_TIME = 60  # (seconds) wait this long before giving up on evaluating this individual
SAVE_LINEAGES = False
MAX_TIME = 8  # (hours) how long to wait before autosuspending
EXTRA_GENS = 0  # extra gens to run when continuing from checkpoint

RUN_DIR = "grasper"  # Subdirectory where results are going to be generated
RUN_NAME = "Basic"
CHECKPOINT_EVERY = 1  # How often to save an snapshot of the execution state to later resume the algorithm
SAVE_POPULATION_EVERY = 1  # How often (every x generations) we save a snapshot of the evolving population

SEED = 1
random.seed(SEED)  # Initializing the random number generator for reproducibility
np.random.seed(SEED)


# Defining a custom genotype, inheriting from base class Genotype
class MyGenotype(Genotype):
    def __init__(self):
        # We instantiate a new genotype for each individual which must have the following properties
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        # The genotype consists of a single Compositional Pattern Producing Network (CPPN),
        # with multiple inter-dependent outputs determining the material constituting each voxel
        # (e.g. two types of active voxels, actuated with a different phase, two types of passive voxels, softer and stiffer)
        # The material IDs that you will see in the phenotype mapping dependencies refer to a predefined palette of materials
        # currently hardcoded in tools/read_write_voxelyze.py:
        # (0: empty, 1: passiveSoft, 2: passiveHard, 3: active+, 4:active-),
        # but this can be changed.
        self.add_network(CPPN(output_node_names=["weight"]))

        self.to_phenotype_mapping.add_map(name="weight", tag="<ClassWeight>",
                                          func=np.abs)

# Define a custom phenotype, inheriting from the Phenotype class
class MyPhenotype(Phenotype):
    def is_valid(self):
        return True


# Setting up the simulation object
my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)

# scenarios = {'pyramid_5' : pyramid_5, 'pyramid_7' : pyramid_7, 'cube_3' : cube_3, 'cube_5' : cube_5}
scenarios = {'pyramid_5' : pyramid_5, 'cube_3' : cube_3}
# Setting up the environment object
my_env = Env(sticky_floor=0, time_between_traces=0, floor_enabled=0, softest_material=1, fixed_shape=base_mat, scenarios=scenarios)

# Now specifying the objectives for the optimization.
# Creating an objectives dictionary
my_objective_dict = ObjectiveDict()


def evalClassPerf(classValues):
    d = np.diff(classValues)
    d = np.abs(d)
    d = np.sum(d)
    return d

# Adding an objective named "fitness", which we want to maximize. This information is returned by Voxelyze
# in a fitness .xml file, with a tag named "NormFinalDist"
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<ClassValue>", isArray=True, evalFun=evalClassPerf)

# Initializing a population of SoftBots
my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE)

# Setting up our optimization
my_optimization = ParetoOptimization(my_sim, my_env, my_pop)

# And, finally, our main
if __name__ == "__main__":

    # my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
    #                     directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
    #                     time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
    #                     save_vxa_every=SAVE_POPULATION_EVERY, save_lineages=SAVE_LINEAGES)

    # Here is how to use the checkpointing mechanism
    if not os.path.isfile("./" + RUN_DIR + "/checkpoint.pickle"):
        # start optimization
        my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                            directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                            time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                            save_vxa_every=SAVE_POPULATION_EVERY, save_lineages=SAVE_LINEAGES)

    else:
        continue_from_checkpoint(directory=RUN_DIR, additional_gens=EXTRA_GENS, max_hours_runtime=MAX_TIME,
                                 max_eval_time=MAX_EVAL_TIME, time_to_try_again=TIME_TO_TRY_AGAIN,
                                 checkpoint_every=CHECKPOINT_EVERY, save_vxa_every=SAVE_POPULATION_EVERY,
                                 save_lineages=SAVE_LINEAGES)
