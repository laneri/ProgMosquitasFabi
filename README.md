# ProgMosquitasFabi
We use the agent-based model written in Fortran by Fabiana Laguna to simulate the *Aedes aegypti* mosquito population and
we extend it to a spatially explicit agent-based model in CUDA language.

# FOLDERS
# *UnaManzanaCyFortran* 
C code by Ana Gramajo for the agent-based model.

# *VariasManzanas*
C++ code using Array of Structs (AoS) by Ana Gramajo for the agent-based model.
The number of blocks is introduced. 

# *esqueleto_cuda*
Example code for CUDA by Karina Laneri.

# *mosquitos-CUDA*
CUDA code by Ana Gramajo and Karina Laneri using Struct of Arrays (SoA) for a spatially explicit agent-based model. 

The implemented code consists of the following steps:
    (1) We define the input parameters related to the biological parameters of the species and the initial conditions of the program: number of blocks, number of buckets distributed per block, and the state of the mosquitoes, initially all alive. We introduce stochasticity into the system by considering a uniform probability distribution for some of the above-mentioned attributes. Each bucket contains one mosquito, so the initial number of mosquitoes is equivalent to the number of buckets. 
    (2) We create a Mosquito object with all the aforementioned attributes. For this purpose, we use an SoA (Struct of Arrays) to access the GPU memory efficiently and reduce the computational cost. Each array allows us to allocate a set of N data of the same type. We consider a one-dimensional array of type int[] for each attribute of the mosquito. 
    (3) We evolve the system for 400 days. Every day, we calculate the mosquito population after updating the following attributes:
            (a) The temperature dependence for the female agent oviposition. 
            (b) Mortality rates at different stages in the life cycle of Aedes  Aegypti (egg -> larvae -> pupa -> adult). 
            (c) The effectiveness of advertising campaigns by emptying a given number of buckets per block during the hottest season. We introduce a desynchronization in the buckets discarding.
            (d) The availability of the buckets or containers once they are emptied. We add a temporary delay measured in days. After this, the containers become available again for the female agent to lay eggs.
            (e) In terms of spatiality, each female agent can be moved to a nearby container in the same block or moved to a neighboring bucket to lay eggs. 
    (4) We show results.
    
The system starts with a fixed number of mosquitoes given by steps (1) and (2) and evolves in time according to step (3) by executing conditions (a)-(e). 
Daily, the number of live mosquitoes is computed, and dead mosquitoes are removed from the calculation. 

# *VersionTachosSincronizados*
We use the same CUDA code mentioned above but in step (3c) we have synchronization in the buckets discarding.
