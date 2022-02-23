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
We introduce a desynchronization in the buckets discarding.

# *VersionTachosSincronizados*
We use the same CUDA code mentioned above but we have synchronization in the buckets discarding.
