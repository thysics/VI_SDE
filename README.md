# In progress: Black Box Variational Inference for SDE

This code includes an implementation of black-box variational inference for SDE based on time-inhomogeneous OU process.

**Note that this repository is in active development.**

Here is a list of codes:
  1. class: variational_process, e.g. time-homogeneous and time-inhomogeneous OU process
  2. MLE estimation for both processes: to examine whether the resulting gradient of likelihood is correct
  
    a. Time-homogeneous OU variational process: EX01_MLE
    b. Time-inhomogeneous OU variaitonal process: EX01_MLE_TOU
    
  4. Demonstrations of Variational Inference on two examples including time-homogeneous OU process and double-well system SDE.
  
    a. Time-homogeneous OU process: EX02_VI
    b. double-well system SDE: EX03_VI
