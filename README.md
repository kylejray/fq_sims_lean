### FQ_sims repo

This repo is meant be be a small self-sufficient repo to play around with 
simulations relating to the SQUID circuit equations of motion.
It is more geared towards prototyping and exploration than full scale
simulations. Important things to know:

1 The main folder contains notebooks for running simulations of model systems
meant to represent the EOM's of our SQUID circuits, neglecting any dynamics or interactions from the readout/measurement device. The 1D_experimental_flip notebook has a bit flip protocol implemented that is yet to be optimized in terms of device parameters, but is a reasonable proof of concept. The RF_coupling notebook is more exploratory at this point.

2 The resources and tutorial directory has a writeup that explains where the equations of motion come from as well as a tutorial notebook for the protocol designing package. 

3 The source directory contains the python modules that the other notebooks rely on
