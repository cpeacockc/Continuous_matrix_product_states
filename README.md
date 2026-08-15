# ContinousMatrixProductStates

Code to find the ground states of mixtures of bosons and fermions using a continous matrix product state ansatz, as peformed in  	
https://doi.org/10.1103/PhysRevResearch.4.L022034

## Running it

You'll need NLopt.jl. Pick your case, set the parameters at the top of the file, and run it:

- `Find_ground_state_bosons.jl` for the bosonic (Lieb-Liniger) case
- `Find_ground_state_mixtures.jl` for the Bose-Fermi mixture

Both take their energy functions from `cMPS_Mixtures.jl`.

Bear in mind this is a global optimization problem. A single run from a single random start will usually just settle into a local minimum, so one should loop over many random initial points and keep the lowest energy found. Simulated annealing is also worth a try.

## Checking the answer

`Lieb_Liniger_exact.jl` solves the Bethe ansatz integral equation for the exact Lieb-Liniger ground state energy, and the bosonic script prints it next to the cMPS result:

```
E = 2.238666591415658 (exact: 2.146366663250587)
```

The cMPS energy is variational, so it can only ever sit above the exact one - it should come down towards it as you raise the bond dimension or optimize harder.
