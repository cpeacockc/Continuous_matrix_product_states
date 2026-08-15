include("cMPS_Mixtures.jl")
include("Lieb_Liniger_exact.jl")

#Bosonic cMPS code to find ground state for the Lieb-Liniger model

#First define system parameters
D = 4 # Bond-dimension
g = 2.0 # overall interaction strength (g=2 for reproducing Fig. 1 in PhysRevResearch.4.L022034)
Nb = 0.25 # Boson density (Nb=0.25 for reproducing Fig.1 in PhysRevResearch.4.L022034)
dens_tol = Nb/100 # tolerance for difference in density after optimization

#Use your favorite optimization routine (Here I use a PRAXIS implementation in NLopt.jl)
#WARNING: this is a global optimization problem. One should optimize by looping over many different random initial points, and compare results to find the true minimum. Using simulated annealing is also encouraged
#note: a Lagrange penalty is used to set density in EBose(), currently set to 10_000

using NLopt
x0 = x_init_bose(D) # initialize random cMPS ansatz which currently contains 3*D^2 numbers
opt = Opt(:LN_PRAXIS,length(x0));
maxeval!(opt, 200_000) # PRAXIS needs a generous budget; too small a cap breaks variational monotonicity in D
ftol_rel!(opt, 1e-13)
min_objective!(opt, (x,grad) -> EBose(x,grad,D,g,Nb))
@time (minf,minx,ret) = NLopt.optimize(opt,x0)


#Recalculate density and energy (cannot trust output from optimization due to lagrange multiplier)
Dens = real(EBoseSelect(minx,D,g,"Dens")); E=real(EBoseSelect(minx,D,g,"H"));

#Minimum dimensionless energy density (for more details and to compare results see PhysRevResearch.4.L022034)
Emin = rescaled_energy(E,Dens)

#Exact Bethe ansatz energy at the same gamma = g/n. cMPS is variational, so Emin should sit just above it
Eexact = lieb_liniger_energy(g/Dens)

if isapprox(Dens,Nb,atol=dens_tol)
    println("E = $Emin (exact: $Eexact)")
else
    error("Optimization failed (densities not within tolerance)")
end

#Now, find various energies of the ground state with EBoseSelect
Ekin = EBoseSelect(minx,D,g,"Ekin")
Eint = EBoseSelect(minx,D,g,"Eint")

