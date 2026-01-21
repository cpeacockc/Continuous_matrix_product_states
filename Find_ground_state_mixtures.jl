include("cMPS_Mixtures.jl")

#First define system parameters
D_2 = 2 # D/2 where D is bond dimension - this forces you to keep D even
g = 2.0 # overall interaction strength, set to 2 in PhysRevResearch.4.L022034
Gbb = 1.0 # boson-boson interaction strength - keep positive to prevent bosonic collapse
Gbf = 1.0 # boson-fermion interaction strength
Nf = 0.125 # Fermion density
Ntot = 0.25 # Total density
Nb = Ntot-Nf # Boson density
dens_tol = Ntot/100 # tolerance for difference in density after optimization

#Use your favorite optimization routine (Here I use a PRAXIS implementation in NLopt.jl)
#WARNING: this is a global optimization problem. One should optimize by looping over many different random initial points, and compare results to find the true minimum. Using simulated annealing is also encouraged
#note: Lagrange multipliers are used to set densities in EMix(), currently set to CP_b = CP_f = 1e9

using NLopt
x0 = x_init(D_2) # initialize random cMPS ansatz which currently contains 4*(D_2^2)+(2D_2)^2+2*(D_2^2)+2 numbers
opt = Opt(:LN_PRAXIS,length(x0));
min_objective!(opt, (x,grad) -> EMix(x,grad,D_2,Nf,Ntot,g,Gbb,Gbf))
@time (minf,minx,ret) = NLopt.optimize(opt,x0)


#Recalculate density and energy (cannot trust output from optimization due to lagrange multiplier)
Densb = EMixSelect(minx,D_2,g,Gbb,Gbf,"Densb");Densf = EMixSelect(minx,D_2,g,Gbb,Gbf,"Densf");E = EMixSelect(minx,D_2,g,Gbb,Gbf,"H");
#Minimum energy:
Emin = rescaled_energy(E,(Densb+Densf))

if isapprox(Densb,Nb,rtol=dens_tol) && isapprox(Densf,Nf,rtol=dens_tol)
    println("Emin = $Emin")
else
    error("Optimization failed (densities not within tolerance)")
end

#Now, find various energies of the ground state with EMixSelect
Ekinb = EMixSelect(minx,D_2,g,Gbb,Gbf,"Ekinb")
Ekinf = EMixSelect(minx,D_2,g,Gbb,Gbf,"Ekinf")
Eintb = EMixSelect(minx,D_2,g,Gbb,Gbf,"Eintb")
Eintbf = EMixSelect(minx,D_2,g,Gbb,Gbf,"Eintbf")
