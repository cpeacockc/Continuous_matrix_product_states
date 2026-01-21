using LinearAlgebra
#Code to find ground states of bosonic and Bose-Fermi mixture Hamiltonians in the continuous limit
#See PhysRevResearch.4.L022034 for analytical definitions and results using this code

#Initialization vectors for bosonic and mixture cases respectively
x_init_bose(D::Integer) = randn(3*D^2)
x_init_mix(D_2::Integer) = randn(4*(D_2^2)+(2D_2)^2+2*(D_2^2)+2)

#Rescaled energy as defined in PhysRevResearch.4.L022034 with hbar=1,m=1
rescaled_energy(E::Number,Dens::Real)=2*abs.(E)/Dens^3 

commutator(A,B) = A*B - B*A

#Function that takes vectorized R and Q matrices (x) of cMPS and outputs different expectation values
function EMixSelect(x::Array,D_2::Int64,g::Real,Gbb::Real,Gbf::Real,flag::String)
    D = 2*D_2 #bond dimension
    #Pauli matrices
    Sigma_plus = [0 1; 0 0]
    Upper_left = [1 0; 0 0]
    Lower_right = [0 0; 0 1]
    Sigma_z = [1 0; 0 -1]

    #A and B matrices used to define bosonic R_b = 
    A = reshape(Complex.(x[1:2:2*D_2^2-1],x[2:2:2*D_2^2]),D_2,D_2)
    B = reshape(Complex.(x[2*D_2^2+1:2:4*(D_2^2)-1],x[2*D_2^2+2:2:4*(D_2^2)]),D_2,D_2)
    
    #Make Hermitian form of A
    A_H = reshape(x[4*D_2^2+1:4*D_2^2+(2D_2)^2],D,D)
    A_H = Hermitian(Complex.(A_H,LowerTriangular(A_H)'-Diagonal(A_H)))
    
    #define \Gamma used for fermionic R_f
    Gamma = reshape(Complex.(x[4*D_2^2+(2D_2)^2+1:2:4*(D_2^2)+(2D_2)^2+2*(D_2^2)-1],x[4*D_2^2+(2D_2)^2+2:2:4*(D_2^2)+(2D_2)^2+2*(D_2^2)]),D_2,D_2)
    
    #bosonic and fermionic phase factors q_b and q_f 
    qb = x[4*(D_2^2)+(2D_2)^2+2*(D_2^2)+1]
    qf = x[4*(D_2^2)+(2D_2)^2+2*(D_2^2)+2]
    
    #due to variable overlap, this E is the D matrix used to define R_b
    E = pinv(Gamma)*A*Gamma #"D" in Bolech's pdf
    
    #Using all the above, define phase-modulated uniform Ansatz for bosonic and fermionic cMPS
    Rb = kron(Upper_left,A) + kron(Sigma_plus,B) + kron(Lower_right,E)
    Rf = kron(Sigma_plus,Gamma)

    #Define Q which gauge fixes a unit norm of the cMPS
    Q = -im*A_H - 0.5*(Rb'*Rb + Rf'*Rf)

    #Define T which defines the cMPS norm 
    T = kron(Q,Diagonal(ones(D))) +
        kron(Diagonal(ones(D)),conj(Q)) +
        kron(Rb,conj(Rb)) +
        kron(Rf,conj(Rf))

    #Now we construct the cMPS norm e^{TL} and iterate, squaring until converged at largest value
    ExpT = exp(T) 
    ExpTL = ExpT^2.0
    epsilon = norm(ExpTL-ExpT,2)/1_000_000
    while norm(ExpTL-ExpT,2) > epsilon
        ExpT = ExpTL
        ExpTL = ExpT*ExpT
    end

    #Define commutators for convenience
    commQRb = (Q*Rb - Rb*Q)
    commQRf = (Q*Rf - Rf*Q)

    #Define energy expectatoin values using cMPS Ansatz
    Ekinb = tr(ExpTL * kron((im*qb*Rb + commQRb),conj(im*qb*Rb + commQRb)))
    Ekinf = tr(ExpTL * kron((im*qf*Rf + commQRf),conj(im*qf*Rf + commQRf)))
    Eintb = tr(ExpTL * kron(Rb^2,conj(Rb^2)))
    Eintbf = tr(ExpTL * kron(Rb*Rf,conj(Rb*Rf)))

    #Define densities
    Densb = tr(ExpTL * kron(Rb,conj(Rb)))
    Densf = tr(ExpTL * kron(Rf,conj(Rf)))

    #Finally, define full Hamiltonian
    H = 0.5(Ekinb+Ekinf+g*(Gbb*Eintb+2*Gbf*Eintbf)) 
    
    if flag == "H"
        return  real(H)
    elseif flag == "Ekinb"
        return real(Ekinb)
    elseif flag == "Ekinf"
        return real(Ekinf)
    elseif flag == "Eintb"
        return real(Eintb)
    elseif flag == "Eintbf"
        return real(Eintbf)
    elseif flag == "Densb"
        return real(Densb)
    elseif flag == "Densf"
        return real(Densf)
        else 
        return "Whoops, that's not an option! :)"
    end
end

#Objective function for minimization, using definitions above
function EMix(x::Vector,grad::Vector,D_2::Int64,Nf::Float64,Ntot::Float64,g::Real,Gbb::Real,Gbf::Real)
    #For definitions, please see the function EMixSelect() above, for which they are identical
    D = 2*D_2 
    Sigma_plus = [0 1; 0 0]
    Upper_left = [1 0; 0 0]
    Lower_right = [0 0; 0 1]
    Sigma_z = [1 0; 0 -1]
    A = reshape(Complex.(x[1:2:2*D_2^2-1],x[2:2:2*D_2^2]),D_2,D_2)
    B = reshape(Complex.(x[2*D_2^2+1:2:4*(D_2^2)-1],x[2*D_2^2+2:2:4*(D_2^2)]),D_2,D_2)
    A_H = reshape(x[4*D_2^2+1:4*D_2^2+(2D_2)^2],D,D)
    A_H = Hermitian(Complex.(A_H,LowerTriangular(A_H)'-Diagonal(A_H)))
    Gamma = reshape(Complex.(x[4*D_2^2+(2D_2)^2+1:2:4*(D_2^2)+(2D_2)^2+2*(D_2^2)-1],x[4*D_2^2+(2D_2)^2+2:2:4*(D_2^2)+(2D_2)^2+2*(D_2^2)]),D_2,D_2)
    qb = x[4*(D_2^2)+(2D_2)^2+2*(D_2^2)+1]
    qf = x[4*(D_2^2)+(2D_2)^2+2*(D_2^2)+2]
    #qb = 0
    #qf = 0
    E = pinv(Gamma)*A*Gamma
    Rf = kron(Sigma_plus,Gamma)
    Rb = kron(Upper_left,A) + kron(Sigma_plus,B) + kron(Lower_right,E)
    Q = -im*A_H - 0.5*(Rb'*Rb + Rf'*Rf)
    T = kron(Q,Diagonal(ones(D))) +
        kron(Diagonal(ones(D)),conj(Q)) +
        kron(Rb,conj(Rb)) +
        kron(Rf,conj(Rf))
    L = 2.0
    ExpT = exp(T) 
    ExpTL = ExpT^L
    epsilon = norm(ExpTL-ExpT,2)/1_000_000
    while norm(ExpTL-ExpT,2) > epsilon
        ExpT = ExpTL
        ExpTL = ExpT*ExpT
    end
    commQRb = (Q*Rb - Rb*Q)
    commQRf = (Q*Rf - Rf*Q)
    Ekinb = tr(ExpTL * kron((im*qb*Rb + commQRb),conj(im*qb*Rb + commQRb)))
    Ekinf = tr(ExpTL * kron((im*qf*Rf + commQRf),conj(im*qf*Rf + commQRf)))
    Eintb = tr(ExpTL * kron(Rb^2,conj(Rb^2)))
    Eintbf = tr(ExpTL * kron(Rb*Rf,conj(Rb*Rf)))
    Densb = tr(ExpTL * kron(Rb,conj(Rb)))
    Densf = tr(ExpTL * kron(Rf,conj(Rf)))
    H_L = 0.5(Ekinb+Ekinf+g*(Gbb*Eintb+Gbf*2*Eintbf)) 
    
    CP_b = CP_f = 1e9 #Lagrange penalty
    Nb = -Nf + Ntot
    #Use Lagrange penalty to set densities 
    Lagrange_Constraints = CP_b*(real(Densb)-Nb)^2 + CP_f*(real(Densf)-Nf)^2 #energy function with lagrange constraints for Dens = 1 exactly how we did with bosons
    #Define objective function for energy minimization
    obj=H_L+Lagrange_Constraints
    return  real(obj)
end

function EBoseSelect(x::Vector,D::Real,g::Real,flag::String)

    #define R matrices used to construct bosonic cMPS Ansatz
    R = reshape(Complex.(x[1:2:2*D^2-1],x[2:2:2*D^2]),D,D)
    
    #Define Hermitian form of A
    A_H = reshape(x[2*D^2+1:3*D^2],D,D)
    A_H = Hermitian(Complex.(A_H,LowerTriangular(A_H)'-Diagonal(A_H)))
    
    #Gauge fixing
    Q = - im*A_H - 0.5*(R'*R)

    #Define T used in cMPS norm
    T = kron(Q,Diagonal(ones(D))) +
        kron(Diagonal(ones(D)),conj(Q)) +
        kron(R,conj(R))
    
    #To construct norm, square until convergence
    ExpT = exp(T)
    ExpTL = ExpT^2
    epsilon = norm(ExpTL-ExpT,1)/1_000_000
    while norm(ExpTL-ExpT,1) > epsilon
        ExpT = ExpTL
        ExpTL = ExpT^2
    end

    commQR = commutator(Q,R)

    #Define energy expectation values in cMPS 
    Ekin = tr(ExpTL * kron(commQR,conj(commQR)))
    Eint = tr(ExpTL * kron(R^2,conj(R^2)))
    Dens = tr(ExpTL * kron(R,conj(R))) 
    H = 0.5((Ekin+g*Eint)) 
    if flag == "H"
        return real(H)
    elseif flag == "Ekin"
        return real(Ekin)
    elseif flag == "Eint"
        return real(Eint)
    elseif flag == "Dens"
        return real(Dens)
        else 
        return "Whoops, that's not an option! :)"
    end
end

#Objective function for bosonic cMPS (Lieb-Liniger)
function EBose(x::Vector,grad::Vector,D::Real,g::Real,Nb::Real)
    #For definitions see EBoseSelect above
    R = reshape(Complex.(x[1:2:2*D^2-1],x[2:2:2*D^2]),D,D)
    A_H = reshape(x[2*D^2+1:3*D^2],D,D)
    A_H = Hermitian(Complex.(A_H,LowerTriangular(A_H)'-Diagonal(A_H)))
    Q = - im*A_H - 0.5*(R'*R)
    T = kron(Q,Diagonal(ones(D))) +
        kron(Diagonal(ones(D)),conj(Q)) +
        kron(R,conj(R))
    ExpT = exp(T) #try taylor expansion of exp
    ExpTL = ExpT^2.
    epsilon = norm(ExpTL-ExpT,1)/1_000_000
    while norm(ExpTL-ExpT,1) > epsilon
        ExpT = ExpTL
        ExpTL = ExpT^2
    end

    commQR = commutator(Q,R)
    Ekin = tr(ExpTL * kron(commQR,conj(commQR)))
    Eint = tr(ExpTL * kron(R^2,conj(R^2))) 
    Dens = tr(ExpTL * kron(R,conj(R)))
    e_c = 0.5((Ekin+g*Eint)) + 10_000*(Dens-Nb)^2 #modified to match mixtures case, c = g. (just multiplied by 0.5)
    return  real(e_c) # This is the output that will be used in the optimization
end
