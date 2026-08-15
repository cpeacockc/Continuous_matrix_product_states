using LinearAlgebra
#Exact Lieb-Liniger ground state energy from the Bethe ansatz, to benchmark the bosonic cMPS code against
#See Lieb and Liniger, Phys. Rev. 130, 1605 (1963)

#Gauss-Legendre nodes and weights on [-1,1], via the eigendecomposition of the Jacobi matrix
function gauss_legendre(n::Integer)
    b = [k/sqrt(4k^2-1) for k in 1:n-1]
    F = eigen(SymTridiagonal(zeros(n),b))
    F.values, 2*(F.vectors[1,:].^2)
end

#Solve the Lieb equation  g(x) = 1/2pi + (1/2pi) int_{-1}^{1} 2L/(L^2+(x-y)^2) g(y) dy
#by the Nystrom method: sample the integral on the quadrature nodes and solve the resulting linear system.
#Here L = c/K is the only free parameter - the Fermi rapidity K cancels out of both gamma and e,
#so sweeping L traces out the whole e(gamma) curve.
function lieb_liniger(L::Real; n::Integer=200)
    x,w = gauss_legendre(n)
    Kern = [(1/(2pi))*2L/(L^2+(x[i]-x[j])^2) for i in 1:n, j in 1:n]
    g = (I(n) - Kern .* w') \ fill(1/(2pi),n)
    dens = dot(w,g)
    (gamma = L/dens, e = dot(w, x.^2 .* g)/dens^3)
end

#gamma increases monotonically with L, so bisect (geometrically, since L ranges over many orders of magnitude)
#to get e at the gamma we actually want. Sanity check: e -> pi^2/3 as gamma -> infinity (Tonks-Girardeau).
function lieb_liniger_energy(gamma::Real; n::Integer=200)
    lo,hi = 1e-8,1e5
    for _ in 1:200
        mid = sqrt(lo*hi)
        lieb_liniger(mid,n=n).gamma < gamma ? (lo = mid) : (hi = mid)
    end
    lieb_liniger(sqrt(lo*hi),n=n).e
end
