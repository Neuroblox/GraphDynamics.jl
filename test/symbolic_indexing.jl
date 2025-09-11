include("particle_osc_example.jl")
using SymbolicIndexingInterface

@testset "Symbolic Indexing of Vectors of observables" begin
    sol = solve_particle_osc(x1=1.0, x2=-1.0)

    a = getsym(sol, :particle1₊a)(sol)[end]
    ω = getsym(sol, :osc₊ω₀)(sol)[end]
    @test getsym(sol, [:osc₊ω₀, :particle1₊a])(sol)[end] == [ω, a]
end

@testset "setp and getp" begin
    prob = particle_osc_prob(; x1 = 1.0, x2 = 0.0)

    # Test setp works
    setp(prob, :particle1₊m)(prob, 2.0)
    @test getp(prob, :particle1₊m)(prob) == 2

    # Error on type-unstable change
    @test_throws ErrorException setp(prob, :particle1₊m)(prob, ones(3))

    # Remake
    prob = remake(prob, p = [:particle1₊m => 2 + 3im, :particle2₊m => 3 + 2im])
    setp(prob, :particle1₊m)(prob, 3 + 3.0im)
    @test getp(prob, :particle1₊m)(prob) == 3 + 3im
end
