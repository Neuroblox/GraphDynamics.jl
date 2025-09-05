include("test/particle_osc_example.jl")

@testset "Symbolic Indexing of Vectors" begin
    sol = solve_particle_osc(x1=1.0, x2=-1.0)

    a = getsym(sol, :particle1₊a)(sol)[end]
    ω = getsym(sol, :osc₊ω₀)(sol)[end]
    @test getsym(sol, [:osc₊ω₀, :particle1₊a])(sol)[end] == [ω, a]
end
