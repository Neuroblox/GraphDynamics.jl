using SafeTestsets

@safetestset "Particle/Oscillator example" begin
    include("particle_osc_example.jl")
    solution_solve_test()
    sensitivity_test()
end
