using GraphDynamics, OrdinaryDiffEqTsit5, Test, ForwardDiff

struct Particle end
function GraphDynamics.subsystem_differential(sys::Subsystem{Particle}, F, t)
    (;x, v, m) = sys
    dx = v
    dv = F/m
    SubsystemStates{Particle}(x=dx, v=dv)
end
function GraphDynamics.computed_properties_with_inputs(::Subsystem{Particle})
    (; a = (p, F) -> F/p.m)
end

GraphDynamics.initialize_input(::Subsystem{Particle}) = 0.0

struct Oscillator end
function GraphDynamics.subsystem_differential(sys::Subsystem{Oscillator}, F, t)
    (;x, v, x₀, m, k) = sys
    dx = v
    dv = (F - k*(x - x₀))/m
    SubsystemStates{Oscillator}(x=dx, v=dv)
end
GraphDynamics.initialize_input(::Subsystem{Oscillator}) = 0.0
GraphDynamics.computed_properies(::Subsystem{Oscillator}) = (;ω₀ = ((;m, k),) -> √(k/m))


struct Spring
    k::Float64
end

function ((;k)::Spring)(a, b)
    k * (a.x - b.x)
end

struct Coulomb
    fac::Float64
end

function ((;fac)::Coulomb)(a, b)
    -fac * a.q * b.q * sign(a.x - b.x)/(abs(a.x - b.x) + 1e-10)^2
end


@testset "basic" begin
    sys = Subsystem{Oscillator}(states=(;x=-2.0, v=1.0), params=(;x₀=0.0, m=30.0, k=1.0, q=1.0))
    @test sys.x == -2.0
    @test sys.ω₀ == √(sys.k/sys.m)
end


function solve_particle_osc(;x1, x2, tspan = (0.0, 10.0), alg=Tsit5())
    # put some garbage values in here for states and params, but we'll set them to reasonable values later with the
    # u0map and param_map
    subsystems_partitioned = ([Subsystem{Particle}(states=(;x= NaN, v=0.0), params=(;m=1.0, q=1.0)),
                               Subsystem{Particle}(states=(;x=-1.0, v=Inf), params=(;m=2.0, q=1.0))],
                              [Subsystem{Oscillator}(states=(;x=-Inf, v=1.0), params=(;x₀=0.0, m=-3000.0, k=1.0, q=1.0))])

    states_partitioned = map(v -> map(get_states, v), subsystems_partitioned)
    params_partitioned = map(v -> map(get_params, v), subsystems_partitioned)
    names_partitioned = ([:particle1, :particle2], [:osc])

    spring_conns_par_par = NotConnected()
    spring_conns_par_osc = [Spring(1)
                            Spring(1);;]
    spring_conns_osc_par = [Spring(1) Spring(1)]
    spring_conns_osc_osc = NotConnected()

    spring_conns = ConnectionMatrix(((spring_conns_par_par, spring_conns_par_osc),
                                     (spring_conns_osc_par, spring_conns_osc_osc)))

    # Spring[⎡. .⎤ ⎡2⎤
    #        ⎣. .⎦ ⎣0⎦
    #        [2 0] [.]]

    coulomb_conns_par_par = [Coulomb(0) Coulomb(.05)
                             Coulomb(.05) Coulomb(0)]
    coulomb_conns_par_osc = [Coulomb(.05)
                             Coulomb(.05);;]
    coulomb_conns_osc_par = [Coulomb(.05) Coulomb(.05)]
    coulomb_conns_osc_osc = NotConnected()

    coulomb_conns = ConnectionMatrix(((coulomb_conns_par_par, coulomb_conns_par_osc),
                                      (coulomb_conns_osc_par, coulomb_conns_osc_osc)))

    # Coulomb[⎡0 1⎤ ⎡1⎤
    #         ⎣1 0⎦ ⎣1⎦
    #         [0 0] [.]]

    connection_matrices = ConnectionMatrices((spring_conns, coulomb_conns))

    sys = ODEGraphSystem(;connection_matrices, states_partitioned, params_partitioned, names_partitioned)

    prob = ODEProblem(sys,
                      # Fix the garbage state values
                      [:particle1₊x => x1, :particle2₊x => x2, :particle2₊v => 0.0, :osc₊x => 0.0],
                      tspan,
                      # fix the garbage param values
                      [:osc₊m => 3.0])
    sol = solve(prob, alg)
end

@testset "solutions" begin
    t = 10.0
    sol = solve_particle_osc(;x1=1.0, x2=-1.0, tspan=(0.0, t))

    @test sol(t; idxs=:particle1₊x) ≈  0.580617 rtol=1e-3
    @test sol(t; idxs=:particle2₊x) ≈ -1.391576 rtol=1e-3
    @test sol(t; idxs=:osc₊x)       ≈ -1.063306 rtol=1e-3
    
    k = GraphDynamics.getp(sol, :osc₊k)(sol)
    m = GraphDynamics.getp(sol, :osc₊m)(sol)
    @test sol(t, idxs=:osc₊ω₀) == √(k/m)

    # :particle1₊a is a computed property that depends on inputs (F). It should be equal to
    # the second derivative of the position (since it's an acceleration)
    # Note that sol(t, Val{2}) means "second derivative of sol at t" (using the interpolation).
    @test sol(t; idxs=:particle1₊a) ≈ sol(t, Val{2}; idxs=:particle1₊x) rtol=1e-5
end

@testset "sensitivies" begin
    jac = ForwardDiff.jacobian([1.0, -1.0]) do (x1, x2)
        sol = solve_particle_osc(;x1, x2)
        [sol[:particle1₊x, end], sol[:particle2₊x, end], sol[:osc₊x, end]]
    end
    @test jac ≈ [-0.50821   -0.740152
                 -0.199444  -0.906593
                 -0.586021   0.118173] rtol=1e-3

end

