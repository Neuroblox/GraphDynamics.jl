using GraphDynamics, OrdinaryDiffEq

struct Particle end
function GraphDynamics.subsystem_differential(sys::Subsystem{Particle}, F, t)
    (;x, v, m) = sys
    dx = v
    dv = F/m
    SubsystemStates{Particle}((;x=dx, v=dv))
end
GraphDynamics.initialize_input(::Subsystem{Particle}) = 0.0

struct Oscillator end
function GraphDynamics.subsystem_differential(sys::Subsystem{Oscillator}, F, t)
    (;x, v, x₀, m, k) = sys
    dx = v
    dv = (F - k*(x - x₀))/m
    SubsystemStates{Oscillator}((;x=dx, v=dv))
end
GraphDynamics.initialize_input(::Subsystem{Oscillator}) = 0.0

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

subsystems_partitioned = ([Subsystem{Particle}(states=(;x= 1.0, v=0.0), params=(;m=1.0, q=1.0)),
                           Subsystem{Particle}(states=(;x=-1.0, v=0.0), params=(;m=2.0, q=1.0))],
                          [Subsystem{Oscillator}(states=(;x=0.0, v=1.0), params=(;x₀=0.0, m=3.0, k=1.0, q=1.0))])

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
tspan = (0.0, 20.0)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob, Tsit5())

@test sol[:particle1₊x][end] ≈ 1.4923823131014389
@test sol[:particle2₊x][end] ≈ -0.11189010002787175
@test sol[:osc₊x][end] ≈ 1.3175449091469553
