# GraphDynamics.jl

GraphDynamics.jl is a tool for describing the dynamics of interacting collections of modular subsystems
in an efficient, and parallelizable way. The current main use-case of this package is as a backend for
[Neuroblox.jl](https://www.neuroblox.org/).

More thorough docs are coming soon, but here's a demo of solving a differential equation using GraphDynamics.jl:

______


First, lets define two different types of subsystems we might want to implement:

+ `Particle` which describes a freely moving particle in 1D
+ `Oscillator` which describes a particle anchored to a spring at a point `x₀`.

We'll do this by defining structs to label these two types of subsystems, and then adding methods to the function `GraphDynamics.subsystem_differential` and `GraphSystems.initialize_input`.

``` julia
using GraphDynamics

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
```

And now lets define two different types of 'connections' through which these subsystems can interact:
+ `Spring` which describes two objects being joined together by a spring
+ `Coulomb` which descrbies two objects repelling eachother by a 1/r^2 law (with a cutoff around r=0 to stop the system from blowing up).

``` julia
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
```

Now, lets suppose we want two `Particle`s and one `Oscillator`. GraphDynamics.jl needs its inputs to be partitioned by type, i.e. we need to give it a `Tuple` of `Vector`s where each vector has a concrete type, so we might write

``` julia
subsystems_partitioned = ([Subsystem{Particle}(states=(;x= 1.0, v=0.0), params=(;m=1.0, q=1.0)),
                           Subsystem{Particle}(states=(;x=-1.0, v=0.0), params=(;m=2.0, q=1.0))],
                          [Subsystem{Oscillator}(states=(;x=0.0, v=1.0), params=(;x₀=0.0, m=3.0, k=1.0, q=1.0))])

```

Now, lets suppose we want each particle to be connected to the oscillator by a spring. We can form a `ConnectionMatrix` for `Spring` connections as

``` julia
spring_conns_par_par = NotConnected()
spring_conns_par_osc = [Spring(1)
                        Spring(1);;]
spring_conns_osc_par = [Spring(1) Spring(1)]
spring_conns_osc_osc = NotConnected()

spring_conns = ConnectionMatrix(((spring_conns_par_par, spring_conns_par_osc),
                                 (spring_conns_osc_par, spring_conns_osc_osc)))

# Spring[⎡. .⎤ ⎡1⎤
#        ⎣. .⎦ ⎣1⎦
#        [1 1] [.]]

```
and lets suppose we want all of them to repel eachother with a Coulomb factor of `0.05` (but not repel themselves!). Then, we would write
``` julia

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
#         [1 1] [.]]

connection_matrices = ConnectionMatrices((spring_conns, coulomb_conns))
```

We can form a `ODEGraphSystem` by passing it the bundle of connection matrices, the states of each susystem, and the paramers of each subsystem:

``` julia
names_partitioned = ([:particle1, :particle2], [:osc])
states_partitioned = map(v -> map(get_states, v), subsystems_partitioned)
params_partitioned = map(v -> map(get_params, v), subsystems_partitioned)

sys = ODEGraphSystem(;connection_matrices, states_partitioned, params_partitioned, names_partitioned)
```

which can then be solved and plotted like so:

```julia
using OrdinaryDiffEq, Plots
tspan = (0.0, 20.0)
prob = ODEProblem(sys, [], tspan)

sol = solve(prob, Tsit5())

using Plots
plot(sol, idxs=[:particle1₊x, :particle2₊x, :osc₊x])
```

![the solution](./sol_example.png)
