using GraphDynamics, OrdinaryDiffEqTsit5, Test, ForwardDiff, FiniteDiff

using GraphDynamics
using Base: @kwdef

# Construct a type `Particle` which we'll later convert to GraphDynamics.Subsystem
# This represents a particle with mass `m` and charge `q`
@kwdef struct Particle
	name::Symbol 
	m::Float64
    q::Float64=1.0
	x_init::Float64 = 0.0
	v_init::Float64 = 0.0
end
function GraphDynamics.to_subsystem(p::Particle)
    # Unpack the fields of the Particle
	(;m, q, x_init, v_init) = p
    # Set the initial states to `x_init` and `v_init`
	states = SubsystemStates{Particle}(;
		x = x_init,
		v = v_init,
	)
    # Use `name`, `m`, and `q` as parameters
    # Every subsystem should have a unique name symbol.
    params = SubsystemParams{Particle}(
        ;m,
        q,
	)
    # Assemble a Subsystem
	Subsystem(states, params)
end

GraphDynamics.initialize_input(::Subsystem{Particle}) = (; F=0.0) # Default force on a `Particle` is 0.0
function GraphDynamics.subsystem_differential(sys::Subsystem{Particle}, input, t)
    (;x, v, m, q) = sys
    (;F) = input # Force `F` is the input to the subsystem
    dx = v   # Derivative of x is just v
    dv = F/m # Derivative of v is F/m (from F = m*a where a = dv/dt)
	
	# Return the differential of the current state:
    SubsystemStates{Particle}(;x=dx, v=dv) 
end
function GraphDynamics.computed_properties_with_inputs(::Type{Particle})
    a(sys, input) = input.F / sys.m 
    (; a)
end

# Construct a type `Oscillator` which we'll later convert to GraphDynamics.Subsystem
# This represents a particle attached to a spring anchored at `x₀` with spring constant `k`
@kwdef struct Oscillator
	name::Symbol
	m::Float64
	x₀::Float64
	k::Float64
	x_init = 0.0
	v_init = 0.0
end
function GraphDynamics.to_subsystem(p::Oscillator)
    # Unpack the fields of the Oscillator
	(;m, x₀, k, x_init, v_init) = p
    # Set the initial states to `x_init` and `v_init`
	states = SubsystemStates{Oscillator}(;
		x = x_init,
		v = v_init,
	)
    # Use `name`, `m`, `k`, `x₀` as parameters
    # Every subsystem should have a unique name symbol.
    params = SubsystemParams{Oscillator}(
		;m,
		k,
		x₀,
	)
	Subsystem(states, params)
end
GraphDynamics.initialize_input(::Subsystem{Oscillator}) = (; F = 0.0)
function GraphDynamics.subsystem_differential(sys::Subsystem{Oscillator}, input, t)
    (;x, v, x₀, m, k) = sys
    (;F) = input # Force `F` is the input to the subsystem
    dx = v                   # Derivative of x is just v
    dv = (F - k*(x - x₀))/m  # Derivative of v is the acceleration due to the input force, and the acceleration due to the spring.
    SubsystemStates{Oscillator}(;x=dx, v=dv)
end
GraphDynamics.computed_properties(::Type{Oscillator}) = (;ω₀ = ((;m, k),) -> √(k/m))

# Now lets construct some `ConnectionRule`s 
struct Spring{T} <: ConnectionRule
    k::T
end
Base.zero(::Type{Spring{T}}) where {T} = Spring(zero(T))

function ((;k)::Spring)(src::Subsystem, dst::Subsystem, t)
    # Calculate the force on subsystem `dst` due to being connected with
    # subsystem `src` by a spring with spring constant `k`.
    F = k * (src.x - dst.x)
    # Return the `input` being sent to the `dst` subsystem
    return (; F)
end

struct Coulomb{T} <: ConnectionRule
    fac::T
end
Base.zero(::Type{Coulomb{T}}) where {T} = Coulomb(zero(T))

function ((;fac)::Coulomb)(src::Subsystem, dst::Subsystem, t)
    # Calculate the Coulomb force on subsystem `dst` due to the charge of subsystem `src`
    F = -fac * src.q * dst.q * sign(src.x - dst.x)/(abs(src.x - dst.x))^2
    # Return the `input` being sent to the `dst` subsystem
    return (; F)
end

function GraphDynamics.connection_property_namemap(::Coulomb, name_src, name_dst)
    (; fac = Symbol(:fac_coulomb_, name_src, :_, name_dst))
end

function particle_osc_prob(;x1, x2, m=3.0, mp1=1.0, kc_p1_p2=1, tspan = (0.0, 10.0), alg=Tsit5())
    # put some garbage values in here for states and params, but we'll set them to reasonable values later with the
    # u0map and param_map
    particle1 = Particle(  name=:particle1, x_init= NaN, v_init=0.0, m=1.0, q=1.0)
    particle2 = Particle(  name=:particle2, x_init=-1.0, v_init=Inf, m=2.0, q=1.0)
    osc = Oscillator(name=:osc, x_init=-Inf, v_init=1.0, m=-3000.0, x₀=0.0, k=1.0)

    g = GraphSystem()

    add_connection!(g, particle1, osc; conn=Spring(1))
    add_connection!(g, particle2, osc; conn=Spring(1))
    add_connection!(g, osc, particle1; conn=Spring(1))
    add_connection!(g, osc, particle2; conn=Spring(1))

    add_connection!(g, particle1, particle2; conn=Coulomb(1))
    add_connection!(g, particle2, particle1; conn=Coulomb(1))

    # gsys._namemap[:w_particle1_particle2_spring] = 
    prob = ODEProblem(
        g,
        [:particle1₊x => x1, :particle2₊x => x2, :particle2₊v => 0.0, :osc₊x => 0.0
         ],
        tspan,
        (:osc₊m => m, :particle1₊m => mp1, :fac_coulomb_particle1_particle2 => kc_p1_p2)
    )
end
function solve_particle_osc(;reltol=nothing, saveat=nothing, kwargs...)
    prob = particle_osc_prob(; kwargs...)
    sol = solve(prob, Tsit5(); reltol)
end

function solution_solve_test()
    @testset "solutions" begin
        t = 10.0
        sol = solve_particle_osc(;x1=1.0, x2=-1.0, tspan=(0.0, 20.0), reltol=1e-8)

        @test sol(t; idxs=:particle1₊x) ≈ -0.071889 rtol=1e-3
        @test sol(t; idxs=:particle2₊x) ≈ -1.223721 rtol=1e-3
        @test sol(t; idxs=:osc₊x)       ≈ -0.728811 rtol=1e-3
        
        k = GraphDynamics.getp(sol, :osc₊k)(sol)
        m = GraphDynamics.getp(sol, :osc₊m)(sol)
        @test sol(t, idxs=:osc₊ω₀) == √(k/m)

        # :particle1₊a is a computed property that depends on inputs (F). It should be equal to
        # the second derivative of the position (since it's an acceleration)
        # Note that sol(t, Val{2}) means "second derivative of sol at t" (using the interpolation).
        @test sol(t-1.0; idxs=:particle1₊a) ≈ sol(t-1.0, Val{2}; idxs=:particle1₊x) rtol=1e-3
    end
end

function test_jacobian(f, v; rtol=1e-3)
    jac_autodiff   = ForwardDiff.jacobian(f, v)
    jac_finitediff = FiniteDiff.finite_difference_jacobian(f, v)
    @test jac_autodiff ≈ jac_finitediff rtol=rtol
end

function test_derivative(f, v; rtol=1e-3)
    d_autodiff   = ForwardDiff.derivative(f, v)
    d_finitediff = FiniteDiff.finite_difference_derivative(f, v)
    @test d_autodiff ≈ d_finitediff rtol=rtol
end

function sensitivity_test()
    # @testset "Sensitivities"
    begin
        # @testset "via constructor"
        begin
            test_jacobian([1.0, -1.0]) do (x1, x2)
                sol = solve_particle_osc(;x1, x2, reltol=1e-8)
                [sol[:particle1₊x][end], sol[:particle2₊x][end], sol[:osc₊x][end]]
            end

            test_derivative(3.0) do m
                sol = solve_particle_osc(;x1=one(typeof(m)), x2=-1.0, m=m, reltol=1e-8)
                sol[:particle1₊x][end]
            end

            test_derivative(1.0) do fac
                sol = solve_particle_osc(;x1=1.0, x2=-1.0, kc_p1_p2=fac, reltol=1e-8)
                [sol[:particle1₊x][end], sol[:particle2₊x][end], sol[:osc₊x][end]]
            end
            
            test_jacobian([1.0, -1.0, 3.0, 1.0, 2.0]) do (x1, x2, m, kc_p1_p2, mp1)
                sol = solve_particle_osc(;x1, x2, m, kc_p1_p2, mp1, reltol=1e-8)
                [sol[:particle1₊x][end], sol[:particle2₊x][end], sol[:osc₊x][end]]
            end
        end

        # @testset "via remake"
        begin
            
            prob = particle_osc_prob(; x1=1.0, x2=-1.0)

            test_jacobian([1.0, -1.0]) do (x1, x2)
                prob2 = remake(prob; u0=[:particle1₊x => x1, :particle2₊x => x2])
                sol = solve(prob2, Tsit5(); reltol=1e-8)
                [sol[:particle1₊x][end], sol[:particle2₊x][end], sol[:osc₊x][end]]
            end

            test_derivative(3.0) do m
                prob2 = remake(prob; p=[:osc₊m => m])
                sol = solve(prob2, Tsit5(); reltol=1e-8)
                sol[:particle1₊x][end]
            end
            
            test_derivative(1.0) do k
                prob2 = remake(prob; p=[:k_Spring_particle1_osc => k])
                sol = solve(prob2, Tsit5(); reltol=1e-8)
                sol[:particle1₊x][end]
            end


            test_jacobian([1.0, -1.0, 3.0, 1.0, 2.0]) do (x1, x2, m, kc_p1_p2, mp1)
                prob2 = remake(prob; u0=[:particle1₊x => x1, :particle2₊x => x2], p=[:osc₊m => m, :particle1₊m => mp1, :fac_coulomb_particle1_particle2 => kc_p1_p2])
                sol = solve(prob2, Tsit5(); reltol=1e-8)
                [sol[:particle1₊x][end], sol[:particle2₊x][end], sol[:osc₊x][end]]
            end
        end
    end
end
