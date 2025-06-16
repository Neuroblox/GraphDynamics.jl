module GraphDynamics

macro public(ex)
    if VERSION >= v"1.11.0-DEV.469"
        args = ex isa Symbol ? (ex,) : Base.isexpr(ex, :tuple) ? ex.args :
            error("malformed input to `@public`: $ex")
        esc(Expr(:public, args...))
    else
        nothing
    end
end

@public (
    subsystem_differential,
    apply_subsystem_noise!,
    subsystem_differential_requires_inputs,

    initialize_input,
    combine,

    has_continuous_events,
    continuous_event_condition,
    apply_continuous_event,
    continuous_events_require_inputs,

    has_discrete_events,
    discrete_event_condition,
    apply_discrete_event!,
    discrete_events_require_inputs,

    isstochastic,

    event_times,
    ForeachConnectedSubsystem,

    computed_properties,
    computed_properties_with_inputs,

    system_wiring_rule!,
    to_subsystem,

    get_name,
    connection_property_namemap,

    make_connection_matrices
)

export
    Subsystem,
    SubsystemParams,
    SubsystemStates,
    GraphSystem,
    PartitionedGraphSystem,
    get_tag,
    get_states,
    get_params,
    ODEProblem,
    SDEProblem,
    ConnectionMatrices,
    ConnectionMatrix,
    NotConnected,
    ConnectionRule,
    connections,
    add_connection!,
    add_node!,
    nodes,
    has_connection,
    delete_connection!


#----------------------------------------------------------
using Base: @kwdef, @propagate_inbounds
using Base.Iterators: map as imap

using Base.Cartesian: @nexprs

using OhMyThreads:
    OhMyThreads,
    SerialScheduler,
    tforeach,
    tmapreduce,
    DynamicScheduler

using SciMLBase:
    SciMLBase,
    ODEProblem,
    SDEProblem,
    CallbackSet,
    VectorContinuousCallback,
    ContinuousCallback,
    DiscreteCallback,
    remake,
    ODEFunction,
    SDEFunction

using SymbolicIndexingInterface:
    SymbolicIndexingInterface,
    setu,
    setp,
    getp,
    observed

using Accessors:
    Accessors,
    @set,
    @reset,
    @insert

using ConstructionBase:
    ConstructionBase,
    setproperties

using SparseArrays:
    SparseArrays,
    SparseMatrixCSC,
    nonzeros,
    findnz,
    rowvals,
    nzrange,
    sparse,
    spzeros,
    dropzeros!

using OrderedCollections:
    OrderedCollections,
    OrderedDict


#----------------------------------------------------------
# Random utils
include("utils.jl")

#----------------------------------------------------------
# API functions to be implemented by new Systems

struct SubsystemStates{T, Eltype, States <: NamedTuple} <: AbstractVector{Eltype}
    states::States
end

struct SubsystemParams{T, Params <: NamedTuple}
    params::Params
end

"""
    Subsystem{T, Eltype, StateNT, ParamNT}

A `Subsystem` struct describes a complete subcomponent to an `GraphSystem`. This stores a `SubsystemStates` to describe the continuous dynamical state of the subsystem, and a `GraphSystemParams` which describes various non-dynamical parameters of the subsystem. The type parameter `T` is the subsystem's \"tag\" which labels what sort of subsystem it is.

See also `subsystem_differential`, `SubsystemStates`, `SubsystemParams`.
"""
struct Subsystem{T, Eltype, States, Params}
    states::SubsystemStates{T, Eltype, States}
    params::SubsystemParams{T, Params}
end

"""
    get_tag(subsystem::Subsystem{T}) = T
    get_tag(states::SubsystemStates{T}) = T
    get_tag(params::SubsystemParams{T}) = T
    get_tag(::Type{<:Subsystem{T}}) = T

Extract the type tag `T` from a subsystem or its components.

The tag identifies the subsystem type and is used for dispatch and organization.

# Examples
```julia
particle = Subsystem{Particle}(states=(x=1.0, v=0.0), params=(m=2.0,))
get_tag(particle)  # returns Particle
get_tag(typeof(particle))  # returns Particle
```
"""
function get_tag end
"""
    get_params(subsystem::Subsystem) -> SubsystemParams

Extract the parameters from a subsystem.

Returns a `SubsystemParams` object containing the non-dynamical parameters
(constants like mass, charge, spring constants, etc.) of the subsystem.

# Example
```julia
sys = Subsystem{Oscillator}(states=(x=1.0, v=0.0), params=(k=2.0, m=1.0))
params = get_params(sys)  # SubsystemParams{Oscillator}(k=2.0, m=1.0)
params.k  # 2.0
```
"""
function get_params end
"""
    get_states(subsystem::Subsystem) -> SubsystemStates

Extract the state variables from a subsystem.

Returns a `SubsystemStates` object containing the dynamical state variables
(position, velocity, concentrations, etc.) that evolve over time.

# Example
```julia
sys = Subsystem{Oscillator}(states=(x=1.0, v=0.0), params=(k=2.0, m=1.0))
states = get_states(sys)  # SubsystemStates{Oscillator}(x=1.0, v=0.0)
states.x  # 1.0
```
"""
function get_states end

"""
   computed_properties(::Subsystem{T}) :: NamedTuple{props, NTuple{N, funcs}}

Signal that a subsystem has properties which can be computed on-the-fly based on it's existing properties. In the termoinology used by ModelingToolkit.jl, these are "observed states".

This function takes in a `Subsystem` and returns a `NamedTuple` where each key is a property name that can be computed, and each value is a function that takes in the subsystem and returns a computed value.

By default, this function returns an empty NamedTuple.

Example:

```julia
struct Position end
function GraphDynamics.computed_properties(::Subsystem{Position})
    (;r = (;x, y) -> √(x^2 + y^2),
      θ = (;x, y) -> atan(y, x))
end

let sys = Subsystem{Position}(states=(x=1, y=2), params=(;))
    sys.r == √(sys.x^2 + sys.y^2)
end
```
"""
computed_properties(s::Subsystem) = (;)

# TODO: delete this in the next breaking release.
# Accidentally released this with computed_properies
# as the name to use
const computed_properies = computed_properties


"""
    computed_properties_with_inputs(s::Subsystem)

Signal that a subsystem has properties which can be computed on-the-fly based on it's existing properties. In the termoinology used by ModelingToolkit.jl, these are "observed states", but they also require the inputs to the subsystem to compute (and thus are non-local to the subsystem).

This function takes in a `Subsystem` and returns a `NamedTuple` where each key is a property name that can be computed, and each value is a function that takes in the subsystem and returns a computed value.

By default, this function returns an empty NamedTuple.

This is typically only usable on objects like ODESolutions.
"""
computed_properties_with_inputs(s::Subsystem) = (;)

"""
     subsystem_differential(subsystem, input, t)

Add methods to this function to describe the derivatives of a given subsystem at time `t`. This should take in `Subsystem{T}` and return a `SubsystemStates{T}` where each element corresponds to the derivative of the subsystem with respect to that element. The `input` argument will be the total of all the `combine`'d inputs from all Subsystems which are connected to `subsystem` in the system graph.
"""
function subsystem_differential end

function apply_subsystem_differential!(vstate, subsystem, input, t)
    vstate[] = subsystem_differential(subsystem, input, t)
end

"""
    subsystem_differential_requires_inputs(::Type{T})

defaults to false, but users may add methods to this function if they have subsystems for which the connection inputs are not required for generating the subsystem differential.
"""
subsystem_differential_requires_inputs(::Type{T}) where {T} = true
subsystem_differential_requires_inputs(::Subsystem{T}) where {T} = subsystem_differential_requires_inputs(T)

"""
    apply_subsystem_noise!(vstate, subsystem, t)

Apply stochastic noise to a subsystem's state at time `t`.

This function modifies the noise terms for a subsystem's stochastic differential equation.
By default, it does nothing (no noise). Override this for stochastic subsystems.

# Arguments
- `vstate`: A view into the noise vector to be modified
- `subsystem`: The `Subsystem` instance whose noise is being computed
- `t`: Current time

# Example
```julia
function GraphDynamics.apply_subsystem_noise!(vstate, sys::Subsystem{BrownianParticle}, t)
    # No noise in position, so we don't modify vstate[:x]
    vstate[:v] = sys.σ    # White noise in velocity with amplitude σ
end
```
"""
function apply_subsystem_noise!(vstate, subsystem, t)
    nothing
end


# """
#     must_run_before(::Type{T}, ::Type{U})

# Overload this function to tell the ODE solver that subsystems of type `T` must run before subsystems of type `U`. Default `false`.
# """
# must_run_before(::Type{T}, ::Type{U}) where {T, U} = false

function continuous_event_condition end
function apply_continuous_event! end
"""
    discrete_event_condition(subsystem, t, F)
    discrete_event_condition(connection_rule, t)

Check if a discrete event should trigger at time `t`.

Returns `true` if the event should trigger, `false` otherwise.

# Arguments
- `subsystem`: The `Subsystem` to check for event conditions
- `t`: Current time
- `F`: A `ForeachConnectedSubsystem` callable for accessing connected subsystems
- `connection_rule`: For connection events, the `ConnectionRule` to check

# Implementation
Override this function for subsystems or connections with discrete events:
```julia
function GraphDynamics.discrete_event_condition(sys::Subsystem{Pendulum}, t, F)
    # Trigger event every 0.1 time units
    return t % 0.1 ≈ 0
end
```
"""
function discrete_event_condition end
"""
    apply_discrete_event!(integrator, sview, pview, subsystem, F[, input])
    apply_discrete_event!(integrator, sview_src, pview_src, sview_dst, pview_dst, 
                         connection_rule, sys_src, sys_dst[, input_src, input_dst])

Apply a discrete event to modify subsystem states or parameters.

This function is called when `discrete_event_condition` returns `true`.

# Arguments
- `integrator`: The ODE/SDE integrator
- `sview`/`pview`: Views into states/params to be modified
- `subsystem`: The subsystem experiencing the event
- `F`: `ForeachConnectedSubsystem` for accessing connected subsystems
- `input`: Optional input from connected subsystems
- For connection events: views and systems for both source and destination

# Example
```julia
function GraphDynamics.apply_discrete_event!(integrator, sview, pview, 
                                            sys::Subsystem{Ball}, F)
    # Bounce: reverse velocity
    sview[:v] = -sys.restitution * sview[:v]
end
```
"""
function apply_discrete_event! end


"""
    initialize_input(::Subsystem{T})

generate the neutral element for inputs to a given `Subsystem` such that

```julia
acc = initialize_input(sys)
combine(acc, another_input) == another_input
```

If `combine` just adds together inputs, then `initalize_input` should be zero or a collection of zeros.
"""
function initialize_input end

"""
    combine(input1, input2)

When a `Subsystem` is connected to multiple other subsystems, all of the inputs sent to that `Subsystem` via the connections must be `combine`'d together into one input representing the accumulation of all of the inputs. `combine` is the function used to accumulate these inputs together at each step. Defaults to addition, but can have methods added to it for more exotic input types.
"""
combine(x::Number, y::Number) = x + y
function combine(x::NamedTuple{names}, y::NamedTuple{names}) where {names}
    NamedTuple{names}(combine.(Tuple(x), Tuple(y)))
end


"""
    isstochastic(::Type{T})

Defaults to `false`, but for subsystems which define a stochastic differential equation, you must add a method to this function of the form

    GraphDynamics.isstochastic(::Type{Mytype}) = true
"""
isstochastic(::Subsystem{T}) where {T} = isstochastic(T)
isstochastic(::T) where {T} = isstochastic(T)
isstochastic(::Type{T}) where {T} = false
isstochastic(::Type{Union{}}) = error("This should be unreachable!")


for s ∈ [:continuous, :discrete]
    has_events = Symbol(:has_, s, :_events)
    events_require_inputs = Symbol(s, :_events_require_inputs)
    @eval begin
        $has_events(::Subsystem{T}) where {T} = $has_events(T)
        $has_events(::Type{Union{}}) = error("This should be unreachable!")
        $has_events(::Type{<:Subsystem{T}}) where {T} = $has_events(T)
        $has_events(::Type{<:SubsystemStates{T}}) where {T} = $has_events(T)
        $has_events(::Type{T}) where {T} = false

        $events_require_inputs(::Subsystem{T}) where {T} = $events_require_inputs(T)
        $events_require_inputs(::Type{<:Subsystem{T}}) where {T} = $events_require_inputs(T)
        $events_require_inputs(::Type{<:SubsystemStates{T}}) where {T} = $events_require_inputs(T)
        $events_require_inputs(::Type{Union{}}) = error("This should be unreachable!")
        $events_require_inputs(::Type{T}) where {T} = false
    end
end

"""
    event_times(::T) = ()

add methods to this function if a subsystem or connection type has a discrete event that triggers at pre-defined times. This will be used to add `tstops` to the `ODEProblem` or `SDEProblem` automatically during `GraphSystem` construction. This is vital for discrete events which only trigger at a specific time.
"""
event_times(::Any) = ()

abstract type ConnectionRule end
(c::ConnectionRule)(src, dst, t) = c(src, dst)
Base.zero(::T) where {T <: ConnectionRule} = zero(T)

struct NotConnected{CR <: ConnectionRule} end
Base.getindex(::NotConnected{CR}, inds...) where {CR} = zero(CR)
Base.copy(c::NotConnected) = c
"""
    ConnectionMatrix{N, CR, Tup}

A block matrix representing connections between N different subsystem types using
connection rules of type `CR`.

# Structure
The matrix is organized as blocks where `data[i][j]` contains connections from
subsystems of type `i` to subsystems of type `j`:

```
        dst type 1    dst type 2    dst type 3
       ┌─────────────┬─────────────┬─────────────┐
type 1 │ data[1][1]  │ data[1][2]  │ data[1][3]  │
       ├─────────────┼─────────────┼─────────────┤
type 2 │ data[2][1]  │ data[2][2]  │ data[2][3]  │ src
       ├─────────────┼─────────────┼─────────────┤
type 3 │ data[3][1]  │ data[3][2]  │ data[3][3]  │
       └─────────────┴─────────────┴─────────────┘
```

Each block is either:
- A sparse matrix of `ConnectionRule` objects
- `NotConnected{CR}()` if no connections exist

# Example
```julia
# Block [2,1] has Spring connections from type 2 → type 1
# Block [1,2] has no connections
ConnectionMatrix with Spring connections:
[1,1]: 2×2 sparse matrix
[1,2]: NotConnected
[2,1]: 3×2 sparse matrix
[2,2]: NotConnected
```

See also: [`ConnectionMatrices`](@ref), [`NotConnected`](@ref)
"""
struct ConnectionMatrix{N, CR, Tup <: NTuple{N, NTuple{N, Union{NotConnected{CR}, AbstractMatrix{CR}}}}}
    data::Tup
end
function Base.copy(c::ConnectionMatrix)
    data′ = map(c.data) do col
        map(col) do mat
            copy(mat)
        end
    end
    ConnectionMatrix(data′)
end

"""
    ConnectionMatrices{NConn, Tup}

Container for multiple connection matrices, each representing a different connection type.

Stores `NConn` different `ConnectionMatrix` objects, allowing heterogeneous connection
types in the same graph system.

# Structure
```
ConnectionMatrices
├── matrices[1]: ConnectionMatrix for connection type 1
├── matrices[2]: ConnectionMatrix for connection type 2
└── matrices[...]: Additional connection matrices
```

# Example
For a system with springs and dampers:
```
ConnectionMatrices with 2 connection types:
[1] Spring connections
[2] Damper connections
```

See also: [`ConnectionMatrix`](@ref)
"""
struct ConnectionMatrices{NConn, Tup <: NTuple{NConn, ConnectionMatrix}}
    matrices::Tup
end
Base.eachindex(cm::ConnectionMatrices{NConn}) where {NConn} = 1:NConn
@inline Base.getindex(m::ConnectionMatrix, i, j) = m.data[i][j]
Base.getindex(m::ConnectionMatrix, ::Val{i}, ::Val{j}) where {i, j} = m.data[i][j]
@inline Base.getindex(m::ConnectionMatrices, i) = m.matrices[i]
Base.length(m::ConnectionMatrices) = length(m.matrices)
Base.size(m::ConnectionMatrix{N}) where {N} = (N, N)
Base.copy(c::ConnectionMatrices) = ConnectionMatrices(map(copy, c.matrices))
rule_type(::ConnectionMatrix{N, CR}) where {N, CR} = CR


"""
    get_name(x)::Symbol

Get the symbolic name of an input node before conversion to `Subsystem` via `to_subsystem`. Overload this function for
your custom types
"""
get_name(x)::Symbol = x.name

"""
    to_subsystem(x)::Subsystem

Implement this function to convert your custom node types to `GraphDynamics.Subsystem` objects.

____

Example

```julia
using GraphDynamics
using Base: @kwdef
@kwdef struct Particle
    name::Symbol 
    m::Float64
    q::Float64=1.0
    x_init::Float64 = 0.0
    v_init::Float64 = 0.0
end
function GraphDynamics.to_subsystem(p::Particle)
    # Unpack the fields of the Particle
    (;name, m, q, x_init, v_init) = p
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
```
"""
function to_subsystem end
to_subsystem(sys::Subsystem) = sys
to_subsystem(::T) where {T} = error("Objects of type $T do not appear to be supported by GraphDynamics. This object must have custom `to_subsystem` method, `subsystem_differential`, and `initialize_inputs` methods.")


"""
    connection_property_namemap(conn, name_src, name_dst) :: NamedTuple

Add methods to this function for your custom connection types in order to support symbolic replacement and setting of connections.

This function should take in a connection instance `conn`, and the name of the source and destination nodes which `conn` connects,
and then should return a `NamedTuple` whose keys are property names of the connection, and the values are generated `Symbol`s.


For a connection rule of the form
```julia
struct Coulomb{T} <: ConnectionRule
    fac::T
end
```
the default implementation would give
```julia

julia> GraphDynamics.connection_property_namemap(Coulomb(1.0), :p1, :p2)
(:fac_Coulomb_p1_p2,)
```
but one can modify this function to follow alternative naming schemes as desired.

The naming scheme determined by this function is used through SymbolicIndexingInterface.jl to
modify or fetch connection values from SciML problems / solutions constructed through GraphDynamics. 
"""
function connection_property_namemap(conn::CR, name_src, name_dst) where CR
    pnames = propertynames(conn)
    vals = map(name -> Symbol(name, :_, nameof(CR), :_, name_src, :_, name_dst), propertynames(conn))
    NamedTuple{pnames}(vals)
end


#----------------------------------------------------------
# Infrastructure for subsystems
include("subsystems.jl")

#----------------------------------------------------------
# The GraphSystem type, and the stuff to turn it into a
# PartitionedGraphSystem
include("graph_system.jl")

#----------------------------------------------------------
# Problem generating API:
include("problems.jl")

#----------------------------------------------------------
# Symbolically indexing the solutions of graph systems
include("symbolic_indexing.jl")

#----------------------------------------------------------
# Solving graph differential equations
include("graph_solve.jl")

#----------------------------------------------------------

end # module GraphDynamics
