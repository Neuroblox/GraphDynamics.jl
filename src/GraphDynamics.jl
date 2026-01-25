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
    connection_needs_ctx,

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

    ArrayOfSubsystems,
    ArrayOfSubsystemStates
)

export
    Subsystem,
    SubsystemParams,
    SubsystemStates,
    GraphSystem,
    GraphSystemParameters,
    PartitioningGraphSystem,
    GraphNamemap,
    PartitionedIndex,
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
    flatten_graph,
    connection_equations,
    is_flat,
    PolyesterScheduler

#----------------------------------------------------------

using Base: @kwdef, @propagate_inbounds, isassigned, isstored
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
    observed,
    is_parameter,
    parameter_index

using Accessors:
    Accessors,
    @set,
    @reset,
    @insert,
    set,
    PropertyLens

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

using DiffEqBase:
    DiffEqBase,
    anyeltypedual

using FieldViews:
    FieldViews,
    FieldViewable,
    FieldView

using Polyester:
    Polyester,
    @batch


#----------------------------------------------------------
# Random utils
include("utils.jl")

struct PolyesterScheduler end

#----------------------------------------------------------
# API functions to be implemented by new Systems

struct SubsystemStates{T, Eltype, States <: NamedTuple} <: AbstractVector{Eltype}
    states::States
end
function FieldViews.fieldmap(p::Type{SubsystemStates{T, Elt, NamedTuple{state_names, tup}}}) where {T, Elt, state_names, tup}
    map(name -> :states => name, state_names)
end

struct SubsystemParams{T, Params <: NamedTuple}
    params::Params
end
function FieldViews.fieldmap(p::Type{SubsystemParams{T, NamedTuple{param_names, tup}}}) where {T, param_names, tup}
    map(name -> :params => name, param_names)
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
function FieldViews.fieldmap(::Type{Subsystem{T, Elt, <:NamedTuple{state_names}, <:NamedTuple{param_names}}}) where {T, Elt, state_names, param_names}
    state_map = map(name -> :states => :states => name, state_names)
    param_map = map(name -> :params => :params => name, param_names)
    (state_map..., param_map...)
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
   computed_properties(::Type{T}) :: NamedTuple{props, NTuple{N, funcs}}

Signal that a subsystem has properties which can be computed on-the-fly based on it's existing properties. In the termoinology used by ModelingToolkit.jl, these are "observed states".

This function takes in a tag type `T` and returns a `NamedTuple` where each key is a property name that can be computed, and each value is a function that takes in a `Subsystem{T}` and returns a computed value.

By default, this function returns an empty NamedTuple.

Example:

```julia
struct Position end
function GraphDynamics.computed_properties(::Type{Position})
    (;r = (;x, y) -> √(x^2 + y^2),
      θ = (;x, y) -> atan(y, x))
end

let sys = Subsystem{Position}(states=(x=1, y=2), params=(;))
    sys.r == √(sys.x^2 + sys.y^2)
end
```

Implementing this method will make the the reported properties work with SymbolicIndexingInterface.jl
"""
computed_properties(s::Type{<:Any}) = (;)
computed_properties(s::Type{Union{}}) = error("This should be unreachable")

"""
    computed_properties_with_inputs(::Type{T}) :: NamedTuple{props, NTuple{N, funcs}}

Signal that a subsystem has properties which can be computed on-the-fly based on it's existing properties. In the termoinology used by ModelingToolkit.jl, these are "observed states", but they also require the inputs to the subsystem to compute (and thus are non-local to the subsystem).

This function takes in a `Subsystem` and returns a `NamedTuple` where each key is a property name that can be computed, and each value is a function that takes in the subsystem and returns a computed value.

By default, this function returns an empty `NamedTuple`.

Implementing this method will make the the reported properties work with SymbolicIndexingInterface.jl.
"""
computed_properties_with_inputs(::Type{<:Any}) = (;)
computed_properties_with_inputs(s::Type{Union{}}) = error("This should be unreachable")

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
    vstate.v[] = sys.σ    # White noise in velocity with amplitude σ
end
```
"""
function apply_subsystem_noise!(vstate, subsystem, t)
    nothing
end


"""
    must_run_before(::Type{T}, ::Type{U})

Overload this function to tell the ODE solver that subsystems of type `T` must run before subsystems of type `U`. Default `false`.
"""
must_run_before(::Type{T}, ::Type{U}) where {T, U} = false

function continuous_event_condition end
function apply_continuous_event! end
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
        $has_events(::Type, ::Type, ::Type) = false

        $events_require_inputs(::Subsystem{T}) where {T} = $events_require_inputs(T)
        $events_require_inputs(::Type{<:Subsystem{T}}) where {T} = $events_require_inputs(T)
        $events_require_inputs(::Type{<:SubsystemStates{T}}) where {T} = $events_require_inputs(T)
        $events_require_inputs(::Type{Union{}}) = error("This should be unreachable!")
        $events_require_inputs(::Type{T}) where {T} = false
    end
end


"""
    event_times(::Subsystem{SysType}) = ()

add methods to this function if a subsystem type `SysType` has a discrete event that triggers at pre-defined times. This will be used to add `tstops` to the `ODEProblem` or `SDEProblem` automatically during `GraphSystem` construction. This is vital for discrete events which only trigger at a specific time.
"""
event_times(::Subsystem) = ()

"""
    event_times(::ConnType, ::Subsystem{SysSrc}, ::Subsystem{SysDst}) = ()

add methods to this function if a connection type `ConnType` has a discrete event that triggers at pre-defined times. This will be used to add `tstops` to the `ODEProblem` or `SDEProblem` automatically during `GraphSystem` construction. This is vital for discrete events which only trigger at a specific time.
"""
event_times(::Any, ::Any, ::Any) = ()

abstract type ConnectionRule end
Base.zero(::T) where {T <: ConnectionRule} = zero(T)

struct NotConnected{CR} end
Base.getindex(::NotConnected{CR}, inds...) where {CR} = zero(CR)
Base.eltype(::NotConnected{CR}) where {CR} = CR

Base.copy(c::NotConnected) = c
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

######## Equations
"""
    node_equations(::Subsystem{T}) where T

Output the differential equations for a node as LaTeX strings. Requires Latexify and Symbolics to be loaded.
"""
function node_equations end
"""
    graph_equations(::PartitionedGraphSystem)

Output the equations for a flattened graph system as LaTeX strings. Requires Latexify and Symbolics to be loaded.
"""
function graph_equations end
"""
    connection_equations(conn::ConnectionRule, src::Subsystem{U}, dst::Subsystem{T})

Output the equations for the connection between node `src` and node `dst`. Requires Latexify and Symbolics to be loaded.
"""
function connection_equations end

function graph_ode! end


"""
    connection_needs_ctx(conn::ConnectionRule) :: Bool

(default: `false`) determines if the call signature to `conn` should be of the form

    conn(src, dst, t, ctx::NamedTuple{states_partitioned, params_partitioned, connection_matrices})

or

    conn(src, dst, t)

Overload this function to return `true` if you have a connection rule type that needs access to the wider-graph
structure.
"""
@inline connection_needs_ctx(x) = false

struct StateIndex
    idx::Int
end
struct ParamIndex
    tup_index::Int
    v_index::Int
    prop::Symbol
end
struct CompuIndex
    tup_index::Int
    v_index::Int
    prop::Symbol
    requires_inputs::Bool
end
struct ConnectionIndex
    nc::Int
    i_src::Int
    i_dst::Int
    j_src::Int
    j_dst::Int
    connection_key::Union{Symbol, Nothing}
    prop::Union{Symbol, Nothing}
end

struct GraphNamemap
    state_namemap::OrderedDict{Symbol, StateIndex}
    param_namemap::OrderedDict{Symbol, ParamIndex}
    compu_namemap::OrderedDict{Symbol, CompuIndex}
    connection_namemap::OrderedDict{Symbol, ConnectionIndex}
end
function Base.copy(g::GraphNamemap)
    GraphNamemap(copy.((
        g.state_namemap,
        g.param_namemap,
        g.compu_namemap,
        g.connection_namemap
    ))...)
end

#----------------------------------------------------------
# Infrastructure for subsystems
include("subsystems.jl")

#----------------------------------------------------------
# The GraphSystem type
include("partitioning_graph_system.jl")
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
