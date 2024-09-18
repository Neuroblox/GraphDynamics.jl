module GraphDynamics

macro public(ex)
    if VERSION >= v"1.11.0-DEV.469"
        args = ex isa Symbol ? (ex,) : Base.isexpr(ex, :tuple) ? ex.args : error("malformed input to `@public`: $ex")
        esc(Expr(:public, args...))
    else
        nothing
    end
end

@public (
    Subsystem,
    SubsystemParams,
    SubsystemStates,
    
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


    must_run_before,

    isstochastic,
    
    GraphSystem,
    ODEGraphSystem,
    SDEGraphSystem,
    
    get_states,
    get_params,
    event_times,

    ConnectionMatrices,
    ConnectionMatrix,
    NotConnected,
)

export
    Subsystem,
    SubsystemParams,
    SubsystemStates,
    GraphSystem,
    ODEGraphSystem,
    SDEGraphSystem,
    get_states,
    get_params,
    ODEProblem,
    SDEProblem,
    ConnectionMatrices,
    ConnectionMatrix,
    NotConnected


#----------------------------------------------------------
using Base: @kwdef, @propagate_inbounds

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
    DiscreteCallback

using RecursiveArrayTools: ArrayPartition

using SymbolicIndexingInterface:
    SymbolicIndexingInterface

using Accessors:
    Accessors,
    @set

using ConstructionBase:
    ConstructionBase

using SparseArrays:
    SparseArrays,
    SparseMatrixCSC,
    nonzeros,
    findnz,
    rowvals,
    nzrange

#----------------------------------------------------------
# Random utils
include("utils.jl")

#----------------------------------------------------------
# API functions to be implemented by new Systems

struct SubsystemParams{Name, Params <: NamedTuple}
    params::Params
end

struct SubsystemStates{Name, Eltype, States <: NamedTuple} <: AbstractVector{Eltype}
    states::States
end

struct Subsystem{Name, Eltype, States, Params}
    states::SubsystemStates{Name, Eltype, States}
    params::SubsystemParams{Name, Params}
end

function get_name end
function get_params end
function get_states end

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

function apply_subsystem_noise!(vstate, subsystem, t)
    nothing
end


"""
    must_run_before(::Type{T}, ::Type{U})

Overload this function to tell the ODE solver that subsystems of type `T` must run before subsystems of type `U`. Default `false`. If you add methods to this function, you must not use a parallel scheduler.
"""
must_run_before(::Type{T}, ::Type{U}) where {T, U} = false

function continuous_event_condition end
function apply_continuous_event! end
function discrete_event_condition end
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
combine(x::NamedTuple, y::NamedTuple) = typeof(x)(combine.(Tuple(x), Tuple(y)))


"""
    isstochastic(::Type{T})

Defaults to `false`, but for subsystems which define a stochastic differential equation, you must add a method to this function of the form

    GraphDynamics.isstochastic(::Type{Mytype}) = true
"""
isstochastic(::Subsystem{T}) where {T} = isstochastic(T)
isstochastic(::T) where {T} = isstochastic(T)
isstochastic(::Type{T}) where {T} = false


for s ∈ [:continuous, :discrete]
    has_events = Symbol(:has_, s, :_events)
    events_require_inputs = Symbol(s, :_events_require_inputs)
    @eval begin
        $has_events(::Subsystem{T}) where {T} = $has_events(T)
        $has_events(::Type{<:Subsystem{T}}) where {T} = $has_events(T)
        $has_events(::Type{<:SubsystemStates{T}}) where {T} = $has_events(T)
        $has_events(::Type{T}) where {T} = false

        $events_require_inputs(::Subsystem{T}) where {T} = $events_require_inputs(T)
        $events_require_inputs(::Type{<:Subsystem{T}}) where {T} = $events_require_inputs(T)
        $events_require_inputs(::Type{<:SubsystemStates{T}}) where {T} = $events_require_inputs(T)
        $events_require_inputs(::Type{T}) where {T} = false
    end
end

"""
    event_times(::T) = ()

add methods to this function if a subsystem or connection type has a discrete event that triggers at pre-defined times. This will be used to add `tstops` to the `ODEProblem` or `SDEProblem` automatically.
"""
event_times(::Any) = ()

abstract type ConnectionRule end
struct NotConnected <: ConnectionRule end
(::NotConnected)(l, r) = zero(promote_type(eltype(l), eltype(r)))
struct ConnectionMatrix{N, CR, Tup <: NTuple{N, NTuple{N, Union{NotConnected, AbstractMatrix{CR}}}}}
    data::Tup
end
struct ConnectionMatrices{NConn, Tup <: NTuple{NConn, ConnectionMatrix}}
    matrices::Tup
end
Base.getindex(m::ConnectionMatrix, i, j) = m.data[i][j]
Base.getindex(m::ConnectionMatrices, i) = m.matrices[i]
Base.length(m::ConnectionMatrices) = length(m.matrices)
Base.size(m::ConnectionMatrix{N}) where {N} = (N, N)

abstract type GraphSystem end

@kwdef struct ODEGraphSystem{CM <: ConnectionMatrices, S, P, EVT, CDEP, CCEP, Ns, SNM, PNM} <: GraphSystem
    connection_matrices::CM
    states_partitioned::S
    params_partitioned::P
    tstops::EVT = Float64[]
    composite_discrete_events_partitioned::CDEP = nothing
    composite_continuous_events_partitioned::CCEP = nothing
    names_partitioned::Ns
    state_namemap::SNM = compute_namemap(names_partitioned, states_partitioned)
    param_namemap::PNM = compute_namemap(names_partitioned, params_partitioned)
end
@kwdef struct SDEGraphSystem{CM <: ConnectionMatrices, S, P, EVT, CDEP, CCEP, Ns, SNM, PNM} <: GraphSystem
    connection_matrices::CM
    states_partitioned::S
    params_partitioned::P
    tstops::EVT = Float64[]
    composite_discrete_events_partitioned::CDEP = nothing
    composite_continuous_events_partitioned::CCEP = nothing
    names_partitioned::Ns
    state_namemap::SNM = compute_namemap(names_partitioned, states_partitioned)
    param_namemap::PNM = compute_namemap(names_partitioned, params_partitioned)
end


#----------------------------------------------------------
# Infrastructure for subsystems
include("subsystems.jl")



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

end # module GraphSystems
