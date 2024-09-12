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
# Infrastructure for subsystems
include("subsystems.jl")


#----------------------------------------------------------
# API functions to be implemented by new Systems
function subsystem_differential end
function apply_subsystem_differential!(vstate, subsystem, jcn, t)
    vstate[] = subsystem_differential(subsystem, jcn, t)
end
subsystem_differential_requires_inputs(::Type{T}) where {T} = true
subsystem_differential_requires_inputs(::Subsystem{T}) where {T} = subsystem_differential_requires_inputs(T)

function apply_subsystem_noise!(vstate, subsystem, t)
    nothing
end

must_run_before(::Type{T}, ::Type{U}) where {T, U} = false
    
function continuous_event_condition end
function apply_continuous_event! end
function discrete_event_condition end
function apply_discrete_event! end

function initialize_input end
isstochastic(::Subsystem{T}) where {T} = isstochastic(T)
isstochastic(::T) where {T} = isstochastic(T)
isstochastic(::Type{T}) where {T} = false

combine(x::Number, y::Number) = x + y
combine(x::NT, y::NT) where {NT <: NamedTuple} = typeof(x)(combine.(Tuple(x), Tuple(y)))

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

struct ODEGraphSystem{G, CM <: ConnectionMatrices, S, P, SNM, PNM} <: GraphSystem
    graph::G
    connection_matrices::CM
    states_partitioned::S
    params::P
    state_namemap::SNM
    param_namemap::PNM
end
struct SDEGraphSystem{G, CM <: ConnectionMatrices,  S, P, SNM, PNM} <: GraphSystem
    graph::G
    connection_matrices::CM
    states_partitioned::S
    params::P
    state_namemap::SNM
    param_namemap::PNM
end


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
