struct GraphSystemFunction{F, GS <: GraphSystem} <: Function
    f::F
    sys::GS
end
(f::GraphSystemFunction{F})(args...; kwargs...) where {F} = f.f(args...; kwargs...)

struct StateIndex
    tup_index::Int
    v_index::Int
    state_index::Int
end
struct ParamIndex #todo: this'll require some generalization to support weight params
    tup_index::Int
    v_index::Int
    prop::Symbol
end

function Base.getindex(u::ArrayPartition, idx::StateIndex)
    u.x[idx]
end
function Base.getindex(u::Tuple, (;tup_index, v_index, state_index)::StateIndex)
    u[tup_index][state_index, v_index]
end


function Base.getindex(u::Tuple, (;tup_index, v_index, prop)::ParamIndex)
    getproperty(u[tup_index][v_index], prop)
end
function Base.getindex(u::GraphSystemParameters, p::ParamIndex)
    u.subsystem_params[p]
end


function SymbolicIndexingInterface.is_variable(g::GraphSystem, sym)
    haskey(g.state_namemap, sym)
end
function SymbolicIndexingInterface.variable_index(f::GraphSystem, sym)
    get(f.state_namemap, sym, nothing)
end
function SymbolicIndexingInterface.variable_symbols(g::GraphSystem)
    collect(keys(g.state_namemap))
end


function SymbolicIndexingInterface.is_parameter(g::GraphSystem, sym)
    haskey(g.param_namemap, sym)
end
function SymbolicIndexingInterface.parameter_index(g::GraphSystem, sym)
    g.param_namemap[sym]
end

function SymbolicIndexingInterface.parameter_values(p::GraphSystemParameters, i::ParamIndex)
    p.subsystem_params[i]
end

function SymbolicIndexingInterface.parameter_symbols(g::GraphSystem)
    collect(keys(f.param_namemap))
end

function SymbolicIndexingInterface.is_independent_variable(sys::GraphSystem, sym)
    sym === :t
end


function SymbolicIndexingInterface.independent_variable_symbols(sys::GraphSystem)
    :t
end

function SymbolicIndexingInterface.is_time_dependent(sys::GraphSystem)
    true
end

function SymbolicIndexingInterface.is_observed(sys::GraphSystem, sym)
    false # TODO: support observed variables
end

# function SymbolicIndexingInterface.all_solvable_symbols(sys::GraphSystem)
#     vcat(
#         collect(keys(sys.state_namemap)),
#         collect(keys(sys.observed_namemap)),
#     )
# end

# function SymbolicIndexingInterface.all_symbols(sys::GraphSystem)
#     vcat(
#         all_solvable_symbols(sys),
#         collect(keys(sys.param_namemap)),
#         :t
#     )
# end
