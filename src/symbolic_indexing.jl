struct GraphSystemFunction{F, PGS <: PartitionedGraphSystem} <: Function
    f::F
    sys::PGS
end
(f::GraphSystemFunction{F})(args...; kwargs...) where {F} = f.f(args...; kwargs...)

struct StateIndex
    idx::Int
end
struct ParamIndex #todo: this'll require some generalization to support weight params
    tup_index::Int
    v_index::Int
    prop::Symbol
end
struct CompuIndex #todo: this'll require some generalization to support weight params
    tup_index::Int
    v_index::Int
    prop::Symbol
    requires_inputs::Bool
end

function make_state_namemap(names_partitioned::NTuple{N, Vector{Symbol}},
                            states_partitioned::NTuple{N, AbstractVector{<:SubsystemStates}}) where {N}
    namemap = OrderedDict{Symbol, StateIndex}()
    idx = 1
    for i ∈ eachindex(names_partitioned, states_partitioned)
        for j ∈ eachindex(names_partitioned[i], states_partitioned[i])
            states = states_partitioned[i][j]
            for (k, name) ∈ enumerate(propertynames(states))
                propname = Symbol(names_partitioned[i][j], "₊", name)
                namemap[propname] = StateIndex(idx)#StateIndex(i, j, k)
                idx += 1
            end
        end
    end
    namemap
end

function make_param_namemap(names_partitioned::NTuple{N, Vector{Symbol}},
                            params_partitioned::NTuple{N, AbstractVector{<:SubsystemParams}}) where {N}
    namemap = OrderedDict{Symbol, ParamIndex}()
    for i ∈ eachindex(names_partitioned, params_partitioned)
        for j ∈ eachindex(names_partitioned[i], params_partitioned[i])
            params = params_partitioned[i][j]
            for name ∈ propertynames(params)
                propname = Symbol(names_partitioned[i][j], "₊", name)
                #TODO: this'll require some generalization to support weight params
                namemap[propname] = ParamIndex(i, j, name)
            end
        end
    end
    namemap
end


function make_compu_namemap(names_partitioned::NTuple{N, Vector{Symbol}},
                          states_partitioned::NTuple{N, AbstractVector{<:SubsystemStates}},
                          params_partitioned::NTuple{N, AbstractVector{<:SubsystemParams}}) where {N}
    namemap = OrderedDict{Symbol, CompuIndex}()
    for i ∈ eachindex(names_partitioned, states_partitioned, params_partitioned)
        for j ∈ eachindex(names_partitioned[i], states_partitioned[i], params_partitioned[i])
            states = states_partitioned[i][j]
            params = params_partitioned[i][j]
            sys = Subsystem(states, params)
            for name ∈ keys(computed_properties(sys))
                requires_inputs = false
                propname = Symbol(names_partitioned[i][j], "₊", name)
                namemap[propname] = CompuIndex(i, j, name, requires_inputs)
            end
            for name ∈ keys(computed_properties_with_inputs(sys))
                requires_inputs = true
                propname = Symbol(names_partitioned[i][j], "₊", name)
                namemap[propname] = CompuIndex(i, j, name, requires_inputs)
            end
        end
    end
    namemap
end


function Base.getindex(u::AbstractArray, idx::StateIndex)
    u[idx.idx]
end
function Base.setindex!(u::AbstractArray, val, idx::StateIndex)
    setindex!(u, val, idx.idx)
end

function Base.getindex(u::GraphSystemParameters, p::ParamIndex)
    u.params_partitioned[p]
end
function Base.getindex(u::Tuple, (;tup_index, v_index, prop)::ParamIndex)
    getproperty(u[tup_index][v_index], prop)
end


function Base.setindex!(u::GraphSystemParameters, val, p::ParamIndex)
    setindex!(u.params_partitioned, val, p)
end
function Base.setindex!(u::Tuple, val, (;tup_index, v_index, prop)::ParamIndex)
    params = u[tup_index][v_index]
    @reset params[prop] = val
    setindex!(u[tup_index], params, v_index)
end



function SymbolicIndexingInterface.is_variable(g::PartitionedGraphSystem, sym)
    haskey(g.state_namemap, sym)
end
function SymbolicIndexingInterface.variable_index(f::PartitionedGraphSystem, sym)
    get(f.state_namemap, sym, nothing)
end
function SymbolicIndexingInterface.variable_symbols(g::PartitionedGraphSystem)
    collect(keys(g.state_namemap))
end


function SymbolicIndexingInterface.is_parameter(g::PartitionedGraphSystem, sym)
    haskey(g.param_namemap, sym)
end
function SymbolicIndexingInterface.parameter_index(g::PartitionedGraphSystem, sym)
    g.param_namemap[sym]
end

function SymbolicIndexingInterface.parameter_values(p::GraphSystemParameters)
    p.params_partitioned
end
function SymbolicIndexingInterface.parameter_values(p::GraphSystemParameters, i::ParamIndex)
    p.params_partitioned[i]
end

function SymbolicIndexingInterface.parameter_symbols(g::PartitionedGraphSystem)
    collect(keys(g.param_namemap))
end

function SymbolicIndexingInterface.is_independent_variable(sys::PartitionedGraphSystem, sym)
    sym === :t
end


function SymbolicIndexingInterface.independent_variable_symbols(sys::PartitionedGraphSystem)
    (:t,)
end

function SymbolicIndexingInterface.is_time_dependent(sys::PartitionedGraphSystem)
    true
end

function SymbolicIndexingInterface.observed(f::ODEFunction{a, b, F}, sym::Symbol) where {a, b, F<:GraphSystemFunction}
    observed(f.f.sys, sym)
end
function SymbolicIndexingInterface.observed(f::SDEFunction{a, b, F}, sym::Symbol) where {a, b, F<:GraphSystemFunction}
    observed(f.f.sys, sym)
end

function SymbolicIndexingInterface.is_observed(sys::PartitionedGraphSystem, sym)
    haskey(sys.compu_namemap, sym)
end

function SymbolicIndexingInterface.observed(sys::PartitionedGraphSystem, sym)
    (; tup_index, v_index, prop, requires_inputs) = sys.compu_namemap[sym]
    
    # lift these to the type domain so that we specialize on them in the returned closures
    val_tup_index = Val(tup_index)
    val_prop = Val(prop)
    
    if requires_inputs
        function (u, p, t)
            (; params_partitioned, partition_plan, connection_matrices) = p
            states_partitioned = partitioned(u, partition_plan)
            i = valueof(val_tup_index)
            subsys = Subsystem(states_partitioned[i][v_index], params_partitioned[i][v_index])
            input = calculate_inputs(val_tup_index, v_index, states_partitioned, params_partitioned, connection_matrices, t)

            comp_props = computed_properties_with_inputs(subsys)
            comp_props[prop](subsys, input)
        end
    else
        function (u, p, t)
            (; params_partitioned, partition_plan) = p
            states_partitioned = partitioned(u, partition_plan)
            i = valueof(val_tup_index)
            subsys = Subsystem(states_partitioned[i][v_index], params_partitioned[i][v_index])
            getproperty(subsys, prop)
        end
    end
end

# function SymbolicIndexingInterface.all_solvable_symbols(sys::PartitionedGraphSystem)
#     vcat(
#         collect(keys(sys.state_namemap)),
#         collect(keys(sys.observed_namemap)),
#     )
# end

# function SymbolicIndexingInterface.all_symbols(sys::PartitionedGraphSystem)
#     vcat(
#         all_solvable_symbols(sys),
#         collect(keys(sys.param_namemap)),
#         :t
#     )
# end
