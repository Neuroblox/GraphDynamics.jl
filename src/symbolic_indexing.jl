struct GraphSystemFunction{F, PGS <: PartitionedGraphSystem} <: Function
    f::F
    sys::PGS
end
(f::GraphSystemFunction{F})(args...; kwargs...) where {F} = f.f(args...; kwargs...)

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
    connection_key::Symbol
    prop::Symbol
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
                namemap[propname] = StateIndex(idx)
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
            tag = get_tag(sys)
            for name ∈ keys(computed_properties(tag))
                requires_inputs = false
                propname = Symbol(names_partitioned[i][j], "₊", name)
                namemap[propname] = CompuIndex(i, j, name, requires_inputs)
            end
            for name ∈ keys(computed_properties_with_inputs(tag))
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

function Base.getindex(u::GraphSystemParameters, i::ConnectionIndex)
    u.connection_matrices[i]
end

function Base.getindex(cm::ConnectionMatrices, (; nc, i_src, i_dst, j_src, j_dst, prop)::ConnectionIndex)
    conn = cm[nc].data[i_src][i_dst][j_src, j_dst]
    getproperty(conn, prop)
end


function Base.setindex!(u::GraphSystemParameters, val, p::ParamIndex)
    setindex!(u.params_partitioned, val, p)
end
function Base.setindex!(u::Tuple, val, (;tup_index, v_index, prop)::ParamIndex)
    params = u[tup_index][v_index]
    @reset params[prop] = val
    setindex!(u[tup_index], params, v_index)
end

function Base.setindex!(u::GraphSystemParameters, val, p::ConnectionIndex)
    setindex!(u.connection_matrices, val, p)
end
function Base.setindex!(u::ConnectionMatrices, val, (; nc, i_src, i_dst, j_src, j_dst, prop)::ConnectionIndex)
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
    haskey(g.param_namemap, sym) || haskey(g.connection_namemap, sym)
end

function SymbolicIndexingInterface.parameter_index(g::PartitionedGraphSystem, sym)
    if haskey(g.param_namemap, sym)
        g.param_namemap[sym]
    else
        g.connection_namemap[sym]
    end
end

function SymbolicIndexingInterface.parameter_values(p::GraphSystemParameters)
    p
end
function SymbolicIndexingInterface.parameter_values(p::GraphSystemParameters, i::ParamIndex)
    p.params_partitioned[i]
end
function SymbolicIndexingInterface.parameter_values(p::GraphSystemParameters, i::ConnectionIndex)
    p.connection_matrices[i]
end

function SymbolicIndexingInterface.set_parameter!(p::GraphSystemParameters, val, idx::ParamIndex)    
    (; tup_index, v_index, prop) = idx
    (; params_partitioned) = p
    params = params_partitioned[tup_index][v_index]
    params_new = set_param_prop(params, prop, val; allow_typechange=false)
    params_partitioned[tup_index][v_index] = params_new
    p
end

function SymbolicIndexingInterface.set_parameter!(buffer::GraphSystemParameters, value, conn_index::ConnectionIndex)    
    (;connection_matrices, connection_namemap) = buffer
    (; nc, i_src, i_dst, j_src, j_dst, connection_key, prop) = conn_index
    conn_old = connection_matrices[nc][i_src, i_dst][j_src, j_dst]
    conn_new = setproperties(conn_old, NamedTuple{(prop,)}(value))
    connection_matrices[nc][i_src, i_dst][j_src, j_dst] = conn_new
    buffer
end


function SymbolicIndexingInterface.parameter_symbols(g::PartitionedGraphSystem)
    collect(Iterators.flatten((keys(g.param_namemap), keys(g.connection_namemap))))
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

function SymbolicIndexingInterface.observed(sys::PartitionedGraphSystem, syms::Union{Vector{Symbol}, Tuple{Vararg{Symbol}}})
    function (u, p, t)
        map(syms) do sym
            observed(sys, sym)(u, p, t)
        end
    end
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
            tag = get_tag(subsys)
            comp_props = computed_properties_with_inputs(tag)
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

function SymbolicIndexingInterface.remake_buffer(sys, oldbuffer::GraphSystemParameters, idxs, vals)
    newbuffer = copy(oldbuffer)
    set_params!!(newbuffer, zip(idxs, vals))
end

function set_params!!(buffer::GraphSystemParameters, param_map)
    (; param_namemap, connection_namemap) = buffer
    for (key, val) ∈ param_map
        if haskey(param_namemap, key)
            buffer = set_param!!(buffer, param_namemap[key], val)
        elseif haskey(connection_namemap, key)
            buffer = set_param!!(buffer, connection_namemap[key], val)
        else
            error("Key $key does not correspond to a known parameter. ")
        end
    end
    buffer
end


# This is a possibly-out-of-place variant of set_parameter! that is meant to be used by `remake` where
# types are allowed to be widened.
function set_param!!(buffer::GraphSystemParameters, (; tup_index, v_index, prop)::ParamIndex, val)
    (; params_partitioned) = buffer
    params = params_partitioned[tup_index][v_index]
    params_new = set_param_prop(params, prop, val; allow_typechange=true)
    peltype = eltype(params_partitioned[tup_index])
    if !(typeof(params_new) <: peltype)
        new_eltype = promote_type(typeof(params_new), peltype)
        @reset params_partitioned[tup_index] = convert.(new_eltype, params_partitioned[tup_index])
        @reset buffer.params_partitioned = params_partitioned
    end
    params_partitioned[tup_index][v_index] = params_new
    buffer
end

function re_eltype_params(params_partitioned)
    map(params_partitioned) do v
        ptype = mapreduce(typeof, promote_type, v)
        if ptype == eltype(v)
            v
        else
            convert.(ptype, v)
        end
    end
end

function set_param!!(buffer::GraphSystemParameters, conn_index::ConnectionIndex, value)
    (;connection_matrices, connection_namemap) = buffer
    (; nc, i_src, i_dst, j_src, j_dst, connection_key, prop) = conn_index
    conn_old = connection_matrices[nc][i_src, i_dst][j_src, j_dst]
    conn_new = setproperties(conn_old, NamedTuple{(prop,)}(value))
    CR_new = typeof(conn_new)
    CR_old = typeof(conn_old)
    if !(CR_new <: CR_old)
        nc_new = findfirst(i -> CR_new <: rule_type(connection_matrices[i]), 1:length(connection_matrices))
        if isnothing(nc_new)
            # This means there's no rules matrix yet of this type, so we'll need to make a whole new one!
            nc_new = length(connection_matrices) + 1
            N = length(connection_matrices.matrices)
            (n, m) = size(connection_matrices[nc][i_src, i_dst])

            CM = (ConnectionMatrix ∘ ntuple)(N) do i_src′
                ntuple(N) do i_dst′
                    if i_src′ == i_src && i_dst′ == i_dst
                        spzeros(CR_new, n, m)
                    else
                        NotConnected{CR_new}()
                    end
                end
            end
            connection_matrices = @insert connection_matrices.matrices[nc_new] = CM
        end # if isnothing(nc_new)
        
        # Update the position in the namemap
        let conn_index_new = @set conn_index.nc = nc_new
            connection_namemap[connection_key] = conn_index_new # This is important so we don't lose track of where the parameter moved to!
        end
        
        # Delete the old element!
        let CM_old = connection_matrices[nc][i_src, i_dst]
            CM_old[j_src, j_dst] = zero(CR_old)
            dropzeros!(CM_old)
        end
        
        # Now set the new connection matrix element in its new position
        connection_matrices[nc_new][i_src, i_dst][j_src, j_dst] = conn_new 
    else
        # In this case we don't have to change the type of anything so it's simple!
        connection_matrices[nc    ][i_src, i_dst][j_src, j_dst] = conn_new 
    end
    @set buffer.connection_matrices = connection_matrices
end
