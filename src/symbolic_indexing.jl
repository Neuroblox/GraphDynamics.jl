function GraphNamemap(names_partitioned::Tuple,
                      states_partitioned::Tuple,
                      params_partitioned::Tuple,
                      connection_matrices::ConnectionMatrices)
    nm = GraphNamemap(OrderedDict{Symbol,StateIndex}(),
                      OrderedDict{Symbol,ParamIndex}(),
                      OrderedDict{Symbol,CompuIndex}(),
                      OrderedDict{Symbol, ConnectionIndex}())
    
    populate_property_namemaps!(nm, names_partitioned, states_partitioned, params_partitioned)
    populate_connection_namemap!(nm.connection_namemap, names_partitioned, connection_matrices)
    nm
end

@generated function populate_property_namemaps!(
    namemaps,
    names_partitioned::NTuple{N, Vector{Symbol}},
    states_partitioned::NTuple{N, AbstractVector{<:SubsystemStates}},
    params_partitioned::NTuple{N, AbstractVector{<:SubsystemParams}}) where {N}
    quote
        sidx = 1
        @nexprs $N i -> begin
            for j ∈ eachindex(names_partitioned[i], states_partitioned[i], params_partitioned[i])
                states = states_partitioned[i][j]
                params = params_partitioned[i][j]
                for name ∈ propertynames(states)
                    propname = Symbol(names_partitioned[i][j], "₊", name)
                    namemaps.state_namemap[propname] = StateIndex(sidx)
                    sidx += 1
                end
                for name ∈ propertynames(params)
                    propname = Symbol(names_partitioned[i][j], "₊", name)
                    namemaps.param_namemap[propname] = ParamIndex(i, j, name)
                end
                sys = Subsystem(states, params)
                tag = get_tag(sys)
                for name ∈ keys(computed_properties(tag))
                    requires_inputs = false
                    propname = Symbol(names_partitioned[i][j], "₊", name)
                    namemaps.compu_namemap[propname] = CompuIndex(i, j, name, requires_inputs)
                end
                for name ∈ keys(computed_properties_with_inputs(tag))
                    requires_inputs = true
                    propname = Symbol(names_partitioned[i][j], "₊", name)
                    namemaps.compu_namemap[propname] = CompuIndex(i, j, name, requires_inputs)
                end
            end
        end
    end
end

@generated function populate_connection_namemap!(
    namemap,
    names_partitioned::NTuple{Len, Any},
    connection_matrices::ConnectionMatrices{NConn}) where {Len, NConn}
    quote
        @nexprs $Len k -> begin
            @nexprs $Len i -> begin
                @nexprs $NConn nc -> begin
                    M = connection_matrices[nc].data[k][i]
                    if !(M isa NotConnected)
                        for j ∈ eachindex(names_partitioned[i])
                            for (l, conn) ∈ maybe_sparse_enumerate_col(M, j)
                                name_kl = names_partitioned[k][l]
                                name_ij = names_partitioned[i][j]
                                for (prop, name) ∈ pairs(connection_property_namemap(conn, name_kl, name_ij))
                                    namemap[name] = ConnectionIndex(nc, k, i, l, j, name, prop)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
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
    if isnothing(prop)
        conn
    else
        getproperty(conn, prop)
    end
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
    M = u[nc].data[i_src][i_dst]
    if isnothing(prop)
        M[j_src, j_dst] = val
    else
        M[j_src, j_dst] = setproperties(M[j_src, j_dst], NamedTuple{(prop,)}(val))
    end
end


function SymbolicIndexingInterface.is_variable(g::GraphNamemap, sym)
    haskey(g.state_namemap, sym)
end
function SymbolicIndexingInterface.variable_index(f::GraphNamemap, sym)
    get(f.state_namemap, sym, nothing)
end
function SymbolicIndexingInterface.variable_symbols(g::GraphNamemap)
    collect(keys(g.state_namemap))
end


function SymbolicIndexingInterface.is_parameter(g::GraphNamemap, sym)
    haskey(g.param_namemap, sym) || haskey(g.connection_namemap, sym)
end

function SymbolicIndexingInterface.parameter_index(g::GraphNamemap, sym)
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
    val
end

function SymbolicIndexingInterface.set_parameter!(buffer::GraphSystemParameters, value, conn_index::ConnectionIndex)    
    (;connection_matrices) = buffer
    (; nc, i_src, i_dst, j_src, j_dst, connection_key, prop) = conn_index
    conn_old = connection_matrices[nc][i_src, i_dst][j_src, j_dst]
    conn_new = setproperties(conn_old, NamedTuple{(prop,)}(value))
    connection_matrices[nc][i_src, i_dst][j_src, j_dst] = conn_new
    value
end

function SymbolicIndexingInterface.parameter_index(p::GraphSystemParameters, sym)
    parameter_index(p.symbolic_indexing_namemap, sym)
end

function SymbolicIndexingInterface.is_parameter(p::GraphSystemParameters, sym)
    is_parameter(p.symbolic_indexing_namemap, sym)
end

function SymbolicIndexingInterface.parameter_symbols(g::GraphNamemap)
    collect(Iterators.flatten((keys(g.param_namemap), keys(g.connection_namemap))))
end

function SymbolicIndexingInterface.is_independent_variable(sys::GraphNamemap, sym)
    sym === :t
end


function SymbolicIndexingInterface.independent_variable_symbols(sys::GraphNamemap)
    (:t,)
end

function SymbolicIndexingInterface.is_time_dependent(sys::GraphNamemap)
    true
end

function SymbolicIndexingInterface.observed(f::ODEFunction{a, b, typeof(graph_ode!)}, sym::Symbol) where {a, b}
    observed(f.sys, sym)
end
function SymbolicIndexingInterface.observed(f::SDEFunction{a, b, typeof(graph_ode!)}, sym::Symbol) where {a, b}
    observed(f.sys, sym)
end

function SymbolicIndexingInterface.is_observed(sys::GraphNamemap, sym)
    haskey(sys.compu_namemap, sym)
end

function SymbolicIndexingInterface.observed(sys::GraphNamemap, syms::Union{Vector{Symbol}, Tuple{Vararg{Symbol}}})
    function (u, p, t)
        map(syms) do sym
            observed(sys, sym)(u, p, t)
        end
    end
end

function SymbolicIndexingInterface.observed(sys::GraphNamemap, sym)
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

function SymbolicIndexingInterface.remake_buffer(sys, oldbuffer::GraphSystemParameters, idxs, vals)
    newbuffer = copy(oldbuffer)
    set_params!!(newbuffer, zip(idxs, vals))
end

function set_params!!(buffer::GraphSystemParameters, param_map)
    (; param_namemap, connection_namemap) = buffer.symbolic_indexing_namemap
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
    (;connection_matrices, symbolic_indexing_namemap) = buffer
    (; nc, i_src, i_dst, j_src, j_dst, connection_key, prop) = conn_index
    conn_old = connection_matrices[nc][i_src, i_dst][j_src, j_dst]
    conn_new = setproperties(conn_old, NamedTuple{(prop,)}((value,)))
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
            symbolic_indexing_namemap.connection_namemap[connection_key] = conn_index_new # This is important so we don't lose track of where the parameter moved to!
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
