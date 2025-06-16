struct GraphSystemConnection
    src
    dst
    data::NamedTuple
end
struct GraphSystem
    data::OrderedDict{Any, OrderedDict{Any, Vector{GraphSystemConnection}}}
end
GraphSystem() = GraphSystem(OrderedDict{Any, OrderedDict{Any, GraphSystemConnection}}())
GraphSystemConnection(src, dst; kwargs...) = GraphSystemConnection(src, dst, NamedTuple(kwargs))

connections(g::GraphSystem) = Iterators.flatmap(g.data) do (_, destinations)
    Iterators.flatmap(destinations) do (_, edges)
        edges
    end
end
nodes(g::GraphSystem) = keys(g.data)
function add_node!(g::GraphSystem, blox)
    get!(g.data, blox) do
        OrderedDict{Any, GraphSystemConnection}()
    end
end

function connections(g::GraphSystem, src, dst)
    g.data[src][dst]
end

function add_connection!(g::GraphSystem, src, dst; kwargs...)
    d_src = add_node!(g, src)
    d_dst = add_node!(g, dst)

    v = get!(d_src, dst, GraphSystemConnection[])
    push!(v, GraphSystemConnection(src, dst, NamedTuple(kwargs)))
end

add_connection!(g::GraphSystem, src, dst, d::AbstractDict) = add_connection!(g, src, dst; d...)
add_connection!(g::GraphSystem, src, dst, nt::NamedTuple) = add_connection!(g, src, dst; nt...)
function add_connection!(g::GraphSystem, (src, dst)::Pair; kwargs...)
    add_connection!(g, src, dst; kwargs...)
end

has_connection(g::GraphSystem, src, dst) = haskey(g.data, src) && haskey(g.data[src], dst)

function Base.merge!(g1::GraphSystem, g2::GraphSystem)
    for x ∈ nodes(g2)
        add_node!(g1, x)
    end
    for (;src, dst, data) ∈ connections(g2)
        add_connection!(g1, src, dst; data...)
    end
    g1
end
function Base.merge(g1::GraphSystem, g2::GraphSystem)
    g3 = GraphSystem()
    merge!(g3, g1)
    merge!(g3, g2)
    g3
end

function delete_connection!(g::GraphSystem, conn::GraphSystemConnection)
    v = g.data[conn.src][conn.dst]
    i = findfirst(==(conn), v)
    if isnothing(i)
        @warn "Attempted to remove a connection that doesn't exist"
    end
    deleteat!(v, i)
end

function system_wiring_rule!(g, node)
    add_node!(g, node)
end
function system_wiring_rule!(g, src, dst; kwargs...)
    if !haskey(kwargs, :conn)
        error("conn keyword argument not specified for connection between $src and $dst")
    end
    add_connection!(g, src, dst; conn=kwargs[:conn], kwargs...)
end

@kwdef struct PartitionedGraphSystem{CM <: ConnectionMatrices, S, P, EVT, Ns, CONM, SNM, PNM, CNM, EP}
    is_stochastic::Bool
    graph::GraphSystem = GraphSystem()
    flat_graph::GraphSystem = GraphSystem()
    connection_matrices::CM
    states_partitioned::S
    params_partitioned::P
    tstops::EVT = Float64[]
    names_partitioned::Ns
    connection_namemap::CONM
    state_namemap::SNM = make_state_namemap(names_partitioned, states_partitioned)
    param_namemap::PNM = make_param_namemap(names_partitioned, params_partitioned)
    compu_namemap::CNM = make_compu_namemap(names_partitioned, states_partitioned, params_partitioned)
    extra_params::EP = (;)
end

function PartitionedGraphSystem(g::GraphSystem)
    g_flat = GraphSystem()
    for sys ∈ nodes(g)
        system_wiring_rule!(g_flat, sys)
    end
    for (;src, dst, data) ∈ connections(g)
        system_wiring_rule!(g_flat, src, dst; data...)
    end
    #==================================================================================================
    Create a list of lists of the lowest level nodes in the flattened graph, partitioned by their type
    so different types can be handled efficiently

    e.g. if we have
    @named n1 = SysType1(x=1, y=2)
	@named n2 = SysType1(x=1, y=3)
	@named n3 = SysType2(a=1, b=2, c=3)

    in the graph, then we'd end up with

    nodes_paritioned = [SysType1[n1, n2], SysType1[n3]]
    
    ===================================================================================================#
    
    node_types = (unique ∘ imap)(typeof, nodes(g_flat))
    nodes_partitioned = map(node_types) do T
        if isstochastic(T)
            system_is_stochastic = true
        end
        filter(collect(nodes(g_flat))) do sys
            sys isa T
        end
    end
    tstops = Float64[]
    subsystems_partitioned = (Tuple ∘ map)(nodes_partitioned) do v
        map(v) do node
            sys = to_subsystem(node)
            for t ∈ event_times(sys)
                push!(tstops, t)
            end
            sys
        end
    end
    states_partitioned = (Tuple ∘ map)(v -> map(get_states, v),  subsystems_partitioned)
    params_partitioned = (Tuple ∘ map)(v -> map(get_params, v),  subsystems_partitioned)
    names_partitioned  = (Tuple ∘ map)(v -> map(x -> convert(Symbol, x.name), v), nodes_partitioned)

    #==================================================================================================
    Create a ConnectionMatrices object containing structured information about how each lowest level nodes 
    is connected to other nodes, partitioned by the types of the nodes, and the types of the connections for
    type stability.
    e.g. if we have
    
    @named n1 = SysType1(x=1, y=2)
	@named n2 = SysType1(x=1, y=3)
	@named n3 = SysType2(a=1, b=2, c=3)
    
    add_connection!(g, n1, n2; conn=Conn1(1))
    add_connection!(g, n2, n3; conn=Conn1(2))
    add_connection!(g, n3, n1; conn=Conn2(3))
    add_connection!(g, n3, n2; conn=Conn2(4))
    
    we'd get
    connection_matrix_1 = Conn1[⎡. 1⎤⎡.⎤
	                            ⎣. .⎦⎣2⎦
	                            [. .][.]]
	
	connection_matrix_2 = Conn2[⎡. .⎤⎡.⎤
	                            ⎣. .⎦⎣.⎦
	                            [3 4][.]]
	
	ConnectionMatrices((connection_matrix_1, connection_matrix_2))

    where the sub-matrices are sparse arrays.

    This allows for type-stable calculations involving the subsystems and their connections
    ===================================================================================================#
    (;connection_matrices, connection_tstops, connection_namemap) = make_connection_matrices(g_flat, nodes_partitioned)

    append!(tstops, connection_tstops)
    

    PartitionedGraphSystem(
        ;graph=g,
        flat_graph=g_flat,
        is_stochastic = any(isstochastic, node_types),
        connection_matrices,
        states_partitioned,
        params_partitioned,
        tstops=unique!(tstops),
        names_partitioned,
        connection_namemap
    )
end

function make_partitioned_nodes(g_flat)
    node_types = (unique ∘ imap)(typeof, nodes(g_flat))
    nodes_partitioned = map(node_types) do T
        if isstochastic(T)
            system_is_stochastic = true
        end
        filter(collect(nodes(g_flat))) do sys
            sys isa T
        end
    end
end

function make_connection_matrices(g_flat, nodes_partitioned=make_partitioned_nodes(g_flat);
                                  pred=(_) -> true,
                                  conn_key=:conn)
    check_no_double_connections(g_flat, conn_key)
    connection_types = (imap)(connections(g_flat)) do (; src, dst, data)
        if haskey(data, conn_key) && pred(data[conn_key])
            typeof(data[conn_key])
        else
            nothing
        end
    end |> unique |> x -> filter(!isnothing, x)
    connection_tstops = Float64[]
    connection_namemap = OrderedDict{Symbol, ConnectionIndex}()
    connection_matrices = (ConnectionMatrices ∘ Tuple ∘ map)(enumerate(connection_types)) do (nc, CT)
        (ConnectionMatrix ∘ Tuple ∘ map)(enumerate(nodes_partitioned)) do (k, nodeks)
            (Tuple ∘ map)(enumerate(nodes_partitioned)) do (i, nodeis)
                ls = Int[]
                js = Int[]
                conns = CT[]
                for (j, nodeij) ∈ enumerate(nodeis)
                    for (l, nodekl) ∈ enumerate(nodeks)
                        if has_connection(g_flat, nodekl, nodeij)
                            for (; data) = connections(g_flat, nodekl, nodeij)
                                if haskey(data, conn_key)
                                    conn = data[conn_key]
                                    if conn isa CT && pred(conn)
                                        push!(js, j)
                                        push!(ls, l)
                                        push!(conns, conn)
                                        
                                        for (prop, name) ∈ pairs(connection_property_namemap(conn, get_name(nodekl), get_name(nodeij)))
                                            connection_namemap[name] = ConnectionIndex(nc, k, i, l, j, prop)
                                        end
                                        
                                        for t ∈ event_times(conn)
                                            push!(connection_tstops, t)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                rule_matrix = if isempty(conns)
                    NotConnected{CT}() #{CT}(length(nodeks), length(nodeis))
                else
                    sparse(ls, js, conns, length(nodeks), length(nodeis))
                end
                rule_matrix
            end
        end
    end
    (; connection_matrices, connection_tstops, connection_namemap)
end

function check_no_double_connections(g, conn_key)
    for src ∈ nodes(g)
        for dst ∈ nodes(g)
            if has_connection(g, src, dst)
                ps = connections(g, src, dst)
                conns = [data[conn_key] for (;data) ∈ connections(g, src, dst) if haskey(data, conn_key)]
                if length(unique(typeof, conns)) < length(conns)
                    error("Cannot have multiple connections between the same two nodes of the same type. Got $(conns) between $src and $dst.")
                end
            end
        end
    end
end
