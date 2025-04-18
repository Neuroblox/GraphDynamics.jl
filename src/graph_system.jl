struct GraphSystemConnection
    src
    dst
    data::NamedTuple
end
struct GraphSystem
    data::OrderedDict{Any, OrderedDict{Any, GraphSystemConnection}}
end
GraphSystem() = GraphSystem(OrderedDict{Any, OrderedDict{Any, GraphSystemConnection}}())
GraphSystemConnection(src, dst; kwargs...) = GraphSystemConnection(src, dst, NamedTuple(kwargs))

connections(g::GraphSystem) = (Iterators.flatten ∘ Iterators.map)(g.data) do (_, destinations)
    Iterators.map(destinations) do (_, edge)
        edge
    end
end
nodes(g::GraphSystem) = keys(g.data)
function add_node!(g::GraphSystem, blox)
    get!(g.data, blox) do
        OrderedDict{Any, GraphSystemConnection}()
    end
end

function props(g::GraphSystem, src, dst)
    g.data[src][dst]
end

function connect!(g::GraphSystem, src, dst; kwargs...)
    d_src = add_node!(g, src)
    d_dst = add_node!(g, dst)

    if haskey(d_src, dst)
        double_connection_error(src, dst)
    end
    d_src[dst] = GraphSystemConnection(src, dst, NamedTuple(kwargs))
end
@noinline double_connection_error(src, dst) = error(
    """
    Attempted to add a connection between two nodes that already have a connection. This is not currently allowed. The source node was
    $src
    and the destination node was
    $dst
    """
)

connect!(g::GraphSystem, src, dst, d::AbstractDict) = connect!(g, src, dst; d...)
connect!(g::GraphSystem, src, dst, nt::NamedTuple) = connect!(g, src, dst; nt...)
function connect!(g::GraphSystem, (src, dst)::Pair; kwargs...)
    connect!(g, src, dst; kwargs...)
end

has_connection(g::GraphSystem, src, dst) = haskey(g.data, src) && haskey(g.data[src], dst)

function Base.merge!(g1::GraphSystem, g2::GraphSystem)
    for x ∈ nodes(g2)
        add_node!(g1, x)
    end
    for (;src, dst, data) ∈ connections(g2)
        connect!(g1, src, dst; data...)
    end
    g1
end

function system_wiring_rule!(g, node)
    add_node!(g, node)
end
function system_wiring_rule!(g, src, dst; conn, kwargs...)
    connect!(g, src, dst; conn, kwargs...)
end



@kwdef struct PartitionedGraphSystem{CM <: ConnectionMatrices, S, P, EVT, Ns, EP, SNM, PNM, CNM}
    is_stochastic::Bool
    graph::GraphSystem = GraphSystem()
    flat_graph::GraphSystem = GraphSystem()
    connection_matrices::CM
    states_partitioned::S
    params_partitioned::P
    tstops::EVT = Float64[]
    names_partitioned::Ns
    extra_params::EP = (;)
    state_namemap::SNM = make_state_namemap(names_partitioned, states_partitioned)
    param_namemap::PNM = make_param_namemap(names_partitioned, params_partitioned)
    compu_namemap::CNM = make_compu_namemap(names_partitioned, states_partitioned, params_partitioned)
end

function to_subsystem end
to_subsystem(sys::Subsystem) = sys
to_subsystem(::T) where {T} = error("Objects of type $T do not appear to be supported by GraphDynamics. This object must have custom `to_subsystem` method, `subsystem_differential`, and `initialize_inputs` methods.")

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
    names_partitioned  = (Tuple ∘ map)(v -> map(x -> x.name, v), nodes_partitioned)

    
    #==================================================================================================
    Create a ConnectionMatrices object containing structured information about how each lowest level nodes 
    is connected to other nodes, partitioned by the types of the nodes, and the types of the connections for
    type stability.
    e.g. if we have
    
    @named n1 = SysType1(x=1, y=2)
	@named n2 = SysType1(x=1, y=3)
	@named n3 = SysType2(a=1, b=2, c=3)
    
    connect!(g, n1, n2; conn=C1(1))
    connect!(g, n2, n3; conn=C1(2))
    connect!(g, n3, n1; conn=C2(3))
    connect!(g, n3, n2; conn=C3(4))
    
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
    (;connection_matrices, connection_tstops) = make_connection_matrices(g_flat, nodes_partitioned)

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

function make_connection_matrices(g_flat, nodes_partitioned=make_partitioned_nodes(g_flat); pred=(_) -> true, conn_key=:conn)
    connection_types = (imap)(connections(g_flat)) do (; src, dst, data)
        if haskey(data, conn_key) && pred(data[conn_key])
            typeof(data[conn_key])
        else
            nothing
        end
    end |> unique |> x -> filter(!isnothing, x)
    connection_tstops = Float64[]
    connection_matrices = (ConnectionMatrices ∘ Tuple ∘ map)(enumerate(connection_types)) do (nc, CT)
        (ConnectionMatrix ∘ Tuple ∘ map)(enumerate(nodes_partitioned)) do (k, nodeks)
            (Tuple ∘ map)(enumerate(nodes_partitioned)) do (i, nodeis)
                ls = Int[]
                js = Int[]
                conns = CT[]
                for (j, nodeij) ∈ enumerate(nodeis)
                    for (l, nodekl) ∈ enumerate(nodeks)
                        if has_connection(g_flat, nodekl, nodeij)
                            (; data) = props(g_flat, nodekl, nodeij)
                            if haskey(data, conn_key)
                                conn = data[conn_key]
                                if conn isa CT && pred(conn)
                                    push!(js, j)
                                    push!(ls, l)
                                    push!(conns, conn)
                                    
                                    for t ∈ event_times(conn)
                                        push!(connection_tstops, t)
                                    end
                                end
                            end
                        end
                    end
                end
                rule_matrix = if isempty(conns)
                    NotConnected()#{CT}(length(nodeks), length(nodeis))
                else
                    sparse(ls, js, conns, length(nodeks), length(nodeis))
                end
                rule_matrix
            end
        end
    end
    (; connection_matrices, connection_tstops)
end
