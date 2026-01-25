struct GraphSystemConnection
    src
    dst
    data::NamedTuple
end

function Base.show(io::IO, conn::GraphSystemConnection)
    printstyled("GraphSystemConnection", bold = true)
    println("\nsrc: $(get_name(conn.src))\ndst: $(get_name(conn.dst))\ndata: $(conn.data)")
end

struct GraphSystem
    name::Union{Nothing, Symbol}
    data::OrderedDict{Any, OrderedDict{Any, Vector{GraphSystemConnection}}}
    flat_graph::PartitioningGraphSystem
end

is_flat(g::GraphSystem) = isnothing(g.flat_graph)

function GraphSystem(name, data)
    flat_graph = PartitioningGraphSystem(Symbol(name, :_flat))
    g = GraphSystem(name, data, flat_graph)
    for n in nodes(g)
        system_wiring_rule!(flat_graph, n)
    end
    for (;src, dst, data) in connections(g)
        system_wiring_rule!(flat_graph, src, dst; data...)
    end
    g
end

function Base.copy(g::GraphSystem)
    GraphSystem(g.name, copy(g.data), copy(g.flat_graph))
end

GraphSystem(; name=nothing) = GraphSystem(name, OrderedDict{Any, OrderedDict{Any, GraphSystemConnection}}())

GraphSystemConnection(src, dst; kwargs...) = GraphSystemConnection(src, dst, NamedTuple(kwargs))

function Base.show(io::IO, sys::GraphSystem)
    n_nodes = count(Returns(true), nodes(sys)) 
    n_conns = count(Returns(true), connections(sys))
    print(io, GraphSystem, "(...$n_nodes nodes and $n_conns connections...)")
end

connections(g::GraphSystem) = Iterators.flatmap(g.data) do (_, destinations)
    Iterators.flatmap(destinations) do (_, edges)
        edges
    end
end
nodes(g::GraphSystem) = keys(g.data)
function add_node!(g::GraphSystem, blox)
    get!(g.data, blox) do
        system_wiring_rule!(g.flat_graph, blox)
        OrderedDict{Any, GraphSystemConnection}()
    end
end

function connections(g::GraphSystem, src, dst)
    g.data[src][dst]
end

function connections(g::GraphSystem, src)
    Iterators.flatmap(g.data[src]) do (_, edges)
        edges
    end
end

function add_connection!(g::GraphSystem, src, dst; kwargs...)
    d_src = add_node!(g, src)
    d_dst = add_node!(g, dst)

    v = get!(d_src, dst, GraphSystemConnection[])
    push!(v, GraphSystemConnection(src, dst, NamedTuple(kwargs)))
    system_wiring_rule!(g.flat_graph, src, dst; kwargs...)
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
    g3 = GraphSystem(;name=g1.name)
    merge!(g3, g1)
    merge!(g3, g2)
    g3
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

# Should not give different results on consecutive re-flattenings.
flatten_graph(g::GraphSystem; name=g.name) = g.flat_graph
