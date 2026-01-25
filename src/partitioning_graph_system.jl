
"""
    PartitionedIndex{i}(j)

Used for indexing into partitioned structures where the `i` index refers to the outer (typically static)
structure, and the inner index is dynamic. In GraphDynamics.jl, we often replace vectors of objects with
many possible types with tuples where each element of the tuple is a vector of a concrete type. This
allows for type-stable indexing and iteration.
"""
struct PartitionedIndex{i}
    j::Int
end
Base.zero(::Type{PartitionedIndex{i}}) where {i} = PartitionedIndex{i}(0)
Base.zero(::PartitionedIndex{i}) where {i} = PartitionedIndex{i}(0)

@propagate_inbounds Base.getindex(v::Union{AbstractVector, Tuple}, (;j)::PartitionedIndex{i}) where {i} = v[i][j]
@propagate_inbounds function Base.getindex(m::AbstractMatrix,
                                           idx1::PartitionedIndex{k},
                                           idx2::PartitionedIndex{i}) where {i,k}
    l = idx1.j
    j = idx2.j
    m[k,i][l,j]
end
function Base.getproperty(idx::PartitionedIndex{i}, s::Symbol) where {i}
    if s == :i
        convert(Int, i)
    else
        getfield(idx, s)
    end
end
Base.propertynames(idx::PartitionedIndex) = (:i, :j)

struct SparseMatrixBuilder{T}
    data::OrderedDict{Tuple{Int, Int}, @NamedTuple{conn::T, kwargs::NamedTuple}}
end
Base.eltype(::SparseMatrixBuilder{T}) where {T} = T
Base.eltype(::Type{SparseMatrixBuilder{T}}) where {T} = T
SparseMatrixBuilder{T}() where {T} = SparseMatrixBuilder{T}(OrderedDict{Tuple{Int, Int}, @NamedTuple{conn::T, kwargs::NamedTuple}}())
function Base.getindex(m::SparseMatrixBuilder, i::Integer, j::Integer)
    m.data[(Int(i), Int(j))]
end

function Base.zeros(::Type{SparseMatrixBuilder{T}}, sz::Integer...) where {T}
    map(CartesianIndices(sz)) do _
        SparseMatrixBuilder{T}()
    end
end
Base.zero(::Type{SparseMatrixBuilder{T}}) where {T} = SparseMatrixBuilder{T}()

function SparseArrays.sparse(m::SparseMatrixBuilder, N::Integer, M::Integer)
    Ls = (inds[1] for inds ∈ keys(m.data))
    Js = (inds[2] for inds ∈ keys(m.data))
    conns = [conn for (; conn) ∈ values(m.data)]
    sparse(collect(Ls), collect(Js), conns, N, M)
end

function Base.setindex!(M::AbstractMatrix{SparseMatrixBuilder{T}}, val, idx1::PartitionedIndex{k}, idx2::PartitionedIndex{i}) where {T, k, i}
    l = idx1.j
    j = idx2.j
    M[k,i].data[(l,j)] = val
end
function extrude(M::AbstractMatrix{SparseMatrixBuilder{T}}) where {T}
    n, m = size(M)
    @assert n == m
    [M                                   zeros(SparseMatrixBuilder{T}, n, 1)
     zeros(SparseMatrixBuilder{T}, 1, n) zero( SparseMatrixBuilder{T})]
end

mutable struct PartitioningGraphSystem
    const name::Union{Nothing, Symbol}
    const node_namemap::OrderedDict{Symbol, PartitionedIndex}
    
    # One vector of nodes per node-type.
    const nodes_partitioned::Vector{Vector}
    const subsystems_partitioned::Vector{Vector}

    # A vector where each element corresponds to one connection type (BasicConnection, ReverseConnection, etc.)
    # The matrices correspond to each node type, and then the SparseMatrixBuilders correspond to
    # connections within one combination of types
    const connections_partitioned::Vector{Matrix{SparseMatrixBuilder{T}} where {T}}
    const tstops::Vector{Float64}
    is_stochastic::Bool
    const extra_params::OrderedDict{Symbol, Any}
end

function Base.copy(g::PartitioningGraphSystem)
    PartitioningGraphSystem(
        g.name,
        copy(g.node_namemap),
        copy(g.nodes_partitioned),
        copy(g.subsystems_partitioned),
        copy(g.connections_partitioned),
        copy(g.tstops),
        g.is_stochastic,
        copy(g.extra_params)
    )
end

function PartitioningGraphSystem(name=nothing)
    PartitioningGraphSystem(name,
                            OrderedDict{Symbol, PartitionedIndex}(),
                            Vector{<:Any}[],
                            Vector{<:Any}[],
                            Matrix{<:SparseMatrixBuilder}[],
                            Float64[],
                            false,
                            OrderedDict{Symbol, Any}())
end

function PartitionedIndex(g::PartitioningGraphSystem, node)
    g.node_namemap[get_name(node)]
end

function add_node!(g::PartitioningGraphSystem, x::T) where {T}
    name = get_name(x)
    if haskey(g.node_namemap, name)
        node_old = g.nodes_partitioned[g.node_namemap[name]]
        if !isequal(x, node_old)
            error("Tried to add node with name $name to a PartitioningGraphSystem, but the PartitioningGraphSystem already had a node with that name which is not equal to the new value.\n New value: $x\n Old value: $node_old")
        else
            return x
        end
    end
    sys = to_subsystem(x)
    i = findfirst(v -> T <: eltype(v), g.nodes_partitioned)
    if isnothing(i)
        push!(g.nodes_partitioned, T[])
        push!(g.subsystems_partitioned, Subsystem{get_tag(sys)}[])
        for nc ∈ eachindex(g.connections_partitioned)
            g.connections_partitioned[nc] = extrude(g.connections_partitioned[nc])
        end
        i = length(g.nodes_partitioned)
    end
    push!(g.nodes_partitioned[i], x)
    push!(g.subsystems_partitioned[i], sys)
    foreach(t -> push!(g.tstops, t), event_times(sys))
    isstochastic(sys) && (g.is_stochastic = true)
    g.node_namemap[name] = PartitionedIndex{i}(lastindex(g.nodes_partitioned[i]))
    x
end

add_connection!(g::PartitioningGraphSystem, src, dst; conn, kwargs...) = add_connection!(g, src, conn, dst; kwargs...)

function add_connection!(g::PartitioningGraphSystem, src::T, conn::Conn, dst::U; kwargs...) where {T, Conn, U}
    name_src = get_name(src)
    name_dst = get_name(dst)
    idx_src = get!(g.node_namemap, name_src) do
        add_node!(g, src)
        g.node_namemap[name_src]
    end
    idx_dst = get!(g.node_namemap, name_dst) do
        add_node!(g, dst)
        g.node_namemap[name_dst]
    end
    nc = findfirst(g.connections_partitioned) do mat
        Conn <: eltype(eltype(mat))
    end
    if isnothing(nc)
        M = zeros(SparseMatrixBuilder{Conn}, length(g.nodes_partitioned), length(g.nodes_partitioned))
        push!(g.connections_partitioned, M)
        nc = length(g.connections_partitioned)
    end
    for t ∈ event_times(conn, g.subsystems_partitioned[idx_src], g.subsystems_partitioned[idx_dst])
        push!(g.tstops, t)
    end
    builder = g.connections_partitioned[nc][idx_src.i, idx_dst.i]
    if haskey(builder.data, (idx_src.j, idx_dst.j))
        error("Tried to add a connection of type $Conn between $name_src and $name_dst, but a connection of that type already exists.")
    else
        g.connections_partitioned[nc][idx_src, idx_dst] = (; conn, kwargs=NamedTuple(kwargs))
    end
end

function nodes(g::PartitioningGraphSystem)
    Iterators.flatten(g.nodes_partitioned)
end
function connections(g::PartitioningGraphSystem)
    Iterators.flatmap(enumerate(g.connections_partitioned)) do (nc, mat)
        Iterators.flatmap(CartesianIndices(mat)) do Idx
            (k, i) = Tuple(Idx)
            builder = mat[k, i]
            Iterators.map(builder.data) do ((l,j), (; conn, kwargs))
                (; src=g.nodes_partitioned[k][l], dst=g.nodes_partitioned[i][j], conn, kwargs, nc, k, i, l, j)
            end
        end
    end
end
function connections(g::PartitioningGraphSystem, src, dst)
    name_src = get_name(src)
    name_dst = get_name(dst)
    (has_node(g, src) && has_node(g, dst)) || return () # empty iterator
    idx_src = g.node_namemap[name_src]
    idx_dst = g.node_namemap[name_dst]

    k = idx_src.i
    l = idx_src.j
    i = idx_dst.i
    j = idx_dst.j

    src = g.nodes_partitioned[idx_src]
    dst = g.nodes_partitioned[idx_dst]
    itr = Iterators.filter(enumerate(g.partitioned_connections)) do (nc, mat)
        haskey(mat[k,i].data, (l,j))
    end
    Iterators.map(itr) do (nc, mat)
        (; src, dst, mat[idx_src, idx_dst]..., nc, k, i, l, j)
    end
end

function has_node(g::PartitioningGraphSystem, x)
    haskey(g.node_namemap, get_name(x))
end
function has_connection(g::PartitioningGraphSystem, src, dst)
    (has_node(g, src) && has_node(g, dst)) || return false
    name_src = get_name(src)
    name_dst = get_name(dst)
    idx_src = g.node_namemap[name_src]
    idx_dst = g.node_namemap[name_dst]
    any(g.connections_partitioned) do mat
        builder = mat[idx_src.i, idx_dst.i]
        haskey(builder.data, (idx_src.j, idx_dst.j))
    end
end

function Base.merge!(g1::PartitioningGraphSystem, g2::PartitioningGraphSystem)
    for x ∈ nodes(g2)
        #overloadable function that defaults to just adding the node to g1
        merge_node!(g1, g2, x)
    end
    for (;src, dst, conn, kwargs) ∈ connections(g2)
        #overloadable function that defaults to just adding the connection to g1
        merge_connection!(g1, g2, src, conn, dst; kwargs...)
    end
    for (k, v) ∈ g2.extra_params
        if !haskey(g1.extra_params, k)
            g1.extra_params[k] = v
        end
    end
    g1
end
function Base.merge(g1::PartitioningGraphSystem, g2::PartitioningGraphSystem)
    g3 = GraphSystem(;name=g1.name)
    merge!(g3, g1)
    merge!(g3, g2)
    g3
end

"""
    merge_node!(g1, g2, x)

Default: `add_node!(g1, x)`. This function is called during `merge!(g1, g2)` to add the nodes from `g2`
into `g1`. Overload it for nodes `x` which may require custom handling.
"""
merge_node!(g1, g2, x) = add_node!(g1, x)

"""
    merge_connection!(g1, g2, src, conn, dst; kwargs...)

Default: `add_connection!(g1, g2, src, conn, dst; kwargs...)`. This function is called during `merge!(g1, g2)`
to add the connections from `g2` into `g1`. Overload it for connections which may require custom handling.
"""
merge_connection!(g1, g2, src, conn, dst; kwargs...) = add_connection!(g1, src, conn, dst; kwargs...)
