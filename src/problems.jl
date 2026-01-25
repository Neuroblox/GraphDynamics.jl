# Fallback for failed problem construction
function (::Type{T})(g::GraphSystem, args...; kwargs...) where {T <: SciMLBase.AbstractSciMLProblem}
    throw(ArgumentError("GraphDynamics.jl does not yet support the use of GraphSystem in $T."))
end

function SciMLBase.ODEProblem(g::GraphSystem, u0map, tspan, param_map=[];
                    scheduler=SerialScheduler(), tstops=Float64[],
                    allow_nonconcrete=false, global_events=(), kwargs...)
    p = GraphSystemParameters(g; scheduler, u0map, param_map)
    (; symbolic_indexing_namemap, states_partitioned) = p
    u0 = make_u0(p; allow_nonconcrete)
    callback = make_callback(p; global_events)
    f = ODEFunction{true, SciMLBase.FullSpecialize}(graph_ode!, sys=symbolic_indexing_namemap)
    if g.flat_graph.is_stochastic
        error("Passed a stochastic GraphSystem to ODEProblem. You probably meant to use SDEProblem")
    end
    tstops = vcat(tstops, g.flat_graph.tstops)
    dtmax = make_dtmax(p)
    prob = ODEProblem(f, u0, tspan, p; callback, tstops, dtmax, kwargs...)
    for (k, v) ∈ u0map
        setu(prob, k)(prob, v)
    end
    @reset prob.p = set_params!!(prob.p, param_map)
    prob
end

function SciMLBase.SDEProblem(g::GraphSystem, u0map, tspan, param_map=[];
                              scheduler=SerialScheduler(), tstops=Float64[],
                              allow_nonconcrete=false, global_events=(), kwargs...)
    p = GraphSystemParameters(g; scheduler, u0map, param_map)
    (; symbolic_indexing_namemap, states_partitioned) = p
    u0 = make_u0(p; allow_nonconcrete)
    callback = make_callback(p; global_events)
    f = ODEFunction{true, SciMLBase.FullSpecialize}(graph_ode!, sys=symbolic_indexing_namemap)
    if !g.flat_graph.is_stochastic
        error("Passed a non-stochastic GraphSystem to SDEProblem. You probably meant to use ODEProblem")
    end
    noise_rate_prototype = nothing # this'll need to change once we support correlated noise
    dtmax = make_dtmax(p)
    tstops = vcat(tstops, g.flat_graph.tstops)
    prob = SDEProblem(f, graph_noise!, u0, tspan, p; callback, noise_rate_prototype, dtmax, kwargs...)
    for (k, v) ∈ u0map
        setu(prob, k)(prob, v)
    end
    @reset prob.p = set_params!!(prob.p, param_map)
    prob
end

Base.@kwdef struct GraphSystemParameters{PP, SP, CM, S, PAP, DEC, NP, EP<:NamedTuple}
    graph::Union{Nothing, GraphSystem}
    states_partitioned::SP
    params_partitioned::PP
    connection_matrices::CM
    scheduler::S
    partition_plan::PAP
    discrete_event_cache::DEC
    names_partitioned::NP
    symbolic_indexing_namemap::GraphNamemap
    extra_params::EP=(;)
end

function Base.copy(p::GraphSystemParameters)
    GraphSystemParameters(
        copy(p.graph),
        copy.(p.states_partitioned),
        copy.(p.params_partitioned),
        copy(p.connection_matrices),
        p.scheduler,
        (p.partition_plan),
        copy.(p.discrete_event_cache),
        copy.(p.names_partitioned),
        copy(p.symbolic_indexing_namemap),
        map(copy, p.extra_params)
    )
end

function DiffEqBase.anyeltypedual(p::GraphSystemParameters, ::Type{Val{counter}}) where {counter}
    anyeltypedual((p.params_partitioned, p.connection_matrices))
end
function DiffEqBase.anyeltypedual(p::ConnectionMatrices, ::Type{Val{counter}}) where {counter}
    anyeltypedual(p.matrices)
end
function DiffEqBase.anyeltypedual(p::ConnectionMatrix, ::Type{Val{counter}}) where {counter}
    anyeltypedual(p.data)
end

function GraphSystemParameters(g::GraphSystem; scheduler=SerialScheduler(), u0map=[], param_map=[])
    (; tstops) = g.flat_graph

    tupmap = Tuple ∘ map
    subsystems_partitioned = Tuple(g.flat_graph.subsystems_partitioned)
    states_partitioned = map(v -> get_states.(v), subsystems_partitioned)
    names_partitioned  = tupmap(g.flat_graph.nodes_partitioned) do v
        get_name.(v)
    end
    connection_matrices = tupmap(enumerate(g.flat_graph.connections_partitioned)) do (nc, mat)
        tupmap(axes(mat, 1)) do k
            tupmap(axes(mat, 2)) do i
                builder = mat[k,i]
                if isempty(builder.data)
                    NotConnected{eltype(builder)}()
                else
                    sparse(builder,
                           length(subsystems_partitioned[k]),
                           length(subsystems_partitioned[i]))
                end
            end
        end |> ConnectionMatrix
    end |> ConnectionMatrices
    extra_params = NamedTuple(g.flat_graph.extra_params)
    
    params_partitioned = map(subsystems_partitioned) do v
        pv = get_params.(v)
        if !isconcretetype(eltype(pv))
            unique_types = unique(typeof.(pv))
            @debug "Non-concrete param types. Promoting" unique_types
            T = mapreduce(typeof, promote_type, pv)
            convert.(T, pv)
        else
            pv
        end
    end
    
    total_eltype = let
        states_eltype = mapreduce(promote_type, states_partitioned) do v
            eltype(eltype(v))
        end
        u0map_eltype = mapreduce(promote_type, u0map; init=Union{}) do (k, v)
            typeof(v)
        end
        numeric_params_eltype = mapreduce(promote_type, params_partitioned) do v
            if isconcretetype(eltype(v))
                promote_numeric_param_eltype(eltype(v))
            else
                mapreduce(promote_type, v) do params
                    promote_numeric_param_eltype(typeof(params))
                end
            end
        end
        numeric_param_map_eltype = let numeric_params_from_map = [v for (_, v) in param_map if v isa Number]
            mapreduce(typeof, promote_type, numeric_params_from_map; init=Union{})
        end
        promote_type(states_eltype, u0map_eltype, numeric_params_eltype, numeric_param_map_eltype)
    end

    re_eltype(s::SubsystemStates{T}) where {T} = convert(SubsystemStates{T, total_eltype}, s) 
    states_partitioned = tupmap(states_partitioned) do v
        if eltype(eltype(v)) <: total_eltype
            v
        else
            re_eltype.(v)
        end
    end
    
    length(states_partitioned) == length(params_partitioned) ||
        error("Incompatible state and parameter lengths")
    for i ∈ eachindex(states_partitioned, params_partitioned)
        length(states_partitioned[i]) == length(params_partitioned[i]) ||
            error("Incompatible state and parameter lengths")
    end

    for nc ∈ 1:length(connection_matrices)
        for i ∈ eachindex(states_partitioned)
            for k ∈ eachindex(states_partitioned)
                M = connection_matrices[nc][i, k]
                if !(M isa NotConnected)
                    size(M) == (length(states_partitioned[i]), length(states_partitioned[k])) ||
                        error("Connection sub-matrix ($nc, $i, $k) has an incorrect size, expected $((length(states_partitioned[i]), length(states_partitioned[k]))), got $(size(connection_matrices[nc][i, k])).")
                end
            end
        end
    end
    partition_plan = let offset=Ref(0)
        map(states_partitioned) do v
            sz = (length(eltype(v)), length(v))
            L = prod(sz)
            inds = (1:L) .+ offset[]
            plan = (;inds, sz, TVal=Val(eltype(v)))
            offset[] += L
            plan
        end
    end
    symbolic_indexing_namemap = GraphNamemap(
        names_partitioned, states_partitioned, params_partitioned, connection_matrices
    )
    discrete_event_cache = ntuple(length(states_partitioned)) do i
        len = has_discrete_events(eltype(states_partitioned[i])) ? length(states_partitioned[i]) : 0
        falses(len)
    end
    GraphSystemParameters(;
                          graph=g,
                          states_partitioned,
                          params_partitioned,
                          connection_matrices,
                          scheduler,
                          partition_plan,
                          discrete_event_cache,
                          names_partitioned,
                          symbolic_indexing_namemap,
                          extra_params)
end

function make_u0(g::GraphSystemParameters; allow_nonconcrete=false)
    (; states_partitioned) = g
    u0 = reduce(vcat, map(v -> reduce(vcat, v), states_partitioned))
    if !allow_nonconcrete && !isconcretetype(eltype(u0)) && !all(isconcretetype ∘ eltype, states_partitioned)
        error(ArgumentError("The provided subsystem states do not have a concrete eltype. All partitions must contain the same eltype. Got `eltype(u) = $(eltype(u0))`."))
    end
    u0
end

function make_callback(g::GraphSystemParameters; global_events=())
    (; states_partitioned) = g
    nce = sum(states_partitioned) do v
        has_continuous_events(eltype(v)) ? length(v) : 0
    end
    nde = sum(states_partitioned) do v
        has_discrete_events(eltype(v)) ? length(v) : 0
    end
    ce = nce > 0 ? VectorContinuousCallback(continuous_condition, continuous_affect!, nce) : nothing
    de = nde > 0 ? DiscreteCallback(discrete_condition, discrete_affect!) : nothing
    callback = CallbackSet(ce, de, global_events...)
end

function make_dtmax(p::GraphSystemParameters)
    (; params_partitioned) = p
    init = typemax(Float64)
    minimum(params_partitioned; init) do v
        minimum(v; init) do p
            get(NamedTuple(p), :dtmax, init)
        end
    end
end
