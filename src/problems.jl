# Fallback for failed problem construction
function (::Type{T})(g::GraphSystem, args...; kwargs...) where {T <: SciMLBase.AbstractSciMLProblem}
    throw(ArgumentError("GraphDynamics.jl does not yet support the use of GraphSystem in $T."))
end

function SciMLBase.ODEProblem(g::GraphSystem, u0map, tspan, param_map=[];
                              scheduler=SerialScheduler(), tstops=Float64[],
                              allow_nonconcrete=false, global_events=(), kwargs...)
    g_part = PartitionedGraphSystem(g)
    ODEProblem(g_part, u0map, tspan, param_map; scheduler, tstops, allow_nonconcrete, global_events, kwargs...)
end
function SciMLBase.SDEProblem(g::GraphSystem, u0map, tspan, param_map=[];
                              scheduler=SerialScheduler(), tstops=Float64[],
                              allow_nonconcrete=false, global_events=(), kwargs...)
    g_part = PartitionedGraphSystem(g)
    SDEProblem(g_part, u0map, tspan, param_map; scheduler, tstops, allow_nonconcrete, global_events, kwargs...)
end


function  SciMLBase.ODEProblem(g::PartitionedGraphSystem, u0map, tspan, param_map=[];
                    scheduler=SerialScheduler(), tstops=Float64[],
                    allow_nonconcrete=false, global_events=(), kwargs...)
    nt = _problem(g, tspan; scheduler, allow_nonconcrete, u0map, param_map, global_events)
    (; f, u, tspan, p, callback) = nt
    if g.is_stochastic
        error("Passed a stochastic GraphSystem to ODEProblem. You probably meant to use SDEProblem")
    end
    tstops = vcat(tstops, nt.tstops)
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(f, u, tspan, p; callback, tstops, kwargs...)
    for (k, v) ∈ u0map
        setu(prob, k)(prob, v)
    end
    @reset prob.p = set_params!!(prob.p, param_map)
    prob
end

function SciMLBase.SDEProblem(g::PartitionedGraphSystem, u0map, tspan, param_map=[];
                              scheduler=SerialScheduler(), tstops=Float64[],
                              allow_nonconcrete=false, global_events=(), kwargs...)
    nt = _problem(g, tspan; scheduler, allow_nonconcrete, u0map, param_map, global_events)
    (; f, u, tspan, p, callback) = nt
    if !g.is_stochastic
        error("Passed a non-stochastic GraphSystem to SDEProblem. You probably meant to use ODEProblem")
    end
    noise_rate_prototype = nothing # this'll need to change once we support correlated noise
    prob = SDEProblem(f, graph_noise!, u, tspan, p; callback, noise_rate_prototype, tstops = vcat(tstops, nt.tstops), kwargs...)
    for (k, v) ∈ u0map
        setu(prob, k)(prob, v)
    end
    @reset prob.p = set_params!!(prob.p, param_map)
    prob
end

Base.@kwdef struct GraphSystemParameters{PP, CM, S, PAP, DEC, NP, CONM, SNM, PNM, CNM, EP<:NamedTuple}
    params_partitioned::PP
    connection_matrices::CM
    scheduler::S
    partition_plan::PAP
    discrete_event_cache::DEC
    names_partitioned::NP
    connection_namemap::CONM
    state_namemap::SNM
    param_namemap::PNM
    compu_namemap::CNM
    extra_params::EP=(;)
end

function Base.copy(p::GraphSystemParameters)
    GraphSystemParameters(
        copy.(p.params_partitioned),
        copy(p.connection_matrices),
        p.scheduler,
        (p.partition_plan),
        copy.(p.discrete_event_cache),
        copy.(p.names_partitioned),
        copy(p.connection_namemap),
        copy(p.state_namemap),
        copy(p.param_namemap),
        copy(p.compu_namemap),
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

function _problem(g::PartitionedGraphSystem, tspan; scheduler, allow_nonconcrete, u0map, param_map, global_events)
    (; states_partitioned,
     params_partitioned,
     connection_matrices,
     tstops,
     names_partitioned,
     connection_namemap,
     state_namemap,
     param_namemap,
     compu_namemap) = g

    params_partitioned = map(params_partitioned) do v
        if !isconcretetype(eltype(v))
            unique_types = unique(typeof.(v))
            @debug "Non-concrete param types. Promoting" unique_types
            T = mapreduce(typeof, promote_type, v)
            convert.(T, v)
        else
            v
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
    states_partitioned = map(states_partitioned) do v
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
                        error("Connection sub-matrix ($nc, $i, $k) has an incorrect size, expected $((length(states_partitioned[i]), length(states_partitioned[k]))), got $(size(connection_matrices[i, k])).")
                end
            end
        end
    end
    nce = sum(states_partitioned) do v
        if has_continuous_events(eltype(v))
            length(v)
        else
            0
        end
    end
    nde = sum(states_partitioned) do v
        if has_discrete_events(eltype(v))
            length(v)
        else
            0
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
    u = reduce(vcat, map(v -> reduce(vcat, v), states_partitioned))
    if !allow_nonconcrete && !isconcretetype(eltype(u)) && !all(isconcretetype ∘ eltype, states_partitioned)
        error(ArgumentError("The provided subsystem states do not have a concrete eltype. All partitions must contain the same eltype. Got `eltype(u) = $(eltype(u))`."))
    end

    discrete_event_cache = ntuple(length(states_partitioned)) do i
        len = has_discrete_events(eltype(states_partitioned[i])) ? length(states_partitioned[i]) : 0
        falses(len)
    end

    ce = nce > 0 ? VectorContinuousCallback(continuous_condition, continuous_affect!, nce) : nothing
    de = nde > 0 ? DiscreteCallback(discrete_condition, discrete_affect!) : nothing
    callback = CallbackSet(ce, de, global_events...)
    f = GraphSystemFunction(graph_ode!, g)
    p = GraphSystemParameters(; params_partitioned,
                              connection_matrices,
                              scheduler,
                              partition_plan,
                              discrete_event_cache,
                              names_partitioned,
                              connection_namemap,
                              state_namemap,
                              param_namemap,
                              compu_namemap)

    (; f, u, tspan, p, callback, tstops)
end

SciMLStructures.isscimlstructure(::GraphSystemParameters) = true
SciMLStructures.ismutablescimlstructure(::GraphSystemParameters) = false
SciMLStructures.hasportion(::Tunable, ::GraphSystemParameters) = true

function SciMLStructures.canonicalize(::Tunable, p::GraphSystemParameters)
    paramvals = map(Iterators.flatten(p.params_partitioned)) do paramobj
        collect(values(NamedTuple(paramobj)))
    end
    buffer = reduce(vcat, paramvals)

    repack = let p = p
        function (newbuffer)
            replace(Tunable(), p, newbuffer)
        end
    end
    buffer, repack, false
end

function SciMLStructures.replace(::Tunable, p::GraphSystemParameters, newbuffer)::GraphSystemParameters
    paramobjs = Iterators.flatten(p.params_partitioned)
    N = sum([length(NamedTuple(obj)) for obj in paramobjs])
    @assert length(newbuffer) == N

    idx = 1
    new_params = map(paramobjs) do paramobj
        syms = keys(NamedTuple(paramobj))
        newparams = SubsystemParams{get_tag(paramobj)}(; (syms .=> view(newbuffer, idx:idx+length(syms)-1))...)
        idx += length(syms)
        newparams
    end
    param_types = (unique ∘ imap)(typeof, new_params)
    params_partitioned = Tuple(map(param_types) do T
        filter(new_params) do p
            p isa T
        end
    end)
    @set p.params_partitioned = params_partitioned
end
