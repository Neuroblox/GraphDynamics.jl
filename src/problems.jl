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
    # let ukeys = map(first, u0map),
    #     uvals = map(last, u0map)
    #     setu(prob, ukeys)(prob, uvals)
    # end
    # let pkeys = map(first, param_map),
    #     pvals = map(last, param_map)
    #     setp(prob, pkeys)(prob, pvals)
    # end
     for (k, v) ∈ u0map
         setu(prob, k)(prob, v)
     end
    for (k, v) ∈ param_map
        setp(prob, k)(prob, v)
    end
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
    let ukeys = map(first, u0map),
        uvals = map(last, u0map)
        setu(prob, ukeys)(prob, uvals)
    end
    let pkeys = map(first, param_map),
        pvals = map(last, param_map)
        setp(prob, pkeys)(prob, pvals)
    end
    prob
end

Base.@kwdef struct GraphSystemParameters{PP, CM, S, PAP, DEC, EP<:NamedTuple}
    params_partitioned::PP
    connection_matrices::CM
    scheduler::S
    partition_plan::PAP
    discrete_event_cache::DEC
    extra_params::EP=(;)
end

function _problem(g::PartitionedGraphSystem, tspan; scheduler, allow_nonconcrete, u0map, param_map, global_events)
    (; states_partitioned,
     params_partitioned,
     connection_matrices,
     tstops) = g

    total_eltype = let
        states_eltype = mapreduce(promote_type, states_partitioned) do v
            eltype(eltype(v))
        end
        u0map_eltype = mapreduce(promote_type, u0map; init=Union{}) do (k, v)
            typeof(v)
        end
        promote_type(states_eltype, u0map_eltype)
    end

    re_eltype(s::SubsystemStates{T}) where {T} = convert(SubsystemStates{T, total_eltype}, s) 
    states_partitioned = map(states_partitioned) do v
        if eltype(eltype(v)) <: total_eltype && eltype(eltype(v)) !== Union{}
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
                              discrete_event_cache)

    (; f, u, tspan, p, callback, tstops)
end
