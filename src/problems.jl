# Fallback for failed problem construction
function (::Type{T})(g::Gsys, args...; kwargs...) where {T <: SciMLBase.AbstractSciMLProblem, Gsys <: GraphSystem}
    throw(ArgumentError("GraphSystems.jl does not yet support the use of $(Gsys.name.wrapper) in $T.\nThis is either a feature that is not yet supported, or you may have accidentally done something incorrect such as passing an ODEGraphSystem to an SDEProblem."))
end

function SciMLBase.ODEProblem(g::ODEGraphSystem, u0map, tspan, param_map=[];
                              scheduler=SerialScheduler(), tstops=Float64[],
                              allow_nonconcrete=false, kwargs...)
    nt = _problem(g, tspan; scheduler, allow_nonconcrete, u0map, param_map)
    (; f, u, tspan, p, callback) = nt
    tstops = vcat(tstops, nt.tstops)
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(f, u, tspan, p; callback, tstops, kwargs...)
    for (k, v) ∈ u0map
        setu(prob, k)(prob, v)
    end
    for (k, v) ∈ param_map
        setp(prob, k)(prob, v)
    end
    prob
end
function SciMLBase.SDEProblem(g::SDEGraphSystem, u0map, tspan, param_map=[];
                              scheduler=SerialScheduler(), tstops=Float64[],
                              allow_nonconcrete=false, kwargs...)
    nt = _problem(g, tspan; scheduler, allow_nonconcrete, u0map, param_map)
    (; f, u, tspan, p, callback) = nt
    
    noise_rate_prototype = nothing # zeros(length(u)) # this'll need to change once we support correlated noise
    prob = SDEProblem(f, graph_noise!, u, tspan, p; callback, noise_rate_prototype, tstops = vcat(tstops, nt.tstops), kwargs...)
    for (k, v) ∈ u0map
        setu(prob, k)(prob, v)
    end
    for (k, v) ∈ param_map
        setp(prob, k)(prob, v)
    end
    prob
end

Base.@kwdef struct GraphSystemParameters{PP, CM, S, STV, DEC, EP<:NamedTuple}
    params_partitioned::PP
    connection_matrices::CM
    scheduler::S
    state_types_val::STV
    discrete_event_cache::DEC
    extra_params::EP=(;)
end

function _problem(g::GraphSystem, tspan; scheduler, allow_nonconcrete, u0map, param_map)
    (; states_partitioned,
     params_partitioned,
     connection_matrices,
     tstops,
     composite_discrete_events_partitioned,
     composite_continuous_events_partitioned,) = g

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
    state_types_val = Val(Tuple{map(eltype, states_partitioned)...})
    
    u = ArrayPartition(map(v -> stack(v), states_partitioned))
    if !allow_nonconcrete && !isconcretetype(eltype(u)) && !all(isconcretetype ∘ eltype, states_partitioned)
        error(ArgumentError("The provided subsystem states do not have a concrete eltype. All partitions must contain the same eltype. Got `eltype(u) = $(eltype(u))`."))
    end

    discrete_event_cache = ntuple(length(states_partitioned)) do i
        len = has_discrete_events(eltype(states_partitioned[i])) ? length(states_partitioned[i]) : 0
        falses(len)
    end

    ce = nce > 0 ? VectorContinuousCallback(continuous_condition, continuous_affect!, nce) : nothing
    de = nde > 0 ? DiscreteCallback(discrete_condition, discrete_affect!) : nothing
    callback = CallbackSet(ce, de, composite_discrete_callbacks(composite_discrete_events_partitioned))
    f = GraphSystemFunction(graph_ode!, g)
    p = GraphSystemParameters(; params_partitioned,
                              connection_matrices,
                              scheduler,
                              state_types_val,
                              discrete_event_cache)
    (; f, u, tspan, p, callback, tstops)
end

composite_discrete_callbacks(::Nothing) = nothing
function composite_discrete_callbacks(composite_discrete_events_partitioned::NTuple{Len, Any}) where {Len}
    function composite_event_conditions(u, t, integrator)
        (;params_partitioned, state_types_val, connection_matrices) = integrator.p
        states_partitioned = to_vec_o_states(u.x, state_types_val)
        for i ∈ 1:Len
            for j ∈ eachindex(composite_discrete_events_partitioned[i])
                ev = composite_discrete_events_partitioned[i][j]
                if discrete_event_condition(states_partitioned,
                                            params_partitioned,
                                            connection_matrices,
                                            ev, t)
                    return true
                end
            end
        end
        false
    end
    function composite_event_affect!(integrator)
        (;params_partitioned, state_types_val, connection_matrices) = integrator.p
        states_partitioned = to_vec_o_states(integrator.u.x, state_types_val)
        (;t) = integrator
        for i ∈ 1:Len
            for j ∈ eachindex(composite_discrete_events_partitioned[i])
                ev = composite_discrete_events_partitioned[i][j]
                if discrete_event_condition(states_partitioned,
                                            params_partitioned,
                                            connection_matrices,
                                            ev, t)
                    apply_discrete_event!(integrator,
                                          states_partitioned,
                                          params_partitioned,
                                          connection_matrices,
                                          t, ev)
                end
            end
        end
    end
    DiscreteCallback(composite_event_conditions, composite_event_affect!)
end

composite_continuous_callbacks(::Nothing) = nothing
function composite_continuous_callbacks(::NTuple{Len, Any}) where {Len}
    error("Composite continuous events are not yet implemented")
end
