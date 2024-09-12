# Fallback for failed problem construction
function (::Type{T})(g::Gsys, args...; kwargs...) where {T <: SciMLBase.AbstractSciMLProblem, Gsys <: GraphSystem}
    throw(ArgumentError("GraphSystems.jl does not yet support the use of $(Gsys.name.wrapper) in $T.\nThis is either a feature that is not yet supported, or you may have accidentally done something incorrect such as passing an ODEGraphSystem to an SDEProblem."))
end

function SciMLBase.ODEProblem(g::ODEGraphSystem, u0map, tspan, param_map=[]; scheduler=SerialScheduler(), tstops=Float64[], kwargs...)
    (; f, u, tspan, p, callback, ev_times) = _problem(g, u0map, tspan, param_map; scheduler)
    tstops = vcat(tstops, ev_times)
    ODEProblem(f, u, tspan, p; callback, tstops, kwargs...)
end
function SciMLBase.SDEProblem(g::SDEGraphSystem, u0map, tspan, param_map=[];
                              scheduler=SerialScheduler(), tstops=Float64[], kwargs...)
    (; f, u, tspan, p, callback, ev_times) = _problem(g, u0map, tspan, param_map; scheduler)
    
    noise_rate_prototype = nothing # zeros(length(u)) # this'll need to change once we support correlated noise
    SDEProblem(f, graph_noise!, u, tspan, p; callback, noise_rate_prototype, tstops = vcat(tstops, ev_times), kwargs...)
end

Base.@kwdef struct GraphSystemParameters{SSP, CM, S, STV}
    subsystem_params_partitioned::SSP
    connection_matrices::CM
    scheduler::S
    state_types_val::STV
end

function _problem(g::GraphSystem, u0map, tspan, param_map=[]; scheduler=SerialScheduler)
    isempty(u0map) || error("Specifying a state map is not yet implemented")
    isempty(param_map)  || error("Specifying a parameter map is not yet implemented")

    (; states_partitioned, connection_matrices) = g
    (; subsystem_params_partitioned,
     ev_times,
     composite_discrete_events_partitioned,
     composite_continuous_events_partitioned,) = g.params

    length(states_partitioned) == length(subsystem_params_partitioned) ||
        error("Incompatible state and parameter lengths")
    for i ∈ eachindex(states_partitioned, subsystem_params_partitioned)
        length(states_partitioned[i]) == length(subsystem_params_partitioned[i]) ||
            error("Incompatible state and parameter lengths")
    end
    for nc ∈ eachindex(connection_matrices)
        for i ∈ eachindex(states_partitioned)
            for k ∈ eachindex(states_partitioned)
                size(connection_matrices[i, k]) == (length(states_partitioned[i]), length(states_partitioned[k])) ||
                    error("Connection sub-matrix ($i, $k) has an incorrect size, expected $((length(states_partitioned[i]), length(states_partitioned[k]))), got $(size(connection_matrices[i, k])).")
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

    ce = nce > 0 ? VectorContinuousCallback(continuous_condition, continuous_affect!, nce) : nothing
    de = nde > 0 ? DiscreteCallback(discrete_condition, discrete_affect!) : nothing
    callback = CallbackSet(ce, de, composite_discrete_callbacks(composite_discrete_events_partitioned))
    f = GraphSystemFunction(graph_ode!, g)
    p = GraphSystemParameters(;subsystem_params_partitioned, connection_matrices, scheduler, state_types_val)
    (; f, u, tspan, p, callback, ev_times)
end

composite_discrete_callbacks(::Nothing) = nothing
function composite_discrete_callbacks(composite_discrete_events_partitioned::NTuple{Len, Any}) where {Len}
    function composite_event_conditions(u, t, integrator)
        (;subsystem_params_partitioned, state_types_val, connection_matrices) = integrator.p
        params_partitioned = subsystem_params_partitioned
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
        (;subsystem_params_partitioned, state_types_val, connection_matrices) = integrator.p
        params_partitioned = subsystem_params_partitioned
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
