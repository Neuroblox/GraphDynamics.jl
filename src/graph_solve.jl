#----------------------------------------------------------
function graph_ode!(du::ArrayPartition,
                    u::ArrayPartition,
                    (;params_partitioned, connection_matrices, scheduler, state_types_val)::P,
                    t) where {P}
    states_partitioned  = to_vec_o_states( u.x, state_types_val)
    dstates_partitioned = to_vec_o_states(du.x, state_types_val)
    _graph_ode!(dstates_partitioned, states_partitioned, params_partitioned, connection_matrices, scheduler, t)
end

function _graph_ode!(dstates_partitioned::NTuple{Len, Any}#=mutated=#,
                     states_partitioned ::NTuple{Len, Any},
                     params_partitioned ::NTuple{Len, Any},
                     connection_matrices::ConnectionMatrices{NConn},
                     scheduler,
                     t,) where {Len, NConn}
    @unroll 16 for i ∈ 1:Len
        f = make_graph_ode_mapping_f(
            Val(i),
            dstates_partitioned,
            states_partitioned,
            params_partitioned,
            connection_matrices,
            scheduler,
            t
        )
        tforeach(f, eachindex(states_partitioned[i]); scheduler)
    end
end


function _graph_ode!(dstates_partitioned::NTuple{Len, Any}#=mutated=#,
                     states_partitioned ::NTuple{Len, Any},
                     params_partitioned ::NTuple{Len, Any},
                     connection_matrices::ConnectionMatrices{NConn},
                     scheduler::SerialScheduler,
                     t) where {Len, NConn}

    @unroll 16 for i ∈ 1:Len
        for j ∈ eachindex(states_partitioned[i])
            _graph_ode_mapping_f(j, Val(i),
                                 dstates_partitioned,
                                 states_partitioned,
                                 params_partitioned,
                                 connection_matrices,
                                 scheduler,
                                 t)
        end
    end
end

function make_graph_ode_mapping_f(I,
                                  dstates_partitioned,
                                  states_partitioned,
                                  params_partitioned,
                                  connection_matrices,
                                  scheduler,
                                  t)
    j -> _graph_ode_mapping_f(
        j,
        I,
        dstates_partitioned,
        states_partitioned,
        params_partitioned,
        connection_matrices,
        scheduler,
        t
    )
end
function _graph_ode_mapping_f(j, ::Val{i},
                              dstates_partitioned::NTuple{Len, Any}#=mutated=#,
                              states_partitioned ::NTuple{Len, Any},
                              params_partitioned ::NTuple{Len, Any},
                              connection_matrices::ConnectionMatrices{NConn},
                              scheduler,
                              t) where {i, Len, NConn}
    sys_dst = Subsystem(states_partitioned[i][j], params_partitioned[i][j])
    input = if subsystem_differential_requires_inputs(sys_dst)
        calculate_inputs(Val(i), j, states_partitioned, params_partitioned, connection_matrices, t)
    else
        initialize_input(sys_dst)
    end
    apply_subsystem_differential!(@view(dstates_partitioned[i][j]), sys_dst, input, t)
end

function calculate_inputs(::Val{i}, j,
                          states_partitioned::NTuple{Len, Any},
                          params_partitioned::NTuple{Len, Any},
                          connection_matrices::ConnectionMatrices{NConn},
                          #TODO: remove the =nothing fallback
                          t=nothing)  where {i, Len, NConn}
    state  = @inbounds states_partitioned[i][j]
    subsys = @inbounds Subsystem(state, params_partitioned[i][j])
    input  = initialize_input(subsys)
    @unroll 16 for k ∈ 1:Len
        @unroll 8 for nc ∈ 1:NConn
            @inbounds begin
                M = connection_matrices[nc][k, i]
                input′ = combine_inputs(subsys, M, j, states_partitioned[k], params_partitioned[k], t, SerialScheduler();)
                input = combine(input, input′)
            end
        end
    end
    input
end

"""
    combine_inputs(subsys::Subsystem,
                   M::AbstractMatrix{<:ConnectionRule},
                   j::Integer,
                   states_partitioned::AbstractVector{<:SubsystemStates},
                   params_partitioned::abstractvector{<:SubsystemParams},
                   scheduler;
                   init=initialize_input(subsys))

Given an input graph-subsystem `subsys`, a (sub-)connection matrix `M` whose `j`-th column describes the
connections between `subsys` and a (sub-)list of subsystems defined by `states_partitioned` and
`params_partitioned`, compute the total input that should be passed to `subsys` by combining all the input
signals sent from each connected subsystem.

e.g. if the inputs are just numbers who are combined by adding them together, then this computes

```math
\\sum_{l} M[l, j](Subsystem(states_partitioned[i], params_partitioned[l]), subsys)
```
"""
function combine_inputs end

function combine_inputs(subsys, M, j, states_partitioned, params_partitioned, t, ::SerialScheduler;
                            init=initialize_input(subsys))
    acc = init
    if M isa SparseMatrixCSC
        @inbounds for (l, Mlj) ∈ maybe_sparse_enumerate_col(M, j)
            acc′ = Mlj(Subsystem(states_partitioned[l], params_partitioned[l]), subsys, t) # Now do the actual reducing step just like the above method
            acc = combine(acc, acc′)
        end
    else
        @inbounds @simd for l ∈ axes(M, 1)
            acc′ = M[l,j](Subsystem(states_partitioned[l], params_partitioned[l]), subsys, t) # Now do the actual reducing step just like the above method
            acc = combine(acc, acc′)
        end
    end
    acc
end

@inline combine_inputs(subsys, M::NotConnected, j, states_partitioned, params_partitioned, t, scheduler::SerialScheduler;
                       init=initialize_input(subsys)) = init

"""
    maybe_sparse_enumerate_col(M::AbstractMatrix, j)

Equivalent to `((l, M[l, j]) for l ∈ axes(M, 1))`, except if `M` isa `SparseMatrixCSC`, this will
only iterate over the non-zero values of `M`.
"""
function maybe_sparse_enumerate_col(M::SparseMatrixCSC, j)
    rows = rowvals(M)
    vals = nonzeros(M)
    Iterators.map(nzrange(M, j)) do idx
        @inbounds begin
            l = rows[idx]
            Mlj = vals[idx]
        end
        (l, Mlj)
    end
end
function maybe_sparse_enumerate_col(M::AbstractMatrix, j)
    enumerate(@view(M[:, j]))
end


#----------------------------------------------------------
# Infra. for stochastic noise
#----------------------------------------------------------

function graph_noise!(du,
                      u::ArrayPartition,
                      (;params_partitioned, connection_matrices, scheduler, state_types_val)::P,
                      t) where {P}
    states_partitioned  = to_vec_o_states( u.x, state_types_val)
    dstates_partitioned = to_vec_o_states(du.x, state_types_val)
    _graph_noise!(dstates_partitioned, states_partitioned, params_partitioned, connection_matrices, t)
end

#TODO: graph_noise! currently doesn't support noise dependant on inputs
# I'm not sure this is a practical problem, but might be something we want to support
# in the future.

#TODO: We currently only support diagonal noise (that is, the noise source in one
# equation can't depend on the noise source from another equation). This needs to be
# generalized, but how to handle it best will require a lot of thought.
function _graph_noise!(dstates_partitioned#=mutated=#,
                                  states_partitioned ::NTuple{Len, Any},
                                  params_partitioned ::NTuple{Len, Any},
                                  connection_matrices::ConnectionMatrices{NConn},
                                  t) where {Len, NConn}

    idx = 1
    for i ∈ 1:Len
        begin
            l  = @inbounds length(states_partitioned[i][1])
            js = @inbounds eachindex(states_partitioned[i])
            @unroll 32 for j ∈ js
                @inbounds begin
                    sys = Subsystem(states_partitioned[i][j], params_partitioned[i][j])
                    apply_subsystem_noise!(@view(dstates_partitioned[i].data[:, j]), sys, t)
                    idx += l
                end
            end
        end
    end
end


#----------------------------------------------------------
# Infra. for continuous events. 
#----------------------------------------------------------

function continuous_condition(out, u, t, integrator)
    (;params_partitioned, state_types_val, connection_matrices) = integrator.p
    states_partitioned = to_vec_o_states(u.x, state_types_val)
    _continuous_condition!(out, states_partitioned, params_partitioned, connection_matrices, t)
end

function _continuous_condition!(out,
                                states_partitioned   ::NTuple{Len, Any},
                                params_partitioned   ::NTuple{Len, Any},
                                connection_matrices,
                                t) where {Len}

    idx = 0
    @unroll 16 for i ∈ 1:Len
        if has_continuous_events(eltype(states_partitioned[i]))
            for j ∈ eachindex(states_partitioned[i])
                idx += 1
                F = ForeachConnectedSubsystem{i}(j, states_partitioned, params_partitioned, connection_matrices)
                sys = Subsystem(states_partitioned[i][j], params_partitioned[i][j])
                out[idx] = continuous_event_condition(sys, t, F)
            end
        end
    end
end


function continuous_affect!(integrator, idx)
    (;params_partitioned, state_types_val, connection_matrices) = integrator.p
    state_data = integrator.u.x
    states_partitioned = to_vec_o_states(state_data, state_types_val)
    _continuous_affect!(integrator, states_partitioned, params_partitioned, connection_matrices, idx)
end

function _continuous_affect!(integrator,
                             states_partitioned::NTuple{Len, Any},
                             params_partitioned::NTuple{Len, Any},
                             connection_matrices::ConnectionMatrices{NConn},
                             idx) where {Len, NConn}
    offset=0
    t = integrator.t
    for i ∈ 1:Len
        @inbounds begin
            if has_continuous_events(eltype(states_partitioned[i]))
                N = length(states_partitioned[i])
                js = (1:N) .+ offset
                if idx ∈ js
                    j = idx - offset
                    sview = @view states_partitioned[i][j]
                    pview = @view params_partitioned[i][j]
                    sys = Subsystem(states_partitioned[i][j], params_partitioned[i][j])
                    F = ForeachConnectedSubsystem{i}(j, states_partitioned, params_partitioned, connection_matrices)
                    if continuous_events_require_inputs(sys)
                        input = calculate_inputs(Val(i), j, states_partitioned, params_partitioned, connection_matrices, t)
                        apply_continuous_event!(integrator, sview, pview, sys, F, input)
                    else
                        apply_continuous_event!(integrator, sview, pview, sys, F)
                    end
                end
                offset += N
            end
        end
    end
end 

#----------------------------------------------------------
# Infra. for discrete events. 
#----------------------------------------------------------

function discrete_condition(u, t, integrator)
    (;params_partitioned, state_types_val, connection_matrices, discrete_event_cache) = integrator.p
    states_partitioned = to_vec_o_states(u.x, state_types_val)
    _discrete_condition!(states_partitioned, params_partitioned, t, connection_matrices, discrete_event_cache)
end

@generated function _discrete_condition!(states_partitioned    ::NTuple{Len, Any},
                                         params_partitioned    ::NTuple{Len, Any},
                                         t,
                                         connection_matrices::ConnectionMatrices{NConn},
                                         discrete_event_cache  ::NTuple{Len, Any}) where {Len, NConn}
    quote
        trigger = false
        @nexprs $Len i -> begin
            if has_discrete_events(eltype(states_partitioned[i]))
                for j ∈ eachindex(states_partitioned[i])
                    F = ForeachConnectedSubsystem{i}(j, states_partitioned, params_partitioned, connection_matrices)
                    sys = Subsystem(states_partitioned[i][j], params_partitioned[i][j])
                    cond = discrete_event_condition(sys, t, F)
                    trigger |= cond
                    discrete_event_cache[i][j] = cond
                end
            end
        end
        trigger && return true
        @nexprs $NConn nc -> begin
            @nexprs $Len i -> begin
                @nexprs $Len k -> begin
                    M = connection_matrices[nc][k, i]
                    if has_discrete_events(eltype(M))
                        for j ∈ eachindex(states_partitioned[i])
                            for (l, Mlj) ∈ maybe_sparse_enumerate_col(M, j)
                                discrete_event_condition(Mlj, t) && return true
                            end
                        end
                    end
                end
            end
        end
        false
    end
end

function discrete_affect!(integrator)
    (;params_partitioned, state_types_val, connection_matrices, discrete_event_cache) = integrator.p
    state_data = integrator.u.x
    states_partitioned = to_vec_o_states(state_data, state_types_val)
    _discrete_affect!(integrator,
                      states_partitioned,
                      params_partitioned,
                      connection_matrices,
                      discrete_event_cache,
                      integrator.t)
end

@generated function _discrete_affect!(integrator,
                                      states_partitioned    ::NTuple{Len, Any},
                                      params_partitioned    ::NTuple{Len, Any},
                                      connection_matrices::ConnectionMatrices{NConn},
                                      discrete_event_cache  ::NTuple{Len, Any},
                                      t) where {Len, NConn}
    quote
        @nexprs $Len i -> begin
            # First we apply events to the states
            if has_discrete_events(eltype(states_partitioned[i]))
                @inbounds for j ∈ eachindex(states_partitioned[i])
                    if discrete_event_cache[i][j]
                        sys = Subsystem(states_partitioned[i][j], params_partitioned[i][j])
                        sview = @view states_partitioned[i][j]
                        pview = @view params_partitioned[i][j]
                        F = ForeachConnectedSubsystem{i}(j, states_partitioned, params_partitioned, connection_matrices)
                        if discrete_events_require_inputs(sys)
                            input = calculate_inputs(Val(i), j, states_partitioned, params_partitioned, connection_matrices, t)
                            apply_discrete_event!(integrator, sview, pview, sys, F, input)
                        else
                            apply_discrete_event!(integrator, sview, pview, sys, F)
                        end
                    end
                end
                discrete_event_cache[i] .= false
            end
            # Then we do the connection events
            @nexprs $NConn nc -> begin
                @nexprs $Len k -> begin
                    f =  _discrete_connection_affect!(Val(i), Val(k), Val(nc), t,
                                                      states_partitioned, params_partitioned, connection_matrices,
                                                      integrator)
                    foreach(f, eachindex(states_partitioned[i]))
                end
            end
        end
    end
end

function _discrete_connection_affect!(::Val{i}, ::Val{k}, ::Val{nc}, t, 
                                      states_partitioned::NTuple{Len, Any},
                                      params_partitioned::NTuple{Len, Any},
                                      connection_matrices::ConnectionMatrices{NConn},
                                      integrator) where {i, k, nc, Len, NConn}
    function (j)
        sys_dst = Subsystem(states_partitioned[i][j], params_partitioned[i][j])
        sview_dst = @view states_partitioned[i][j]
        pview_dst = @view params_partitioned[i][j]
        M = connection_matrices.matrices[nc].data[k][i]
        if has_discrete_events(eltype(M))
            for (l, Mlj) ∈ maybe_sparse_enumerate_col(M, j)
                if discrete_event_condition(Mlj, t)
                    sys_src = Subsystem(states_partitioned[k][l], params_partitioned[k][l])
                    sview_src = @view states_partitioned[k][l]
                    pview_src = @view params_partitioned[k][l]
                    if discrete_events_require_inputs(typeof(Mlj))
                        input_dst = calculate_inputs(Val(i), j,
                                                     states_partitioned,
                                                     params_partitioned,
                                                     connection_matrices,
                                                     t)
                        input_src = calculate_inputs(Val(k), l,
                                                     states_partitioned,
                                                     params_partitioned,
                                                     connection_matrices,
                                                     t)
                        apply_discrete_event!(integrator,
                                              sview_src, pview_src,
                                              sview_dst, pview_dst,
                                              Mlj,
                                              sys_src, input_src,
                                              sys_dst, input_dst)
                    else
                        apply_discrete_event!(integrator,
                                              sview_src, pview_src,
                                              sview_dst, pview_dst,
                                              Mlj,
                                              sys_src, sys_dst)
                    end
                end
            end
        end
    end
end


#-----------------------------------------------------------------------

"""
    ForeachConnectedSubsystem

This is a callable struct which takes in a function, and then calls that function on each subsystem which has a connection leading to it
from some previously specified subsystem.

That is, writing
```julia
F = ForeachConnectedSubsystem{k}(l, states_partitioned, params_partitioned, connection_matrices)

F() do conn, sys_dst, states_view_dst, params_view_dst
    [...]
end
```
is like a type stable version of writing
```
for i in eachindex(states_partitioned)
    for nc in eachindex(connection_matrices)
        M = connection_matrices[nc][i, k]
        for j in eachindex(states_partitioned[k])
            conn = M[l, j]
            if !iszero(conn)
                states_view_dst = @view states_partitioned[i][j]
                params_view_dst = @view params_partitioned[i][j]
                sys_dst = Subsystem(states_view_dst[], params_view_dst[])
                [...] # <------- User code here
            ends
        end
    end
end
```
"""
struct ForeachConnectedSubsystem{k, Len, NConn, S, P, CMs}
    l::Int
    states_partitioned::S
    params_partitioned::P
    connection_matrices::CMs
    function ForeachConnectedSubsystem{k}(l,
                                          states_partitioned::NTuple{Len, Any},
                                          params_partitioned::NTuple{Len, Any},
                                          connection_matrices::ConnectionMatrices{NConn}) where {k, Len, NConn}
        S = typeof(states_partitioned)
        P = typeof(params_partitioned)
        CMs = typeof(connection_matrices)
        new{k, Len, NConn, S, P, CMs}(l, states_partitioned, params_partitioned, connection_matrices)
    end
end

@generated function Base.mapreduce(f::F, op::Op, FCS::ForeachConnectedSubsystem{k, Len, NConn}; init) where {k, Len, NConn, F, Op}
    quote
        (;l, states_partitioned, params_partitioned, connection_matrices) = FCS
        state = init
        @nexprs $Len i -> begin
            @nexprs $NConn nc -> begin
                M = connection_matrices[nc][k, i]
                if M isa NotConnected
                    nothing
                else
                    for j ∈ eachindex(states_partitioned[i])
                        @inbounds conn = M[l, j]
                        if !iszero(conn)
                            @inbounds states_view_dst = @view states_partitioned[i][j]
                            @inbounds params_view_dst = @view params_partitioned[i][j]
                            sys_dst = Subsystem(states_view_dst[], params_view_dst[])
                            res = f(conn, sys_dst, states_view_dst, params_view_dst)
                            state = op(state, res)
                        end
                    end
                end 
            end
        end
        state
    end
end
(FCS::ForeachConnectedSubsystem)(f::F) where {F} = mapreduce(f, (_, _) -> nothing, FCS; init=nothing)


