## v0.8.0

### New features

+ A `PolyesterScheduler()` scheduling object has been added which allows for parallelizing solves using Polyester.jl. This option helps performance during ODE solves where GC time dominates if multithreaded.
+ If any subsystem in a problem has a parameter named `dtmax`, then the smallest `dtmax` parameter in the system is forwarded as a keyword argument to `ODEProblem` and `SDEProblem`s to limit the maximum stepsizes allowed by the solver. 
+ If a connection type overrides `GraphDynamics.connection_needs_ctx` to give true, then connections of that type will be given a fourth `ctx` argument which gives it access to the full list of `states_partitioned`, `params_partitioned`, and `connection_matrices` when accumulating inputs.
+ GraphDynamics integrates with Latexify.jl to latexify the equations of a GraphSystem.

### Breaking Changes

+ `PartitionedGraphSystem` has been removed; `GraphSystem`s holds a field `g.flat_graph` field with a `PartitioningGraphSystem` object which actively flattens and partitions the graph during solving
+ `apply_discrete_event!` and `apply_continuous_event!` now take `SubsytemView` objects instead of separate view objects for the states and parameters. 

## GraphDynamics v0.7.0

### Breaking changes

+ `apply_discrete_event!`, `apply_continuous_event!`, and `ForeachConnectedSubsystem` have had their `vstates` and `vparams` arguments combined into a `sys_view` argument, which gives a view into the affected system for (and the connection form gets a `sys_view_src` and `sys_view_dst`). This `sys_view` can have it's fields be updated in place like so:
```julia
function GraphDynamics.apply_discrete_event!(integrator, sys_view, sys::Subsystem{MyType}, _)
    sys_view.x[] = sys.y
end
```
This will modify the `x` state or parameter of the system when the event triggers.

+ `apply_subsystem_noise!`'s first argument is now modified in the same way as the view arguments of `apply_discrete_event!`. One would now write e.g.
```julia
function GraphDynamics.apply_subsystem_noise!(vstate, sys::Subsystem{BrownianParticle}, t)
    # No noise in position, so we don't modify vstate[:x]
    vstate.v[] = sys.σ    # White noise in velocity with amplitude σ
end
```
rather than `vstate[:v] = sys.σ`

## GraphDynamics v0.6.0

[Diff since v0.5.0](https://github.com/Neuroblox/GraphDynamics.jl/compare/v0.5.0...v0.6.0)

### Breaking changes

Switched `computed_properties` and `computed_properties_with_inputs` to take a type tag instead of subsystem argument. Now, instead of adding methods like
```julia
function GraphDynamics.computed_properties_with_inputs(::Subsystem{Particle})
    a(sys, input) = input.F / sys.m 
    (; a)
end
```
users should do
```julia
function GraphDynamics.computed_properties_with_inputs(::Type{Particle})
    a(sys, input) = input.F / sys.m 
    (; a)
end
```

## GraphDynamics v0.5.0

[Diff since v0.4.9](https://github.com/Neuroblox/GraphDynamics.jl/compare/v0.4.9...v0.5.0)

### Breaking changes

- Removed the fallback `(cr::ConnectionRule)(src, dst, t) = cr(src, dst)` which was added previously to avoid breakage when we made connection rules take a time argument in addition to the `src` and `dst` subsystems. The presence of this method was a bit of a annoying crutch
- Changed the arguments of `discrete_event_condition` from just `(conn, t)` to `(conn, t, sys_src, sys_dst)` so that events can trigger based off information in the source or destination subsystems as well
- Changed the arguments of `has_discrete_events` from just `(typeof(conn),)` for connection events to `(typeof(conn), get_tag(src), get_tag(dst))`, i.e. it now also can depend on the types of the source and dest subsystems
- Changed the arguments of `event_times` from just `(conn,)` to `(conn, sys_src, sys_dst)` for connection events.

**Merged pull requests:**
- Remove `(::ConnectionRule)(src, dst, t)` fallback method; make `discrete_event_condition` on connections take `src` / `dst` args. (#44) (@MasonProtter)
