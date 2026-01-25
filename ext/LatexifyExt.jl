module LatexifyExt

using Symbolics

using Latexify

using GraphDynamics:
    GraphDynamics,
    get_tag,
    get_name,
    initialize_input,
    subsystem_differential,
    get_states,
    get_params,
    to_subsystem,
    Subsystem,
    SubsystemStates,
    SubsystemParams,
    ConnectionRule,
    connection_property_namemap,
    GraphSystem,
    nodes,
    connections,
    graph_equations,
    node_equations,
    connection_equations,
    ConnectionIndex,
    GraphSystemConnection,
    system_wiring_rule!,
    PartitioningGraphSystem

@variables t
const _D = Differential(t)
const NAMESPACE_SEPARATOR_SYMBOL = Symbol(Symbolics.NAMESPACE_SEPARATOR)

function GraphDynamics.node_equations(sys::T) where T
    sym_subsys, state_vars, input_vars = to_symbolic_subsystem(sys)
    rhss = subsystem_differential(sym_subsys, input_vars, t)
    eqs = [_D(state_vars[i]) ~ rhss[i] for i in 1:length(rhss)]
end

function GraphDynamics.node_equations(list_of_sys::Union{Tuple, Vector, Set, Base.KeySet})
    equations = map(collect(list_of_sys)) do n 
        node_equations(n)
    end
    reduce(vcat, equations)
end

function GraphDynamics.connection_equations(sys::GraphSystem, src, dst)
    subgraph = GraphSystem()
    for (; data) in connections(sys, src, dst)
        add_connection!(subgraph, src, dst; data...)
    end
    subnodes = [n for n ∈ nodes(subgraph) if n ∉ (src, dst)]
    [connection_equations(subgraph); node_equations(subnodes)]
end

function GraphDynamics.connection_equations(sys::PartitioningGraphSystem)
    system_cache = Dict{Any, Tuple{Subsystem, NamedTuple, NamedTuple}}()
    eqs = map(connections(sys)) do (; src, dst, conn)
        GraphDynamics.connection_equations(conn, src, dst; system_cache)
    end
    reduce(vcat, eqs)
end

function GraphDynamics.connection_equations(sys::GraphSystem)
    GraphDynamics.connection_equations(sys.flat_graph)
end

function GraphDynamics.connection_equations(conn::ConnectionRule, src, dst; system_cache=nothing)
    if isnothing(system_cache)
        sym_srcsys, _, _      = to_symbolic_subsystem(src)
        sym_dstsys, inputs, _ = to_symbolic_subsystem(dst)
    else
        sym_srcsys, _, _ = get!(system_cache, src) do
            to_symbolic_subsystem(src)
        end
        sym_dstsys, inputs, _ = get!(system_cache, dst) do
            to_symbolic_subsystem(dst)
        end
    end
    conn_props = connection_property_namemap(conn, get_name(src), get_name(dst))
    syms = map(collect(pairs(conn_props))) do (k, v)
        if getfield(conn, k) isa Function
            only(@variables $v(..))
        else
            only(@variables $v)
        end
    end

    cons = typeof(conn).name.wrapper
    sym_conn = cons(syms...)
    rhss = sym_conn(sym_srcsys, sym_dstsys, t)
    lhss = gen_variables(Val(keys(rhss)), Val(get_name(dst)), Val(true))

    eqs = map(zip(lhss, rhss)) do (lhs, rhs)
        lhs ~ rhs
    end
end

function namespaced_vars(namespace, syms; of_t = true)
    ns_syms = map(syms) do name
        Symbol(namespace, NAMESPACE_SEPARATOR_SYMBOL, name)
    end

    if of_t
        Tuple([Expr(:call, sym, :t) for sym in ns_syms])
    else
        Tuple(ns_syms)
    end
end

function to_symbolic_subsystem(sys)
    to_symbolic_subsystem(to_subsystem(sys); namespace = get_name(sys))
end

@generated function gen_variables(::Val{syms}, ::Val{namespace}, ::Val{of_t}) where {syms, namespace, of_t}
    nsyms = namespaced_vars(namespace, syms; of_t)
    quote
        vars = @variables $(Expr(:tuple, nsyms...))
        NamedTuple{$(syms)}(vars)
    end
end

function to_symbolic_subsystem(sys::Subsystem{T}; namespace = :sys) where T
    states = propertynames(get_states(sys))
    params = propertynames(get_params(sys))
    inputs = propertynames(initialize_input(sys))

    ns = Val(namespace)
    state_vars = gen_variables(Val(states), ns, Val(true))
    input_vars = gen_variables(Val(inputs), ns, Val(true))
    param_vars = gen_variables(Val(params), ns, Val(false))

    sym_subsys_states = SubsystemStates{get_tag(sys)}(NamedTuple{states}(state_vars))
    sym_subsys_params = SubsystemParams{get_tag(sys)}(NamedTuple{params}(param_vars))
    sym_subsys = Subsystem{get_tag(sys)}(sym_subsys_states, sym_subsys_params)

    sym_subsys, state_vars, input_vars
end

@latexrecipe function f(sys::Union{GraphSystem, PartitioningGraphSystem}; show_connection_equations = true)
    sys = sys isa GraphSystem ? sys.flat_graph : sys
    if show_connection_equations
        return latexify([node_equations(collect(nodes(sys)));
                         connection_equations(sys)])
    else
        return latexify(node_equations(collect(nodes(sys))))
    end
end

end
