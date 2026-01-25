module SymbolicsExt

using Symbolics: Symbolics, tosymbol, Num
using GraphDynamics: GraphDynamics, GraphSystemParameters, GraphNamemap
using SymbolicIndexingInterface: SymbolicIndexingInterface

function SymbolicIndexingInterface.is_variable(sys::GraphNamemap, var::Num)
    SymbolicIndexingInterface.is_variable(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.variable_index(sys::GraphNamemap, var::Num)
    SymbolicIndexingInterface.variable_index(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.is_parameter(sys::GraphNamemap, var::Num)
    SymbolicIndexingInterface.is_parameter(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.parameter_index(sys::GraphNamemap, var::Num)
    SymbolicIndexingInterface.parameter_index(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.is_independent_variable(sys::GraphNamemap, var::Num)
    SymbolicIndexingInterface.is_independent_variable(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.is_observed(sys::GraphNamemap, var::Num)
    SymbolicIndexingInterface.is_observed(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.observed(sys::GraphNamemap, var::Num)
    SymbolicIndexingInterface.observed(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.observed(sys::GraphNamemap, vars::Union{Vector{Num}, Tuple{Vararg{Num}}})
    SymbolicIndexingInterface.observed(sys, tosymbol.(vars; escape=false))
end

end
