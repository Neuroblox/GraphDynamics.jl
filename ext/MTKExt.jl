module MTKExt

using ModelingToolkit: ModelingToolkit, Num
using Symbolics: Symbolics, tosymbol
using GraphDynamics: GraphDynamics, GraphSystem
using SymbolicIndexingInterface: SymbolicIndexingInterface

function SymbolicIndexingInterface.is_variable(sys::GraphSystem, var::Num)
    SymbolicIndexingInterface.is_variable(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.variable_index(sys::GraphSystem, var::Num)
    SymbolicIndexingInterface.variable_index(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.is_parameter(sys::GraphSystem, var::Num)
    SymbolicIndexingInterface.is_parameter(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.parameter_index(sys::GraphSystem, var::Num)
    SymbolicIndexingInterface.parameter_index(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.is_independent_variable(sys::GraphSystem, var::Num)
    SymbolicIndexingInterface.is_independent_variable(sys, tosymbol(var; escape=false))
end

end
