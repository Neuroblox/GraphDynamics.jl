module MTKExt

using ModelingToolkit: ModelingToolkit, Num
using Symbolics: Symbolics, tosymbol
using GraphDynamics: GraphDynamics, GraphSystemParameters, PartitionedGraphSystem
using SymbolicIndexingInterface: SymbolicIndexingInterface

function SymbolicIndexingInterface.is_variable(sys::PartitionedGraphSystem, var::Num)
    SymbolicIndexingInterface.is_variable(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.variable_index(sys::PartitionedGraphSystem, var::Num)
    SymbolicIndexingInterface.variable_index(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.is_parameter(sys::PartitionedGraphSystem, var::Num)
    SymbolicIndexingInterface.is_parameter(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.parameter_index(sys::PartitionedGraphSystem, var::Num)
    SymbolicIndexingInterface.parameter_index(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.is_independent_variable(sys::PartitionedGraphSystem, var::Num)
    SymbolicIndexingInterface.is_independent_variable(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.is_observed(sys::PartitionedGraphSystem, var::Num)
    SymbolicIndexingInterface.is_observed(sys, tosymbol(var; escape=false))
end

function SymbolicIndexingInterface.observed(sys::PartitionedGraphSystem, var::Num)
    SymbolicIndexingInterface.observed(sys, tosymbol(var; escape=false))
end

end
