#------------------------------------------------------------
# Subsystem parameters
function SubsystemParams{Name}(nt::NT) where {Name, NT <: NamedTuple}
    SubsystemParams{Name, NT}(nt)
end

function Base.show(io::IO, params::SubsystemParams{Name}) where {Name}
    print(io, "$SubsystemParams{$Name}(", join(("$k = $v" for (k, v) ∈ pairs(NamedTuple(params))), ", "), ")")
end

function ConstructionBase.getproperties(s::SubsystemParams)
    NamedTuple(s)
end

function ConstructionBase.setproperties(s::SubsystemParams{T}, patch::NamedTuple) where {T}
    props = NamedTuple(s)
    props′ = merge(props, patch)
    if typeof(props) != typeof(props′)
        error("Type unstable change to subsystem params!")
    end
    SubsystemParams{T}(props′)
end

get_tag(::SubsystemParams{Name}) where {Name} = Name
get_tag(::Type{<:SubsystemParams{Name}}) where {Name} = Name
Base.NamedTuple(p::SubsystemParams) = getfield(p, :params)
Base.Tuple(s::SubsystemParams) = Tuple(getfield(s, :params))
Base.getproperty(p::SubsystemParams, prop::Symbol) = getproperty(NamedTuple(p), prop)
Base.propertynames(p::SubsystemParams) = propertynames(NamedTuple(p))
function Base.setindex(p::SubsystemParams{Name}, val, param) where {Name}
    SubsystemParams{Name}(Base.setindex(NamedTuple(p), val, param))
end
function Base.convert(::Type{SubsystemParams{Name, NT}}, p::SubsystemParams{Name}) where {Name, NT}
    SubsystemParams{Name}(convert(NT, NamedTuple(p)))
end


#------------------------------------------------------------
# Subsystem states
function SubsystemStates{Name, Eltype, States}(v::AbstractVector) where {Name, Eltype, States <: NamedTuple}
    SubsystemStates{Name, Eltype, States}(States(v))
end
function SubsystemStates{Name}(nt::NamedTuple{state_names, NTuple{N, Eltype}}) where {Name, state_names, N, Eltype}
    SubsystemStates{Name, Eltype, typeof(nt)}(nt)
end
function SubsystemStates{Name}(nt::NamedTuple{state_names, <:NTuple{N, Any}}) where {Name, state_names, N}
    nt_promoted = NamedTuple{state_names}(promote(nt...))
    SubsystemStates{Name}(nt_promoted)
end
function SubsystemStates{Name}(nt::NamedTuple{(), Tuple{}}) where {Name}
    SubsystemStates{Name, Union{}, NamedTuple{(), Tuple{}}}(nt)
end

function Base.show(io::IO, states::SubsystemStates{Name, Eltype}) where {Name, Eltype}
    print(io, "$SubsystemStates{$Name, $Eltype}(", join(("$k = $v" for (k, v) ∈ pairs(NamedTuple(states))), ", "), ")")
end

function Base.zero(s::SubsystemStates{Name, Eltype, States}) where {Name, Eltype, States}
    zero(typeof(s))
end
function Base.zero(::Type{SubsystemStates{Name, Eltype, NamedTuple{names, Tup}}}) where {Name, Eltype, names, Tup}
    tup = ntuple(_ -> zero(Eltype), length(names))
    SubsystemStates{Name}(NamedTuple{names}(tup))
end


Base.getindex(s::SubsystemStates, i::Integer) = NamedTuple(s)[i]
Base.size(s::SubsystemStates) = (length(typeof(s)),)
function Base.length(
    ::Type{SubsystemStates{Name, Eltype, NamedTuple{names, NTuple{N, Eltype}}}}
    ) where {Name, Eltype, names, N}
    N
end

function ConstructionBase.getproperties(s::SubsystemStates)
    NamedTuple(s)
end

function ConstructionBase.setproperties(s::SubsystemStates{T}, patch::NamedTuple) where {T}
    props = NamedTuple(s)
    props′ = merge(props, patch)
    if typeof(props) != typeof(props′)
        error("Type unstable change to subsystem states!")
    end
    SubsystemStates{T}(props′)
end

get_tag(::SubsystemStates{Name}) where {Name} = Name
get_tag(::Type{<:SubsystemStates{Name}}) where {Name} = Name
Base.NamedTuple(s::SubsystemStates) = getfield(s, :states)
Base.Tuple(s::SubsystemStates) = Tuple(getfield(s, :states))
Base.getproperty(s::SubsystemStates, prop::Symbol) = getproperty(NamedTuple(s), prop)
Base.propertynames(s::SubsystemStates) = propertynames(NamedTuple(s))

function state_ind(::Type{SubsystemStates{Name, Eltype, NamedTuple{names, Tup}}},
                   s::Symbol) where {Name, Eltype, names, Tup}
    i = findfirst(==(s), names)
end

function Base.convert(::Type{SubsystemStates{Name, Eltype, NT}}, s::SubsystemStates{Name}) where {Name, Eltype, NT}
    SubsystemStates{Name}(convert(NT, NamedTuple(s)))
end
function Base.convert(::Type{SubsystemStates{Name, Eltype}},
                      s::SubsystemStates{Name, <:Any, <:NamedTuple{state_names}}) where {Name, Eltype, state_names}
    nt = NamedTuple{state_names}(convert.(Eltype, Tuple(s)))
    SubsystemStates{Name, Eltype, typeof(nt)}(nt)
end

#------------------------------------------------------------
# Subsystem
function Subsystem{T}(;states, params) where {T}
    Subsystem{T}(SubsystemStates{T}(states), SubsystemParams{T}(params))
end
function Subsystem{T}(states::SubsystemStates{T, Eltype, States},
                      params::SubsystemParams{T, Params}) where {T, Eltype, States, Params}
    Subsystem{T, Eltype, States, Params}(states, params)
end

function Base.show(io::IO, sys::Subsystem{Name, Eltype}) where {Name, Eltype}
    print(io,
          "$Subsystem{$Name, $Eltype}(states = ",
          NamedTuple(get_states(sys)),
          ", params = ",
          NamedTuple(get_params(sys)),
          ")")
end

function ConstructionBase.getproperties(s::Subsystem)
    states = NamedTuple(get_states(s))
    params = NamedTuple(get_params(s))
    merge(states, params)
end
function ConstructionBase.setproperties(s::Subsystem{T, Eltype, States, Params}, patch::NamedTuple) where {T, Eltype, States, Params}
    states = NamedTuple(get_states(s))
    params = NamedTuple(get_params(s))
    props = merge(states, params)
    props′ = merge(props, patch)
    states′ = NamedTuple{keys(states)}(props′)
    params′ = NamedTuple{keys(params)}(props′)
    
    Subsystem{T, Eltype, States, Params}(SubsystemStates{T}(states′), SubsystemParams{T}(params′))
end

function Base.convert(::Type{Subsystem{Name, Eltype, SNT, PNT}}, s::Subsystem{Name}) where {Name, Eltype, SNT, PNT}
    Subsystem{Name}(convert(SubsystemStates{Name, Eltype, SNT}, get_states(s)),
                    convert(SubsystemParams{Name, PNT}, get_params(s)))
end
function Base.convert(::Type{Subsystem{Name, Eltype}}, s::Subsystem{Name}) where {Name, Eltype}
    Subsystem{Name}(convert(SubsystemStates{Name, Eltype}, get_states(s)), get_params(s))
end

@generated function promote_nt_type(::Type{NamedTuple{names, T1}},
                                    ::Type{NamedTuple{names, T2}}) where {names, T1, T2}
    NamedTuple{names, Tuple{(promote_type(T1.parameters[i], T2.parameters[i]) for i ∈ eachindex(names))...}}
end

function Base.promote_rule(::Type{SubsystemParams{Name, NT1}},
                           ::Type{SubsystemParams{Name, NT2}}) where {Name, NT1, NT2}
    SubsystemParams{Name, promote_nt_type(NT1, NT2)}
end
function Base.promote_rule(::Type{SubsystemStates{Name, ET1, NT1}},
                           ::Type{SubsystemStates{Name, ET2, NT2}}) where {Name, ET1, ET2, NT1, NT2}
    SubsystemStates{Name, promote_type(ET1, ET2), promote_nt_type(NT1, NT2)}
end

function Base.promote_rule(::Type{Subsystem{Name, ET1, SNT1, PNT1}},
                           ::Type{Subsystem{Name, ET2, SNT2, PNT2}}) where {Name, ET1, SNT1, PNT1, ET2, SNT2, PNT2}
    Subsystem{Name, promote_type(ET1, ET2), promote_nt_type(SNT1, SNT2), promote_nt_type(PNT1, PNT2)}
end

get_states(s::Subsystem) = getfield(s, :states)
get_params(s::Subsystem) = getfield(s, :params)
get_tag(::Subsystem{Name}) where {Name} = Name



get_tag(::Type{<:Subsystem{Name}}) where {Name} = Name


function Base.getproperty(s::Subsystem{<:Any, States, Params},
                          prop::Symbol) where {States, Params}
    states = NamedTuple(get_states(s))
    params = NamedTuple(get_params(s))
    if prop ∈ keys(states)
        getproperty(states, prop)
    elseif prop ∈ keys(params)
        getproperty(params, prop)
    else
        subsystem_prop_err(s, prop)
    end
end
@noinline subsystem_prop_err(s::Subsystem{Name}, prop) where {Name} = error(ArgumentError(
    "property $(prop) of ::Subsystem{$Name} not found, valid properties are $(propertynames(merge(NamedTuple(get_states(s)), NamedTuple(get_params(s)))))"
))

Base.eltype(::Subsystem{<:Any, T}) where {T} = T
Base.eltype(::Type{<:Subsystem{<:Any, T}}) where {T} = T

#-------------------------------------------------------------------------

struct VectorOfSubsystemStates{States, Mat <: AbstractMatrix} <: AbstractVector{States}
    data::Mat
end
VectorOfSubsystemStates{States}(v::Mat) where {States, Mat} = VectorOfSubsystemStates{States, Mat}(v)

Base.size(v::VectorOfSubsystemStates{States}) where {States} = (size(v.data, 2),)

@propagate_inbounds function Base.getindex(v::VectorOfSubsystemStates{States}, idx::Integer) where {States <: SubsystemStates}
    l = length(States)
    #@boundscheck checkbounds(v.data, 1:l, idx)
    @inbounds States(view(v.data, 1:l, idx))
end
@propagate_inbounds function Base.getindex(v::VectorOfSubsystemStates{States}, s::Symbol, idx::Integer) where {States <: SubsystemStates}
    i = state_ind(States, s)
    if isnothing(i)
        error("Something helpful")
    end
    v.data[i, idx]
end

@propagate_inbounds function Base.setindex!(v::VectorOfSubsystemStates{States}, state::States, idx::Integer) where {States <: SubsystemStates}
    l = length(States)
    @boundscheck checkbounds(v.data, 1:l, idx)
    @inbounds v.data[1:l, idx] .= Tuple(state)
    v
end

@propagate_inbounds function Base.setindex!(v::VectorOfSubsystemStates{States},
                                            val,
                                            s::Symbol,
                                            idx::Integer) where {States <: SubsystemStates}
    i = state_ind(States, s)
    if isnothing(i)
        error("Something helpful")
    end
    v.data[i, idx] = val
end

@generated function to_vec_o_states(state_data::NTuple{Len, Any}, ::Val{StateTypes}) where {Len, StateTypes}
    state_types = StateTypes.parameters
    Expr(:tuple, (:(VectorOfSubsystemStates{$(state_types[i])}(state_data[$i])) for i ∈ 1:Len)...)
end
