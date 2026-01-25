#------------------------------------------------------------
# Subsystem parameters
function SubsystemParams{Name}(nt::NT) where {Name, NT <: NamedTuple}
    SubsystemParams{Name, NT}(nt)
end
SubsystemParams{Name}(;kwargs...) where {Name} = SubsystemParams{Name}(NamedTuple(kwargs))

function Base.show(io::IO, params::SubsystemParams{Name}) where {Name}
    print(io, "$SubsystemParams{$Name}(", join(("$k = $v" for (k, v) ∈ pairs(NamedTuple(params))), ", "), ")")
end

function ConstructionBase.getproperties(s::SubsystemParams)
    NamedTuple(s)
end

function ConstructionBase.setproperties(s::SubsystemParams{T}, patch::NamedTuple) where {T}
    set_param_prop(s, patch; allow_typechange=false)
end
function set_param_prop(s::SubsystemParams{T}, key, val; allow_typechange=false) where {T}
    set_param_prop(s, NamedTuple{(key,)}((val,)); allow_typechange)
end
function set_param_prop(s::SubsystemParams{T}, patch; allow_typechange=false) where {T}
    props = NamedTuple(s)
    props′ = merge(props, patch)
    if typeof(props) != typeof(props′) && !allow_typechange
        props′ = convert(typeof(props), props′)
    end
    SubsystemParams{T}(props′)
end

@noinline function param_setproperror(props, props′)
    error("Type unstable change to subsystem params! Expected properties of type\n  $(typeof(props))\nbut got\n  $(typeof(props′))")
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
@generated function promote_numeric_param_eltype(::Type{SubsystemParams{Name, NamedTuple{props, Tup}}}) where {Name, props, Tup}
    :(promote_type($(param for param in Tup.parameters if param <: Number)...))
end

function Base.length(
    ::Type{SubsystemParams{Name, NamedTuple{names, Tup}}}
    ) where {Name, names, Tup}
    length(names)
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
SubsystemStates{Name}(;kwargs...) where {Name} = SubsystemStates{Name}(NamedTuple(kwargs))

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
    # TODO: should there be observed states in here? I don't want to accidentally waste CPU cycles on them
    # but it might be necessary? Not sure, since they can't be `setproperty`-d
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
    proms = [:(promote_type($(T1.parameters[i]), $(T2.parameters[i]))) for i in eachindex(names)]
    :(NamedTuple{names, Tuple{$(proms...)}})
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

function Base.getproperty(s::Subsystem{Name, States, Params},
                          prop::Symbol) where {Name, States, Params}
    states = NamedTuple(get_states(s))
    params = NamedTuple(get_params(s))
    if prop ∈ keys(states)
        getproperty(states, prop)
    elseif prop ∈ keys(params)
        getproperty(params, prop)
    else
        comp_props = computed_properties(Name)
        if prop ∈ keys(comp_props)
            comp_props[prop](s)
        else
            subsystem_prop_err(s, prop)
        end
    end
end
@noinline subsystem_prop_err(s::Subsystem{Name}, prop) where {Name} = error(ArgumentError(
    "property $(prop) of ::Subsystem{$Name} not found, valid properties are $(propertynames(merge(NamedTuple(get_states(s)), NamedTuple(get_params(s)), computed_properties(Name))))"
))

Base.eltype(::Subsystem{<:Any, T}) where {T} = T
Base.eltype(::Type{<:Subsystem{<:Any, T}}) where {T} = T

#-------------------------------------------------------------------------
_deval(::Val{T}) where {T} = T
function partitioned(v, partition_plan::NTuple{N, Any}) where {N}
    map(partition_plan) do (;inds, sz, TVal)
        # to_structarray(_deval(TVal), v, inds, sz[2])
        M = reshape(view(v, inds), sz...)
        ArrayOfSubsystemStates{_deval(TVal)}(M)
    end
end

struct ArrayOfSubsystemStates{States, N, Store <: StridedArray} <: DenseArray{States, N}
    parent::Store
    function ArrayOfSubsystemStates{SubsystemStates{Name, T, NamedTuple{snames, Tup}}}(v::StridedArray{U, M}) where {Name, T, U, M, snames, Tup}
        @assert size(v,1) == length(snames)
        V = promote_type(T,U)
        States = SubsystemStates{Name, V, NamedTuple{snames, NTuple{length(snames), V}}}
        new{States, M-1, typeof(v)}(v)
    end
end
const VectorOfSubsystemStates{States, Store} = ArrayOfSubsystemStates{States, 1, Store}
Base.size(v::ArrayOfSubsystemStates{States}) where {States} = size(parent(v))[2:end]
Base.parent(v::ArrayOfSubsystemStates) = getfield(v, :parent)
Base.IndexStyle(::Type{<:ArrayOfSubsystemStates}) = IndexCartesian()
Base.pointer(v::ArrayOfSubsystemStates) = pointer(parent(v))
function Base.elsize(::Type{ArrayOfSubsystemStates{States, N, Store}}) where {States, N, Store}
    sizeof(States)
end

@propagate_inbounds function Base.getindex(v::ArrayOfSubsystemStates{States}, idx::Integer...) where {States <: SubsystemStates}
    l = length(States)
    data = parent(v)
    @boundscheck checkbounds(data, 1:l, idx...)
    @inbounds States(view(data, 1:l, idx...))
end
@propagate_inbounds function Base.setindex!(v::ArrayOfSubsystemStates{States}, state::States′, idx::Integer...) where {States <: SubsystemStates, States′ <: SubsystemStates}
    l = length(States)
    data = parent(v)
    @boundscheck checkbounds(data, 1:l, idx...)
    @inbounds data[1:l, idx...] .= Tuple(convert(States, state))
    v
end


Base.IndexStyle(::Type{<:VectorOfSubsystemStates}) = IndexLinear()

@propagate_inbounds function Base.getindex(v::VectorOfSubsystemStates{States}, idx::Integer) where {States <: SubsystemStates}
    l = length(States)
    data = parent(v)
    # @boundscheck checkbounds(data, :, idx)
    # @inbounds 
    States(view(data, 1:l, idx))
end
@propagate_inbounds function Base.setindex!(v::VectorOfSubsystemStates{States}, state::States, idx::Integer) where {States <: SubsystemStates}
    l = length(States)
    data = parent(v)
    # @boundscheck checkbounds(data, :, idx)
    # @inbounds 
    data[1:l, idx] .= Tuple(state)
    v
end

function Base.getproperty(v::ArrayOfSubsystemStates, prop::Symbol)
    FieldView{prop}(v)
end
@propagate_inbounds function Base.view(v::ArrayOfSubsystemStates{States}, inds...) where {States}
    l = length(States)
    ArrayOfSubsystemStates{States}(view(parent(v), :, inds...))
end

#-------------------------------------------------------------------------

struct ArrayOfSubsystems{T, N, Subsys<:Subsystem{T}, StateStore <:AbstractArray{<:SubsystemStates, N}, ParamStore <: AbstractArray{<:SubsystemParams, N}} <: AbstractArray{Subsys, N}
    states::StateStore
    params::ParamStore
    function ArrayOfSubsystems(vstates::AbstractArray{SubsystemStates{T, Elt, SNT}, N},
                               vparams::AbstractArray{SubsystemParams{T, PNT}, N}
                               ) where {T, Elt, N, SNT, PNT}
        @assert size(vstates) == size(vparams)
        new{T, N, Subsystem{T, Elt, SNT, PNT}, typeof(vstates), typeof(vparams)}(vstates, vparams)
    end
end
const VectorOfSubsystems{States, Store} = ArrayOfSubsystems{States, 1, Store}
Base.size(v::ArrayOfSubsystems) = size(getfield(v, :states))
Base.IndexStyle(::Type{<:ArrayOfSubsystems}) = IndexLinear()
get_states(x::ArrayOfSubsystems) = getfield(x, :states)
get_params(x::ArrayOfSubsystems) = getfield(x, :params)


@propagate_inbounds function Base.getindex(v::ArrayOfSubsystems, idx::Integer)
    vstates = getfield(v, :states)
    vparams = getfield(v, :params)
    @boundscheck checkbounds(vstates, idx)
    states = @inbounds vstates[idx]
    params = @inbounds vparams[idx]
    Subsystem(states, params)
end

@propagate_inbounds function Base.setindex!(v::ArrayOfSubsystems{T, N, Subsys}, sys::Subsystem, idx::Integer) where {T, N, Subsys}
    vstates = getfield(v, :states)
    vparams = getfield(v, :params)
    @boundscheck checkbounds(vstates, idx)
    states = @inbounds vstates[idx] = get_states(sys)
    params = @inbounds vparams[idx] = get_params(sys)
    v
end

function Base.getproperty(v::ArrayOfSubsystems{T, N, Subsystem{T, Elt, SNT, PNT}}, prop::Symbol) where {T, N, Elt, SNT, PNT}
    if hasfield(SNT, prop)
        FieldView{prop}(getfield(v, :states))
    elseif hasfield(PNT, prop)
        FieldView{prop}(getfield(v, :params))
    else
        @noinline errf(T, prop) = error("Type $T has no property $prop")
        errf(eltype(v), prop)
    end
end

function Base.view(v::ArrayOfSubsystems, args...)
    vstates = view(getfield(v, :states), args...)
    vparams = view(getfield(v, :params), args...)
    ArrayOfSubsystems(vstates, vparams)
end

get_parent_index(x::SubArray{T, 0}) where {T} = only(x.indices)
get_parent_index(x::ArrayOfSubsystems{T, 0}) where {T} = get_parent_index(get_params(x))
