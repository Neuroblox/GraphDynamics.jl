valueof(::Val{X}) where {X} = X
valueof(x) = x

# this just makes it so that I can easily replace all uses of `@inbounds ex` with just `ex`.
macro inbounds(ex)
    #esc(ex)
    esc(:($Base.@inbounds $ex))
end

macro unroll(N::Int, loop)
    Base.isexpr(loop, :for) || error("only works on for loops")
    Base.isexpr(loop.args[1], :(=)) || error("This loop pattern isn't supported")
    val, itr = esc.(loop.args[1].args)
    body = esc(loop.args[2])
    @gensym loopend
    label = :(@label $loopend)
    goto = :(@goto $loopend)
    out = Expr(:block, :(itr = $itr), :(next = iterate(itr)))
    unrolled = map(1:N) do _
        quote
            isnothing(next) && @goto loopend
            $val, state = next
            $body
            next = iterate(itr, state)
        end
    end
    append!(out.args, unrolled)
    remainder = quote
        while !isnothing(next)
            $val, state = next
            $body
            next = iterate(itr, state)
        end
        @label loopend
    end
    push!(out.args, remainder)
    out
end
macro unroll2(N::Int, loop)
    Base.isexpr(loop, :for) || error("only works on for loops")
    Base.isexpr(loop.args[1], :(=)) || error("This loop pattern isn't supported")
    val, itr = esc.(loop.args[1].args)
    body = esc(loop.args[2])
    label = :(@label loopend)
    goto = :(@goto loopend)
    out = Expr(:block, :(itr = $itr), :(next = iterate(itr)))
    unrolled = map(1:N) do _
        quote
            $val, state = next
            $body
            next = iterate(itr, state)
        end
    end
    append!(out.args, unrolled)
    remainder = quote
        while !isnothing(next)
            $val, state = next
            $body
            next = iterate(itr, state)
        end
        @label loopend
    end
    push!(out.args, remainder)
    quote
        if length($itr) >= N
            $out
        else
            @unroll $N $loop
        end
    end
    out
end
