using Plots
using BenchmarkTools

function calc_attractor!(out, r, x₀; warmup=400)
    num_attract = length(out)
    # first do warmup then write each step to `out`
    x = x₀
    for i in 1:warmup
        x = r * x * (1-x)
    end
    for i in 1:num_attract
        x = r * x * (1-x)
        out[i] = x
    end
end

## serial code
function serial(r, out_matrix)
    for j in 1:length(r)
        calc_attractor!(@view(out_matrix[:,j]), r[j], 0.25)
    end
end

r = collect(2.9:0.001:4)
out_matrix = Array{Float64}(undef, 150, length(2.9:0.001:4))

serial(r, out_matrix)
scatter(r, out_matrix', legend=false, markersize=1, zcolor=0)
@btime serial(r, out_matrix)

## multithreading
using Base.Threads
nthreads()
function multithreading(r, out_matrix)
    Threads.@threads for j in 1:length(r)
        calc_attractor!(@view(out_matrix[:,j]), r[j], 0.25)
    end
end
r = collect(2.9:0.001:4)
out_matrix = Array{Float64}(undef, 150, length(2.9:0.001:4))
multithreading(r, out_matrix)
scatter(r, out_matrix', legend=false, markersize=1, zcolor=0)
@btime multithreading(r, out_matrix)

## multiprocess
using Distributed
println(workers())

if nworkers()==1
  addprocs(5)  # Unlike threads you can addprocs in the middle of a julia session
  println(workers())
end

@everywhere function calc_attractor2(r, x₀; warmup=400)
    out = Array{Float64}(undef, 150)
    num_attract = length(out)
    # first do warmup then write each step to `out`
    x = x₀
    for i in 1:warmup
        x = r * x * (1-x)
    end
    for i in 1:num_attract
        x = r * x * (1-x)
        out[i] = x
    end
    return out
end

function multiprocess(r)
    return @distributed hcat for i in r
        calc_attractor2(i, 0.25)
    end 
end

r = 2.9:0.001:4
out_matrix_multiprocess = multiprocess(r)
scatter(r, out_matrix_multiprocess', legend=false, markersize=1, zcolor=0)
@btime multiprocess(r)