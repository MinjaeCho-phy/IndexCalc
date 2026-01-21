module Types

using LinearAlgebra
using ..Utils

export Metric, Tensor

"""
    Metric

A structure representing a metric tensor.
- `matrix`: The matrix representation of the metric (g_{ab}).
- `name`: Name of the metric (e.g., "g").
- `inverse`: The inverse matrix (g^{ab}).
"""
struct Metric
    matrix::Matrix{Float64}
    name::String
    inverse::Matrix{Float64}

    function Metric(matrix::Matrix{Float64}, name::String="g")
        if !check_symmetric(matrix)
            @warn "Metric '$name' is not symmetric!"
        end
        inverse = inv(matrix)
        new(matrix, name, inverse)
    end
end

"""
    Tensor

A structure representing a tensor.
- `data`: The multidimensional array holding tensor components.
- `name`: Name of the tensor (e.g., "T").
- `indices`: String representation of indices (e.g., "++-" for T^{ab}_c).
             '+' denotes upper index, '-' denotes lower index.
- `metric`: The metric associated with this tensor.
"""
struct Tensor
    data::Array{Float64}
    name::String
    indices::String
    metric::Metric

    function Tensor(data::Array{Float64}, name::String, indices::String, metric::Metric)
        rank = ndims(data)
        if length(indices) != rank
            error("Rank of data ($rank) does not match length of indices string ($(length(indices))) for tensor '$name'.")
        end
        new(data, name, indices, metric)
    end
end

end # module Types
