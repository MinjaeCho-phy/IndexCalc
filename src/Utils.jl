module Utils

using LinearAlgebra

export check_symmetric

"""
    check_symmetric(matrix::AbstractMatrix)

Checks if the given matrix is symmetric. 
Returns `true` if symmetric, `false` otherwise.
"""
function check_symmetric(matrix::AbstractMatrix)
    return isapprox(matrix, transpose(matrix))
end

end # module Utils
