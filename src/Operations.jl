module Operations

using ..Types
using LinearAlgebra

export raise_index, lower_index

"""
    raise_index(T::Tensor, index_pos::Int)

Raises the index at position `index_pos` (1-based) using the metric.
Returns a new Tensor.
"""
function raise_index(T::Tensor, index_pos::Int)
    # Check if index is already up
    if T.indices[index_pos] == '+'
        return T # No op
    end

    # Implementation: T^...a... = g^ab * T_...b...
    # We contract the `index_pos`-th index of a T with one index of inverse metric.
    # For simplicity in this basic version, we'll perform a tensordot-like operation.
    
    # 1. Permute the target index to the end for easier multiplication
    rank = ndims(T.data)
    perm = [i for i in 1:rank if i != index_pos]
    push!(perm, index_pos)
    
    T_permuted = permutedims(T.data, perm)
    
    # 2. Reshape to matrix for multiplication: (rest_indices) x (target_index)
    rest_dims = size(T_permuted)[1:end-1]
    last_dim = size(T_permuted)[end]
    
    flat_data = reshape(T_permuted, prod(rest_dims), last_dim)
    
    # 3. Multiply with Inverse Metric:  Flat * g_inv
    # g_inv is (d x d). We need matches. 
    # If the index is DOWN (-), we contract with UP-UP metric (inverse).
    # Resulting index is UP.
    
    new_data_flat = flat_data * T.metric.inverse
    
    # 4. Reshape back
    new_data_permuted = reshape(new_data_flat, rest_dims..., last_dim)
    
    # 5. Permute back to original order
    # Current indices order is [1..pos-1, pos+1..rank, pos_new]
    # We want [1..pos, ..]
    # So we need the inverse permutation of step 1.
    inv_perm = sortperm(perm)
    
    final_data = permutedims(new_data_permuted, inv_perm)
    
    # Update indices string
    const_indices = collect(T.indices)
    const_indices[index_pos] = '+'
    new_indices = String(const_indices)
    
    return Tensor(final_data, T.name, new_indices, T.metric)
end

"""
    lower_index(T::Tensor, index_pos::Int)

Lowers the index at position `index_pos` (1-based) using the metric.
Returns a new Tensor.
"""
function lower_index(T::Tensor, index_pos::Int)
    # Check if index is already down
    if T.indices[index_pos] == '-'
        return T # No op
    end
    
    # Implementation: T_...a... = g_ab * T^...b...
    # Metric (g_ab) is symmetric, so order doesn't strictly matter for contraction 
    # but we typically do g_ab * vector^b
    
    rank = ndims(T.data)
    perm = [i for i in 1:rank if i != index_pos]
    push!(perm, index_pos)
    
    T_permuted = permutedims(T.data, perm)
    rest_dims = size(T_permuted)[1:end-1]
    last_dim = size(T_permuted)[end]
    
    flat_data = reshape(T_permuted, prod(rest_dims), last_dim)
    
    # Multiply with Metric: Flat * g
    new_data_flat = flat_data * T.metric.matrix
    
    new_data_permuted = reshape(new_data_flat, rest_dims..., last_dim)
    inv_perm = sortperm(perm)
    final_data = permutedims(new_data_permuted, inv_perm)
    
    const_indices = collect(T.indices)
    const_indices[index_pos] = '-'
    new_indices = String(const_indices)
    
    return Tensor(final_data, T.name, new_indices, T.metric)
end

end # module Operations
