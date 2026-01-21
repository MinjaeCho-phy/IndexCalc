module Parser

using ..Types
using ..Operations

export parse_latex_and_execute

"""
    parse_idx_string(s::String)

Parses a string like "^{a}_{b}" into a list of indices and their positions/types.
Returns a list of tuples: [('a', :up), ('b', :down)]
"""
function parse_idx_string(s::String)
    indices = []
    i = 1
    len = length(s)
    current_pos = :none
    
    while i <= len
        char = s[i]
        if char == '^'
            current_pos = :up
        elseif char == '_'
            current_pos = :down
        elseif char == '{'
            i += 1
            # Read until '}'
            start_j = i
            while i <= len && s[i] != '}'
                # Assuming single char indices for simplicity based on prompt "i1, i2..." or "a,b"
                # User prompt: "a", "b". single char.
                val = s[i] # taking the character
                # Actually, user said "i1, i2" internal, but input "a,b".
                # Let's treat everything inside {} as the index name.
                push!(indices, (String([val]), current_pos))
                i += 1
            end
        else
            # Ignore spaces, \T, etc handled outside? 
            # This function expects just the ^{...}_{...} part usually.
        end
        i += 1
    end
    return indices
end

"""
    dummy_contraction(tensor_map, expression)

Basic parser flow:
1. Split by ';' for independent terms? No, user said "; " means multiply (or separate usage). 
   Let's assume the user input is a single multiplication term: "\T^{a}_{b} ; \T^{b}_{c}"
   Wait, ";" usually separates commands. But user example: "\T^{a}_{b} ; \T^{b}_{c} ... 곱한 후"
   This implies semicolon was used as a definition of terms to multiply.
"""
function parse_latex_and_execute(tensor_map::Dict{String, Tensor}, expression::String)
    # 1. Cleaning: remove '\'
    clean_expr = replace(expression, "\\" => "")
    
    # 2. Split by ';' to get factors
    # e.g. "T^{a}_{b} ; T^{b}_{c}" -> ["T^{a}_{b}", " T^{b}_{c}"]
    factors_str = split(clean_expr, ";")
    
    tensors_to_multiply = []
    all_indices = [] # Track all indices to find dummies
    
    for factor in factors_str
        factor = strip(factor)
        # Regex to capture Name and Indices part: Name(^{...}_{...})
        # Assuming Name is alphanumeric.
        m = match(r"^([a-zA-Z0-9]+)(.*)$", factor)
        if m === nothing
            error("Invalid factor format: $factor")
        end
        
        name = m.captures[1]
        idx_str = m.captures[2]
        
        if !haskey(tensor_map, name)
            error("Tensor '$name' not defined.")
        end
        
        original_tensor = tensor_map[name]
        parsed_indices = parse_idx_string(idx_str)
        
        # Verify rank
        if length(parsed_indices) != ndims(original_tensor.data)
             error("Index count does not match rank for '$name'.")
        end
        
        # Adjust indices (Raise/Lower) to match requested 'parsed_indices'
        # If user asks for T^{a} but we have T_{a} (stored as "--" e.g.), we raise it.
        # But wait, user said: "If defined is only '--', automatically ... raise/lower"
        
        current_tensor = original_tensor
        
        # Match current state to requested state
        # We need to ensure the tensor is in the configuration requested by the string.
        # e.g. Stored: "-+" (T_a^b). Requested: "++" (T^a^b).
        # We must iterate positions and fix.
        
        for k in 1:length(parsed_indices)
            req_idx_name, req_idx_pos = parsed_indices[k]
            
            # Check current state of tensor at pos k
            current_idx_type = current_tensor.indices[k] # '+' or '-'
            
            desired_type = (req_idx_pos == :up) ? '+' : '-'
            
            if current_idx_type != desired_type
                if desired_type == '+'
                    current_tensor = raise_index(current_tensor, k)
                else
                    current_tensor = lower_index(current_tensor, k)
                end
            end
        end
        
        push!(tensors_to_multiply, (current_tensor, parsed_indices))
    end
    
    # 3. Perform Multiplication / Contraction
    # Naive approach: Contract first two, then result with next... (reduce)
    
    result_tensor = tensors_to_multiply[1][1]
    result_indices = tensors_to_multiply[1][2] # list of (name, type)
    
    # Loop over rest
    for i in 2:length(tensors_to_multiply)
        T_next = tensors_to_multiply[i][1]
        next_indices = tensors_to_multiply[i][2]
        
        # Find common index names to contract
        # We need to perform an outer product then trace/sum over common indices.
        # For simplicity in this demo, let's look for exact matching names.
        
        # Contract logic is complex to write from scratch efficiently.
        # Simplification:
        # 1. Outer Product (Tensor Product)
        # 2. Loop over indices and trace if names match.
        
        # Actually, simpler: Use `einsum` style from a library or pure Julia loop?
        # Let's do a pure loop over data (very slow but correct for demo) or basic matrix mul if rank 2.
        
        # Let's implementing a basic "Einstein Summation" over the data arrays.
        # We need to map dimension indices.
        
        # Construct full list of indices
        left_idx_names = [x[1] for x in result_indices]
        right_idx_names = [x[1] for x in next_indices]
        
        common = intersect(left_idx_names, right_idx_names)
        
        # Check validity: One up, one down?
        # User requirement: "Tensor(matrix, name="T", "++", metric=metric_name()) = inv_metric * Tensor(matrix, name="T", "-+", metric=metric_name())"
        # User implies we just grab values.
        # "T^{a}_{b} ; T^{b}_{c} ... 곱한 후 b에 대한 합"
        # Standard contraction.
        
        # We will use `tensor_contract` helper (to be implemented conceptually).
        # Since implementing full generic tensor contraction in raw Julia without dependencies like TensorOperations.jl is verbose:
        # We will assume `LinearAlgebra`.
        
        # Special Case: Matrix Multiplication T^a_b * T^b_c -> (A * B)
        if length(left_idx_names) == 2 && length(right_idx_names) == 2 && length(common) == 1
             # Detect if it is matmul structure
             # if common is col of Left and row of Right...
             # For now, let's leave a placeholder "Generic Contraction Not Fully Implemented" warning or do a simple case.
             # User asked for "General matrix calculation check".
             
             # Let's implement Matrix * Matrix explicitly correctly.
             # Pos of 'b' in Left? Pos of 'b' in Right?
             comm_idx = common[1]
             idx_L = findfirst(==(comm_idx), left_idx_names)
             idx_R = findfirst(==(comm_idx), right_idx_names)
             
             # Perform contraction on these axes.
             # Slice and multiply.
             
             # New Data
             # Using a helper for contraction would be best.
             result_data = _simple_contract(result_tensor.data, T_next.data, idx_L, idx_R)
             
             # Update Indices list
             new_indices_list = filter(x->x[1]!=comm_idx, result_indices)
             append!(new_indices_list, filter(x->x[1]!=comm_idx, next_indices))
             
             # Construct new Tensor struct
             new_idx_str = "" # reconstruct string?? "+-"
             # Correct string reconstruction is tricky without tracking types of survivors.
             # We take types from survivors.
             
             new_str = ""
             for item in new_indices_list
                 new_str *= (item[2] == :up ? "+" : "-")
             end
             
             result_tensor = Tensor(result_data, "Result", new_str, result_tensor.metric)
             result_indices = new_indices_list
        else
            error("Only rank-2 single-index contraction is currently supported in this prototype.")
        end
    end
    
    return result_tensor
end

function _simple_contract(A::AbstractArray, B::AbstractArray, dimA::Int, dimB::Int)
    # Contracts dimA of A with dimB of B
    # A has dims (..., dA, ...)
    # B has dims (..., dB, ...)
    # dA == dB
    
    # Move contraction axes to the end of A and start of B
    rankA = ndims(A)
    permA = [i for i in 1:rankA if i != dimA]
    push!(permA, dimA)
    
    rankB = ndims(B)
    permB = [dimB]
    append!(permB, [i for i in 1:rankB if i != dimB])
    
    A_p = permutedims(A, permA)
    B_p = permutedims(B, permB)
    
    # Reshape A to (Rows_A, Common)
    # Reshape B to (Common, Cols_B)
    
    common_dim = size(A, dimA)
    
    rows_A = prod(size(A)) ÷ common_dim
    cols_B = prod(size(B)) ÷ common_dim
    
    MatA = reshape(A_p, rows_A, common_dim)
    MatB = reshape(B_p, common_dim, cols_B)
    
    Result = MatA * MatB
    
    # Reshape result back
    # Shape: (Indices A..., Indices B...) (excluding common)
    new_shape = (size(A_p)[1:end-1]..., size(B_p)[2:end]...)
    
    return reshape(Result, new_shape)
end


end # module Parser
