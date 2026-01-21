# Direct include verification script
using Test
using LinearAlgebra

# Direct include to bypass Package Manager precompile locking issues in scratch env
include("../src/Utils.jl")
include("../src/Types.jl")
include("../src/Operations.jl")
include("../src/Parser.jl")
# IndexCalc just re-exports, we can skip it or manually use modules.

using .Types
using .Operations
using .Parser

@testset "Direct Include Verification" begin
    # 1. Define Metric
    g_matrix = [1.0 0.0; 0.0 -1.0]
    g = Metric(g_matrix, "g")
    
    # 2. Define Tensor
    data = [1.0 2.0; 3.0 4.0]
    T = Tensor(data, "T", "+-", g)
    
    # 3. Raise/Lower
    T_up = raise_index(T, 2)
    @test T_up.indices == "++"
    @test T_up.data ≈ [1.0 -2.0; 3.0 -4.0]
    
    # 4. Symbolic Expansion
    # Define J tensor
    data_J = [1.0, 2.0, 3.0, 4.0]
    J = Tensor(data_J, "J", "+", g)
    
    tensor_map = Dict{String, Tensor}()
    tensor_map["T"] = T
    tensor_map["J"] = J
    
    expr = "\\T^{a}\\,_{b} ; \\J^{b}"
    expanded = expand_latex(tensor_map, expr, 4)
    
    println("Expanded Output: ", expanded)
    
    @test contains(expanded, "T^{1}_{1} ; J^{1}")
    @test contains(expanded, "T^{1}_{4} ; J^{4}")
end
