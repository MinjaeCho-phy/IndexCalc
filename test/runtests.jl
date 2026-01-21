using IndexCalc
using Test
using LinearAlgebra

@testset "IndexCalc Tests" begin
    # 1. Define Metric
    # Minkowski-like 2D for simplicity
    g_matrix = [1.0 0.0; 0.0 -1.0]
    g = Metric(g_matrix, "g")
    
    @test g.name == "g"
    @test g.inverse == inv(g_matrix)

    # 2. Define Tensor
    # T^a_b
    data = [1.0 2.0; 3.0 4.0]
    T = Tensor(data, "T", "+-", g)
    
    @test T.indices == "+-"
    
    # 3. Raise Index
    # Raise 'b' (pos 2). T^a^b = T^a_c * g^cb
    T_up = raise_index(T, 2)
    @test T_up.indices == "++"
    
    # Manual check: [1 2; 3 4] * [1 0; 0 -1] = [1 -2; 3 -4]
    @test T_up.data ≈ [1.0 -2.0; 3.0 -4.0]
    
    # 4. Lower Index
    # Lower 'a' (pos 1). T_a_b = g_ac * T^c_b
    T_down = lower_index(T, 1)
    @test T_down.indices == "--"
    
    # Manual: [1 0; 0 -1] * [1 2; 3 4] = [1 2; -3 -4]
    @test T_down.data ≈ [1.0 2.0; -3.0 -4.0]
    
    # 5. Contraction (Parsing)
    # T^a_b ; T^b_c => Matrix Mult
    # [1 2; 3 4] * [1 2; 3 4] = [7 10; 15 22]
    
    # Expression: "\T^{a}_{b} ; \T^{b}_{c}"
    # Note: Our simple parser expects definitions in a dictionary.
    
    tensor_map = Dict("T" => T)
    result = parse_latex_and_execute(tensor_map, "\\T^{a}_{b} ; \\T^{b}_{c}")
    
    @test result.indices == "+-" # a is up, c is down
    @test result.data ≈ [7.0 10.0; 15.0 22.0]
    
    # 6. Auto-raising in Parse
    # We have T (+-). Input calls for T_{a}^{b} (-+).
    # Parser should auto-lower 1 and raise 2 before mult?
    # This is complex, but let's see if partial works.
    # Let's try matching indices locally.
    
    # Test passed if code compiles and runs logic correctly.
end
