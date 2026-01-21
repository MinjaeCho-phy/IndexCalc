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
    

    # 7. Symbolic Expansion
    # \T^{a}\,_{b} ; \J^{b} (a free, b dummy)
    # Dimension 4
    
    # Define J tensor for the map
    data_J = [1.0, 2.0, 3.0, 4.0]
    J = Tensor(data_J, "J", "+", g)
    
    tensor_map["J"] = J
    
    expr = "\\T^{a}\\,_{b} ; \\J^{b}"
    expanded = expand_latex(tensor_map, expr, 4)
    
    # Check format: "{ T^{1}_{1} ; J^{1} + ... + T^{1}_{4} ; J^{4}, 1->2, ..., 1->4 }[a]"
    # Note: Our code cleans \, and spaces. Factors order preserved.
    # Expected core: "T^{1}_{1} ; J^{1} + T^{1}_{2} ; J^{2} + T^{1}_{3} ; J^{3} + T^{1}_{4} ; J^{4}"
    
    println("Expanded: ", expanded)
    
    @test contains(expanded, "T^{1}_{1} ; J^{1}")
    @test contains(expanded, "T^{1}_{4} ; J^{4}")
    @test contains(expanded, ", 1->2, ..., 1->4")
    @test endswith(expanded, "[a]")
    
    # Test passed if code compiles and runs logic correctly.
end
