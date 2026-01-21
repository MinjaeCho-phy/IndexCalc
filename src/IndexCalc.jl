module IndexCalc

include("Utils.jl")
include("Types.jl")
include("Operations.jl")
include("Parser.jl")

using .Types
using .Operations
using .Parser

export Metric, Tensor
export raise_index, lower_index
export parse_latex_and_execute, expand_latex

end # module IndexCalc
