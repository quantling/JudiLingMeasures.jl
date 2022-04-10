using Documenter
using JudiLingMeasures

makedocs(
    sitename = "JudiLingMeasures.jl",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Introduction" => "index.md",
        "Measures" => "measures.md",
        "Helper Functions" => "helpers.md"
    ]
)

deploydocs(
    repo = "github.com/MariaHei/JudiLingMeasures.jl.git",
    devbranch = "main"
)
