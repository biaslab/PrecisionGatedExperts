.PHONY: format

scripts_init:
	julia --startup-file=no --project=linter/ -e 'using Pkg; Pkg.instantiate(); Pkg.update(); Pkg.precompile();'

format: scripts_init ## Code formating run
	julia --startup-file=no --project=linter/ linter/format.jl --overwrite

build-results: ## Build results for all models and datasets
	julia --startup-file=no --project=src/ scripts/build_comparision_table.jl