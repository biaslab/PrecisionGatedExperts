.PHONY: format

scripts_init:
	julia --startup-file=no --project=linter/ -e 'using Pkg; Pkg.instantiate(); Pkg.update(); Pkg.precompile();'

format: scripts_init ## Code formating run
	julia --startup-file=no --project=linter/ linter/format.jl --overwrite