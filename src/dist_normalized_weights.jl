export normalized_influence_dist

function normalized_influence_dist(rng, influence_matrix, n_mc)
    n_f = size(influence_matrix, 1)
    n_t = size(influence_matrix, 2)
    normalized_samples = zeros(n_mc, n_f, n_t)

    for s in 1:n_mc
        for j in 1:n_t
            raw = [rand(rng, influence_matrix[i, j]) for i in 1:n_f]
            total = sum(raw)
            normalized_samples[s, :, j] = raw ./ total
        end
    end

    return normalized_samples
end