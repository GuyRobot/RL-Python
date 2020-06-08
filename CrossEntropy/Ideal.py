"""
Cross entropy:
    - H(X): reward value
Estimation:
    l = Ef[H(X)] = Z(H(x)f(x)dx) (Z is integral)

The idea of the CE method is to choose the importance sampling
density g in a specified class of densities such that the cross-entropy or KullbackLeibler divergence between the optimal importance sampling density g
∗ and g
is minimal

Kullback-Leibler
    D(g, h) = Eg[ln g(X)h(X)]
            = Z (g(x) * ln g(x) / h(x) dx)
            = Z (g(x) * ln g(x) dx) − Z(g(x) * lnh(x)dx) .

iterative algorithm, which starts
with and on every step improves. This is an approximation of
p(x)H(x) with an update:
    q_(i+1) = argmin Eg[p_i(x)/q_i(x) * H(x)]log(q_(i+1)(x))

The method is quite clear: we sample
episodes using our current policy (starting with some random initial policy)
and minimize the negative log likelihood of the most successful samples and
our policy
"""