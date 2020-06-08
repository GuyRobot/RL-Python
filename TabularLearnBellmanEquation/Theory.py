"""
Bellman equation:
    V(a) = Es~S[r_(s,a) + yV_s] = sum_s~S(r_s,a + yV_s)
    V = max[Es~S[r_(s,a) + yV_s]] = max[sum_s~S(r_s,a + yV_s)]
    Note p_a,i->j that means the probability of action a, issued in state i, to end up
    in state j

Q-learning:
    Q_s,a = Es'~S[r_s,a + yV_s']
          = sum_s~S[p_a,s->s`'(r_s,a + yV_s')
    V_s = max_a~A(Q_s,a)
    Q(s,a) = r_s,a + ymax_a'~A[Q(s',a')]

    Example: see in images/Q_learning_example
        Q(s0, up)   = 0.33V1 + 0.33V2 + 0.33V4
                    = 0.33 * 1 + 0.33 * 2 + 0.33 * 4
                    = 2.31
        Q(s0, left) = 0.33V1 + 0.33V2 + 0.33V3
                    = 1.98
        Q(s0, up)   = 0.33V4 + 0.33V1 + 0.33V3
                    = 2.64
        Q(s0, up)   = 0.33V2 + 0.33V3 + 0.33V4
                    = 2.97
Bellman update iterative:
    1Initialize values of all states to some initial value (usually zero)
    2. For every state s in the MDP, perform the Bellman update:
        V_s <- max_a sum_s'[p_a,s->s'(r_s,a + yV_s')]
    3. Repeat step 2 for some large number of steps or until changes become
    too small

Q_learning update iterative:
    1. Initialize all to zero
    2. For every state s and every action a in this state, perform update:
        Q_s,a <- sum_s'[p_a,s->s'(r_s,a + y * max(x_a')Q_s',a')]
    3. Repeat step 2

"""