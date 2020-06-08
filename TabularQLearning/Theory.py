"""
    We can use states obtained from the
    environment to update values of states, which can save us lots of work.
    Step:
        1. Start with an empty table, mapping states to values of actions.
        2. By interacting with the environment, obtain the tuple s, a, r, s′ (state,
            action, reward, and the new state). In this step, we need to decide which
            action to take, and there is no single proper way to make this decision.
            We discussed this problem as exploration versus exploitation and will
            talk a lot about this.
        3. Update the Q(s, a) value using the Bellman approximation:
            Q_s,a <- r + y * max_a'~A(Q_s',a')
        4. Repeat from step 2.

    Using blending update Q:
        Q_s,a = (1 - a)Q_s,a + a(r + y*max_a'~A(Q_s',a')
    Final Algorithm (Tabular QL):
        1. Start with an empty table for Q(s, a).
        2. Obtain (s, a, r, s′) from the environment.
        3. Make a Bellman update:
            Q_s,a = (1 - a)Q_s,a + a(r + y*max_a'~A(Q_s',a')
        4. Check convergence conditions. If not met, repeat from step 2.

"""