from qtable import fixed
from envs.economy import jesusfv
from policy import diagnostics

env = jesusfv

# verifies that at the optimal the euler rule is satisfied
qtable = fixed.FixedQTable(env.alpha, .95, env.delta)
print(diagnostics.euler_error(qtable, env, 500))
