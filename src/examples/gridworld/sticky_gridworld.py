from numpy.random import RandomState
from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.gridworld import Gridworld
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from rlai.q_S_A.tabular import TabularStateActionValueEstimator
from rlai.rewards import Reward


def main():

    random = RandomState(12345)
    gridworld = Gridworld.example_4_1(random, None)

    # the bottom-right corner (3,3) is a goal state. get the states surrounding this goal. these will become the sticky
    # states.
    sticky_states = [
        gridworld.grid[2, 2],
        gridworld.grid[2, 3],
        gridworld.grid[3, 2]
    ]

    # amplify all negative rewards in the sticky states by a factor of 10, keeping the probabilities the same.
    for sticky_state in sticky_states:
        for a in gridworld.p_S_prime_R_given_S_A[sticky_state]:
            for s_prime in gridworld.p_S_prime_R_given_S_A[sticky_state][a]:
                gridworld.p_S_prime_R_given_S_A[sticky_state][a][s_prime] = {
                    Reward(r.i, (r.r * 10.0 if r.r < 0.0 else r.r)):  gridworld.p_S_prime_R_given_S_A[sticky_state][a][s_prime][r]
                    for r in gridworld.p_S_prime_R_given_S_A[sticky_state][a][s_prime]
                }

    epsilon = 0.1

    q_S_A = TabularStateActionValueEstimator(
        environment=gridworld,
        epsilon=epsilon,
        continuous_state_discretization_resolution=None
    )

    pi = q_S_A.get_initial_policy()

    mdp_agent = StochasticMdpAgent(
        name='agent',
        random_state=random,
        pi=pi,
        gamma=1.0
    )

    # iterate the agents policy using q-learning temporal differencing
    iterate_value_q_pi(
        agent=mdp_agent,
        environment=gridworld,
        num_improvements=20,
        num_episodes_per_improvement=100,
        num_updates_per_improvement=None,
        alpha=None,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        planning_environment=None,
        make_final_policy_greedy=True,
        q_S_A=q_S_A,
        num_improvements_per_plot=20
    )

    for s in pi:
        print(f'State {s.i}:')
        for a in pi[s]:
            if pi[s][a] > 0.0:
                print(f'\tPr({a.name}):  {pi[s][a]}')


if __name__ == '__main__':
    main()
