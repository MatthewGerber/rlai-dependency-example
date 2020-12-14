from numpy.random import RandomState
from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import Gridworld
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from rlai.rewards import Reward


def main():

    random = RandomState(12345)
    gridworld = Gridworld.example_4_1(random)

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

    mdp_agent = StochasticMdpAgent(
        name='agent',
        random_state=random,
        continuous_state_discretization_resolution=None,
        gamma=1.0
    )

    mdp_agent.initialize_equiprobable_policy(gridworld.SS)

    # iterate the agents policy using q-learning temporal differencing
    iterate_value_q_pi(
        agent=mdp_agent,
        environment=gridworld,
        num_improvements=20,
        num_episodes_per_improvement=100,
        alpha=None,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        epsilon=0.1,
        planning_environment=None,
        make_final_policy_greedy=True,
        num_improvements_per_plot=20
    )

    for s in mdp_agent.pi:
        print(f'State {s.i}:')
        for a in mdp_agent.pi[s]:
            if mdp_agent.pi[s][a] > 0.0:
                print(f'\tPr({a.name}):  {mdp_agent.pi[s][a]}')


if __name__ == '__main__':
    main()
