from numpy.random import RandomState
from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.openai_gym import Gym, CartpoleFeatureExtractor
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from rlai.q_S_A.function_approximation.estimators import ApproximateStateActionValueEstimator
from rlai.q_S_A.function_approximation.models.sklearn import SKLearnSGD


def main():

    random_state = RandomState(12345)

    environment = Gym(
        random_state=random_state,
        T=None,
        gym_id='CartPole-v1',
        continuous_action_discretization_resolution=None,
        render_every_nth_episode=100
    )

    model = SKLearnSGD(
        loss='squared_loss',
        alpha=0.0,
        learning_rate='constant',
        eta0=0.0001,
        scale_eta0_for_y=False
    )

    feature_extractor = CartpoleFeatureExtractor(
        environment=environment
    )

    q_S_A = ApproximateStateActionValueEstimator(
        environment=environment,
        epsilon=0.02,
        model=model,
        feature_extractor=feature_extractor,
        formula=None,
        plot_model=False,
        plot_model_per_improvements=None,
        plot_model_bins=None
    )

    agent = StochasticMdpAgent(
        name='Cartpole Agent',
        random_state=random_state,
        pi=q_S_A.get_initial_policy(),
        gamma=0.95
    )

    iterate_value_q_pi(
        agent=agent,
        environment=environment,
        num_improvements=15000,
        num_episodes_per_improvement=1,
        num_updates_per_improvement=1,
        alpha=None,
        mode=Mode.SARSA,
        n_steps=None,
        planning_environment=None,
        make_final_policy_greedy=True,
        q_S_A=q_S_A,
        num_improvements_per_plot=100
    )


if __name__ == '__main__':
    main()
