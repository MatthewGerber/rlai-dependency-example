from numpy.random import RandomState
from rlai.core.environments.gymnasium import Gym, CartpoleFeatureExtractor
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.gpi.state_action_value.function_approximation import ApproximateStateActionValueEstimator
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from rlai.models.sklearn import SKLearnSGD

from rlai.gpi.state_action_value.function_approximation.models.sklearn import SKLearnSGD as SKLearnSGDApproximator


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
        loss='squared_error',
        alpha=0.0,
        learning_rate='constant',
        eta0=0.0001
    )

    function_approximator = SKLearnSGDApproximator(model)

    feature_extractor = CartpoleFeatureExtractor(
        environment=environment
    )

    q_S_A = ApproximateStateActionValueEstimator(
        environment=environment,
        epsilon=0.02,
        model=function_approximator,
        feature_extractor=feature_extractor,
        formula=None,
        plot_model=False,
        plot_model_per_improvements=None,
        plot_model_bins=None
    )

    agent = ActionValueMdpAgent(
        name='Cartpole Agent',
        random_state=random_state,
        gamma=0.95,
        q_S_A=q_S_A
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
        num_improvements_per_plot=100
    )


if __name__ == '__main__':
    main()
