from argparse import ArgumentParser
import pickle

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from vpt.agent import MineRLAgent, ENV_KWARGS, validate_env


def load_agent(model_path, weights_path):
    agent_parameters = pickle.load(open(model_path, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights_path)
    return agent


def main(model, weights):
    env = HumanSurvival(**ENV_KWARGS).make()
    validate_env(env)

    print("---Loading model---")
    agent = load_agent(model, weights)

    print("---Launching MineRL enviroment (be patient)---")
    obs = env.reset()

    while True:
        minerl_action = agent.get_action(obs)
        obs, reward, done, info = env.step(minerl_action)
        env.render()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the '.weights' file to be loaded.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the '.model' file to be loaded.",
    )

    args = parser.parse_args()

    main(args.model, args.weights)
