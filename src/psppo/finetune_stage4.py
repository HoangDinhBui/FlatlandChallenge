from argparse import Namespace
from flatland.envs.malfunction_generators import MalfunctionParameters
from src.psppo.ps_ppo_flatland import train_multiple_agents

def finetune():
    base_env = {
        "seed": 14,
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 30,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "malfunction_parameters": MalfunctionParameters(
            malfunction_rate=0.005, min_duration=15, max_duration=50),
        "speed_profiles": {1.: 0.25, 1./2.: 0.25, 1./3.: 0.25, 1./4.: 0.25},
        "custom_observations": False,
        "reward_shaping": True, "uniform_reward": True,
        "stop_penalty": -0.2, "invalid_action_penalty": -0.0,
        "deadlock_penalty": -10.0,
        "shortest_path_penalty_coefficient": 1.2,
        "done_bonus": 2.0,  # ← tăng mạnh
    }

    train_params = {
        "shared": False, "shared_recurrent": True,
        "linear_size": 128, "hidden_size": 64,
        "critic_mlp_width": 128, "critic_mlp_depth": 3,
        "last_critic_layer_scaling": 0.1,
        "actor_mlp_width": 128, "actor_mlp_depth": 3,
        "last_actor_layer_scaling": 0.01,
        # "learning_rate": 0.002, "adam_eps": 1e-5,  # original
        "learning_rate": 0.0005, "adam_eps": 1e-5,   # ← giảm lr để fine-tune ổn định
        "activation": "Tanh", "lmbda": 0.95,
        # "entropy_coefficient": 0.01, "value_loss_coefficient": 0.001,  # original
        "entropy_coefficient": 0.005, "value_loss_coefficient": 0.001,  # ← giảm entropy để exploit
        "n_episodes": 500,
        "horizon": 2048, "epochs": 8,
        "batch_size": 256, "batch_mode": "shuffle",
        "discount_factor": 0.99, "max_grad_norm": 0.5,
        "eps_clip": 0.3, "advantage_estimator": "gae",
        "checkpoint_interval": 100,
        "evaluation_mode": False, "eval_episodes": 25,
        "use_gpu": False, "render": False, "print_stats": True,
        "wandb_project": "flatland-challenge-ps-ppo-test",
        "wandb_entity": "fiorenzoparascandolo",
        "wandb_tag": "finetune-stage4",
        "automatic_name_saving": False,
        "tensorboard_path": "logs/finetune/",
        "action_masking": True, "allow_no_op": False,
        "use_gnn": True, "gnn_hidden_dim": 128,
        "gnn_heads": 4, "gnn_layers": 2,
        "load_model_path": "curriculum_stage3.pt",  # ← load từ stage 3
        "save_model_path": "curriculum_stage4.pt",
    }

    print(">>> Fine-tuning Stage 4: 8 tàu, done_bonus=2.0")
    env = {**base_env, "n_agents": 8, "x_dim": 48, "y_dim": 27, "n_cities": 5}
    train_multiple_agents(Namespace(**env), Namespace(**train_params))
    print(">>> Done! Saved: curriculum_stage4.pt")

if __name__ == "__main__":
    finetune()