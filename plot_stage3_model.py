import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from argparse import Namespace

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.malfunction_generators import MalfunctionParameters

from src.common.action_skipping_masking import get_action_masking
from src.common.flatland_railenv import FlatlandRailEnv
from src.psppo.policy import PsPPOPolicy
from src.psppo.ps_ppo_flatland import get_agent_ids

def run_evaluation_and_plot():
    base_env = {
        "seed": 42,
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 30,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "speed_profiles": {1.: 0.25, 1./2.: 0.25, 1./3.: 0.25, 1./4.: 0.25},
        "custom_observations": False,
        "reward_shaping": True, "uniform_reward": True,
        "stop_penalty": -0.2, "invalid_action_penalty": -0.0,
        "deadlock_penalty": -10.0,
        "shortest_path_penalty_coefficient": 1.2,
        "done_bonus": 0.5,
        "malfunction_parameters": MalfunctionParameters(
            malfunction_rate=0.01,
            min_duration=15,
            max_duration=50),
        "n_agents": 8, "x_dim": 48, "y_dim": 27, "n_cities": 5
    }
    
    base_training = {
        "shared": False, "shared_recurrent": True,
        "linear_size": 128, "hidden_size": 64,
        "critic_mlp_width": 128, "critic_mlp_depth": 3,
        "last_critic_layer_scaling": 0.1,
        "actor_mlp_width": 128, "actor_mlp_depth": 3,
        "last_actor_layer_scaling": 0.01,
        "learning_rate": 0.002, "adam_eps": 1e-5,
        "activation": "Tanh", "lmbda": 0.95,
        "entropy_coefficient": 0.01,
        "value_loss_coefficient": 0.001,
        "n_episodes": 50,
        "horizon": 2048, "epochs": 8,
        "batch_size": 256, "batch_mode": "shuffle",
        "discount_factor": 0.99, "max_grad_norm": 0.5,
        "eps_clip": 0.3, "advantage_estimator": "gae",
        "checkpoint_interval": 100,
        "evaluation_mode": True, "eval_episodes": 50,
        "use_gpu": False, "render": False, "print_stats": False,
        "action_masking": True, "allow_no_op": False,
        "use_gnn": True, "gnn_hidden_dim": 128,
        "gnn_heads": 4, "gnn_layers": 2,
        "load_model_path": "curriculum_stage3.pt",
        "save_model_path": None,
        "wandb_project": None, "wandb_entity": None,
        "wandb_tag": None, "automatic_name_saving": False,
        "tensorboard_path": None,
    }

    env_params = Namespace(**base_env)
    train_params = Namespace(**base_training)

    # Khởi tạo environment parameters
    predictor = ShortestPathPredictorForRailEnv(env_params.observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=env_params.observation_tree_depth, predictor=predictor)
    
    print("Khởi tạo môi trường...")
    env = FlatlandRailEnv(train_params, env_params, tree_observation, env_params.custom_observations, env_params.reward_shaping, True)
    print("Resetting môi trường...")
    env.reset()
    print("Môi trường sẵn sàng.")
    
    action_size = env.get_rail_env().action_space[0]
    ppo = PsPPOPolicy(env.state_size + 1, action_size, train_params, env_params.n_agents)

    # Load weights cho model
    model_path = os.path.join(os.getcwd(), 'curriculum_stage3.pt')
    if os.path.exists(model_path):
        ppo.policy.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded model từ {model_path}")
    else:
        print(f"Không tìm thấy model tại {model_path}")
        return

    n_eval_episodes = train_params.eval_episodes
    max_steps = int(4 * 2 * (env_params.y_dim + env_params.x_dim + (env_params.n_agents / env_params.n_cities)))

    scores = []
    completions = []
    deadlocks = []

    print(f"Đang chạy đánh giá {n_eval_episodes} episodes để lấy logs...")
    for ep in range(n_eval_episodes):
        prev_obs, info = env.reset()
        done = {a: False for a in range(env_params.n_agents)}
        done["__all__"] = False
        
        agent_ids = get_agent_ids(env.get_rail_env().agents, env_params.malfunction_parameters.malfunction_rate)

        for step in range(max_steps):
            action_dict = dict()
            for agent in prev_obs:
                action_mask = get_action_masking(env, agent, action_size, train_params)
                if info["action_required"][agent]:
                    action_dict[agent] = ppo.act(np.append(prev_obs[agent], [agent_ids[agent]]), action_mask, agent_id=agent)

            next_obs, rewards, done, info = env.step(action_dict)
            
            for a in range(env_params.n_agents):
                if not done[a]:
                    prev_obs[a] = next_obs[a].copy()

            if done['__all__']:
                break

        score = env.env.normalized_score
        completion = env.env.completion_percentage
        dl = env.env.deadlocks_percentage
        
        scores.append(score)
        completions.append(completion * 100) # chuyển sang phần trăm
        deadlocks.append(dl * 100)
        print(f"Episode {ep+1:2d}/{n_eval_episodes} - Score: {score:7.3f} | Hoàn thành: {completion:7.2%} | Tắc nghẽn: {dl:7.2%}")

    # Vẽ biểu đồ
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    
    axs[0].plot(scores, label='Score', color='green', marker='o', markersize=3)
    axs[0].set_title('Normalized Score / Episode')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Score')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    axs[1].plot(completions, label='Completion %', color='blue', marker='o', markersize=3)
    axs[1].set_title('Completion Percentage (%)')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Completion (%)')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    axs[2].plot(deadlocks, label='Deadlock %', color='red', marker='o', markersize=3)
    axs[2].set_title('Deadlocks Percentage (%)')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Deadlocks (%)')
    axs[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    output_png = 'curriculum_stage3_results.png'
    plt.savefig(output_png, dpi=200)
    print(f"\n[+] ĐÃ VẼ VÀ LƯU BIỂU ĐỒ TẠI {output_png}")
    # plt.show() # Un-comment nếu bạn muốn pop-up GUI hiển thị ngay

if __name__ == '__main__':
    run_evaluation_and_plot()
