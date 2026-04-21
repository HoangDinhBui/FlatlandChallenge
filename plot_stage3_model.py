import os
import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import pandas as pd
from argparse import Namespace

# Đưa Monkey-patching GridTransitionMap cho Flatland 4.2.4 vào để script không lỗi
import flatland.core.transition_map
old_get_transitions = flatland.core.transition_map.GridTransitionMap.get_transitions
def _patched_get_transitions(self, *args, **kwargs):
    if len(args) == 3:
        return old_get_transitions(self, ((args[0], args[1]), args[2]))
    return old_get_transitions(self, *args, **kwargs)
flatland.core.transition_map.GridTransitionMap.get_transitions = _patched_get_transitions

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

    n_eval_episodes = 50 # Tự động chạy 50 ván nếu chưa có CSV
    max_steps = int(4 * 2 * (env_params.y_dim + env_params.x_dim + (env_params.n_agents / env_params.n_cities)))

    csv_path = 'evaluation_stage3_results.csv'
    
    # 1. KIỂM TRA XEM ĐÃ CÓ DATA LƯU SẴN CHƯA
    if os.path.exists(csv_path):
        print(f"[*] Tìm thấy file '{csv_path}'. Đang đọc dữ liệu để vẽ biểu đồ TỨC THÌ (không cần chạy lại)...")
        df = pd.read_csv(csv_path)
        n_eval_episodes = len(df)
    else:
        # Nếu chưa có, tiến hành MÔ PHỎNG môi trường để lấy data
        print(f"[*] Chưa có dữ liệu lưu sẵn. Đang chạy mô phỏng {n_eval_episodes} episodes để lấy logs (việc này sẽ tốn thời gian)...")
        
        # Load weights cho model
        model_path = os.path.join(os.getcwd(), 'curriculum_stage3.pt')
        if os.path.exists(model_path):
            ppo.policy.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            print(f"Không tìm thấy model tại {model_path}")
            return

        scores = []
        completions = []
        deadlocks = []
        forward_ratios = []

        for ep in range(n_eval_episodes):
            prev_obs, info = env.reset()
            done = {a: False for a in range(env_params.n_agents)}
            done["__all__"] = False
            
            agent_ids = get_agent_ids(env.get_rail_env().agents, env_params.malfunction_parameters.malfunction_rate)
            action_count = [0] * action_size

            for step in range(max_steps):
                action_dict = dict()
                for agent in prev_obs:
                    action_mask = get_action_masking(env, agent, action_size, train_params)
                    if info["action_required"][agent]:
                        a = ppo.act(np.append(prev_obs[agent], [agent_ids[agent]]), action_mask, agent_id=agent)
                        action_dict[agent] = a
                        action_count[a] += 1

                next_obs, rewards, done, info = env.step(action_dict)
                
                for a in range(env_params.n_agents):
                    if not done[a]:
                        prev_obs[a] = next_obs[a].copy()

                if done['__all__']:
                    break

            score = env.env.normalized_score
            completion = env.env.completion_percentage
            dl = env.env.deadlocks_percentage
            
            total_acts = sum(action_count)
            fwd_ratio = (action_count[2] / max(1, total_acts)) * 100.0
            
            scores.append(score)
            completions.append(completion * 100)
            deadlocks.append(dl * 100)
            forward_ratios.append(fwd_ratio)
            
            print(f"Episode {ep+1:2d}/{n_eval_episodes} - Score: {score:7.3f} | Hoàn thành: {completion:7.2%} | Tắc nghẽn: {dl:7.2%} | Fwd Ratio: {fwd_ratio:5.1f}%")

        # Lưu dữ liệu ra Dataframe & CSV để lưu lại cho lần sau
        df = pd.DataFrame({
            'episode': range(1, n_eval_episodes + 1),
            'score': scores,
            'completion': completions,
            'deadlock': deadlocks,
            'forward_ratio': forward_ratios
        })
        df.to_csv(csv_path, index=False)
        print(f"[*] Đã lưu dữ liệu mô phỏng thành công vào '{csv_path}' để lần sau không cần chạy lại!")

    # 2. TÍNH TOÁN & VẼ BIỂU ĐỒ (Phần này sẽ diễn ra ngay lập tức nếu đọc từ CSV)
    # Tính Rolling Mean w=5
    df['completion_rolling'] = df['completion'].rolling(window=5, min_periods=1).mean()
    df['deadlock_rolling'] = df['deadlock'].rolling(window=5, min_periods=1).mean()
    df['forward_rolling'] = df['forward_ratio'].rolling(window=5, min_periods=1).mean()
    
    # --- VẼ HÌNH 1: Score & Completion Rate ---
    fig1, ax1 = plt.subplots(1, 2, figsize=(15, 5))
    
    # (a) Score
    ax1[0].plot(df['episode'], df['score'], label='score', color='#4C72B0', linewidth=1.5)
    ax1[0].set_title('Curriculum Stage3 - Eval Score', fontsize=12)
    ax1[0].set_xlabel('episode')
    ax1[0].set_ylabel('score')
    ax1[0].grid(True, linestyle='-', alpha=0.3)
    ax1[0].text(0.5, -0.15, '(a)', transform=ax1[0].transAxes, fontsize=14, ha='center')
    
    # (b) Completion Rate (%)
    ax1[1].plot(df['episode'], df['completion'], label='Completion', color='#008000', linewidth=1.5)
    ax1[1].set_title('Curriculum Stage3 - Completion Rate (%)', fontsize=12)
    ax1[1].set_xlabel('episode')
    ax1[1].set_ylabel('Completion (%)')
    ax1[1].grid(True, linestyle='-', alpha=0.3)
    ax1[1].text(0.5, -0.15, '(b)', transform=ax1[1].transAxes, fontsize=14, ha='center')
    
    plt.tight_layout()
    plt.savefig('fig1_score_and_completion.png', dpi=200)
    plt.close(fig1)

    # --- VẼ HÌNH 2: Rolling Mean Charts (Per-Episode x Rolling mean w=5) ---
    fig2, ax2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Completion per Episode
    ax2[0].plot(df['episode'], df['completion'], label='Per-episode', color='#A6CBE2', linewidth=1.5)
    ax2[0].plot(df['episode'], df['completion_rolling'], label='Rolling mean (w=5)', color='#FF7F0E', linewidth=2.5)
    ax2[0].set_title('Completion per Episode', fontsize=12)
    ax2[0].set_xlabel('Episode')
    ax2[0].set_ylabel('Completion (%)')
    ax2[0].grid(True, linestyle='-', alpha=0.3)
    ax2[0].legend()
    ax2[0].text(0.5, -0.15, '(c)', transform=ax2[0].transAxes, fontsize=14, ha='center')
    
    # Plot 2: Deadlock per Episode
    ax2[1].plot(df['episode'], df['deadlock'], label='Per-episode', color='#F5B8B8', linewidth=1.5)
    ax2[1].plot(df['episode'], df['deadlock_rolling'], label='Rolling mean (w=5)', color='#800000', linewidth=2.5)
    ax2[1].set_title('Deadlock per Episode', fontsize=12)
    ax2[1].set_xlabel('Episode')
    ax2[1].set_ylabel('Deadlock (%)')
    ax2[1].grid(True, linestyle='-', alpha=0.3)
    ax2[1].legend()
    ax2[1].text(0.5, -0.15, '(d)', transform=ax2[1].transAxes, fontsize=14, ha='center')

    # Plot 3: Forward Action Ratio per Episode
    ax2[2].plot(df['episode'], df['forward_ratio'], label='Per-episode', color='#A9DFB8', linewidth=1.5)
    ax2[2].plot(df['episode'], df['forward_rolling'], label='Rolling mean (w=5)', color='#006400', linewidth=2.5)
    ax2[2].set_title('Forward Action Ratio per Episode', fontsize=12)
    ax2[2].set_xlabel('Episode')
    ax2[2].set_ylabel('Forward (%)')
    ax2[2].grid(True, linestyle='-', alpha=0.3)
    ax2[2].legend()
    ax2[2].text(0.5, -0.15, '(e)', transform=ax2[2].transAxes, fontsize=14, ha='center')
    
    plt.tight_layout()
    plt.savefig('fig2_rolling_metrics.png', dpi=200)
    plt.close(fig2)

    print(f"\n[+] ĐÃ VẼ XONG 2 ẢNH BIỂU ĐỒ (fig1_score_and_completion.png, fig2_rolling_metrics.png)")

if __name__ == '__main__':
    run_evaluation_and_plot()
