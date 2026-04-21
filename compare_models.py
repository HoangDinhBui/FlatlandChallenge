import os
import random
import inspect
import numpy as np
import torch
import pandas as pd
from argparse import Namespace

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.malfunction_generators import MalfunctionParameters
from flatland.envs import malfunction_generators as mg_mod
from flatland.envs.rail_env import RailEnv

try:
    from flatland.envs.rail_generators import sparse_rail_generator
except ImportError:
    pass

try:
    from flatland.envs.line_generators import sparse_line_generator
except Exception:
    sparse_line_generator = None

try:
    from flatland.envs.schedule_generators import sparse_schedule_generator
except Exception:
    sparse_schedule_generator = None

try:
    import gym
except Exception:
    import gymnasium as gym

from src.common.deadlocks import DeadlocksDetector
from src.common.observation import NormalizeObservations
from src.common.wrappers import RewardsWrapper, StatsWrapper
from src.common.action_skipping_masking import get_action_masking
from src.psppo.policy import PsPPOPolicy
from src.psppo.ps_ppo_flatland import get_agent_ids

# Monkey-patching GridTransitionMap compatibility cho Flatland 4.2.4
import flatland.core.transition_map
old_get_transitions = flatland.core.transition_map.GridTransitionMap.get_transitions
def _patched_get_transitions(self, *args, **kwargs):
    if len(args) == 3:
        return old_get_transitions(self, ((args[0], args[1]), args[2]))
    return old_get_transitions(self, *args, **kwargs)
flatland.core.transition_map.GridTransitionMap.get_transitions = _patched_get_transitions


# Backward-compat
if not hasattr(mg_mod, 'ParamMalfunctionGen') and hasattr(mg_mod, 'malfunction_from_params'):
    def _compat_ParamMalfunctionGen(params):
        obj = mg_mod.malfunction_from_params(params)
        if isinstance(obj, tuple):
            return obj[0]
        return obj
    mg_mod.ParamMalfunctionGen = _compat_ParamMalfunctionGen

def _build_rail_generator(env_params):
    try:
        return sparse_rail_generator(
            max_num_cities=env_params.n_cities, grid_mode=False,
            max_rails_between_cities=env_params.max_rails_between_cities,
            max_rail_pairs_in_city=env_params.max_rails_in_city, seed=env_params.seed,
        )
    except TypeError:
        return sparse_rail_generator(
            max_num_cities=env_params.n_cities, grid_mode=False,
            max_rails_between_cities=env_params.max_rails_between_cities,
            max_rails_in_city=env_params.max_rails_in_city, seed=env_params.seed,
        )

def _build_rail_env_kwargs(env_params, observation):
    sig = inspect.signature(RailEnv)
    env_kwargs = {
        'width': env_params.x_dim, 'height': env_params.y_dim,
        'rail_generator': _build_rail_generator(env_params),
        'number_of_agents': env_params.n_agents,
        'obs_builder_object': observation, 'random_seed': env_params.seed,
    }
    if 'line_generator' in sig.parameters and sparse_line_generator is not None:
        env_kwargs['line_generator'] = sparse_line_generator()
    elif 'schedule_generator' in sig.parameters and sparse_schedule_generator is not None:
        env_kwargs['schedule_generator'] = sparse_schedule_generator(env_params.speed_profiles)

    mgpd = mg_mod.malfunction_from_params(env_params.malfunction_parameters) if hasattr(mg_mod, 'malfunction_from_params') else None
    if 'malfunction_generator_and_process_data' in sig.parameters:
        if mgpd is None:
            gen = mg_mod.ParamMalfunctionGen(env_params.malfunction_parameters)
            mgpd = (gen, None)
        env_kwargs['malfunction_generator_and_process_data'] = mgpd
    elif 'malfunction_generator' in sig.parameters:
        if hasattr(mg_mod, 'ParamMalfunctionGen'):
            env_kwargs['malfunction_generator'] = mg_mod.ParamMalfunctionGen(env_params.malfunction_parameters)
        elif mgpd is not None:
            env_kwargs['malfunction_generator'] = mgpd[0] if isinstance(mgpd, tuple) else mgpd
    return env_kwargs

class FlatlandGymEnv(gym.Env):
    def __init__(self, rail_env, custom_observations, env_params):
        self.rail_env = rail_env
        self.deadlocks_detector = DeadlocksDetector()
        self.observation_normalizer = NormalizeObservations(
            self.rail_env.obs_builder.observation_dim, env_params.observation_tree_depth,
            custom_observations, self.rail_env.width, self.rail_env.height, env_params.observation_radius,
        )
        self.state_size = self.observation_normalizer.state_size

    def reset(self):
        obs, info = self.rail_env.reset(regenerate_rail=True, regenerate_schedule=True)
        self.observation_normalizer.reset_custom_obs(self.rail_env)
        self.deadlocks_detector.reset(self.rail_env.get_num_agents())
        info['deadlocks'] = {a: self.deadlocks_detector.deadlocks[a] for a in range(self.rail_env.get_num_agents())}
        for a in obs:
            if obs[a] is not None:
                obs[a] = self.observation_normalizer.normalize_observation(obs[a], self.rail_env, a, info['deadlocks'][a])
        return obs, info

    def step(self, action_dict):
        obs, rewards, dones, info = self.rail_env.step(action_dict)
        deadlocks = self.deadlocks_detector.step(self.rail_env)
        info['deadlocks'] = {a: deadlocks[a] for a in range(self.rail_env.get_num_agents())}
        for a in obs:
            if obs[a] is not None:
                obs[a] = self.observation_normalizer.normalize_observation(obs[a], self.rail_env, a, info['deadlocks'][a])
        return obs, rewards, dones, info

class FlatlandRailEnv:
    def __init__(self, train_params, env_params, observation, custom_observations, reward_wrapper, stats_wrapper):
        self.env = FlatlandGymEnv(RailEnv(**_build_rail_env_kwargs(env_params, observation)), custom_observations, env_params)
        self.state_size = self.env.state_size
        self.env = RewardsWrapper(self.env, env_params.invalid_action_penalty, env_params.stop_penalty,
            env_params.deadlock_penalty, env_params.shortest_path_penalty_coefficient,
            env_params.done_bonus, env_params.uniform_reward)
        self.env = StatsWrapper(self.env, env_params)

    def reset(self): return self.env.reset()
    def step(self, action_dict): return self.env.step(action_dict)
    def get_rail_env(self): return self.env.rail_env

def evaluate_model(model_path, run_seed=42, eval_episodes=5):
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    env_cfg = {
        'seed': run_seed, 'observation_tree_depth': 2, 'observation_radius': 10, 'observation_max_path_depth': 30,
        'max_rails_between_cities': 2, 'max_rails_in_city': 3,
        'speed_profiles': {1.: 0.25, 1./2.: 0.25, 1./3.: 0.25, 1./4.: 0.25},
        'custom_observations': False, 'reward_shaping': True, 'uniform_reward': True,
        'stop_penalty': -0.2, 'invalid_action_penalty': -0.0, 'deadlock_penalty': -10.0,
        'shortest_path_penalty_coefficient': 1.2, 'done_bonus': 0.5,
        'malfunction_parameters': MalfunctionParameters(malfunction_rate=0.01, min_duration=15, max_duration=50),
        'n_agents': 8, 'x_dim': 48, 'y_dim': 27, 'n_cities': 5,
    }

    tr_cfg = {
        'shared': False, 'shared_recurrent': True, 'linear_size': 128, 'hidden_size': 64,
        'critic_mlp_width': 128, 'critic_mlp_depth': 3, 'last_critic_layer_scaling': 0.1,
        'actor_mlp_width': 128, 'actor_mlp_depth': 3, 'last_actor_layer_scaling': 0.01,
        'learning_rate': 0.002, 'adam_eps': 1e-5, 'activation': 'Tanh', 'lmbda': 0.95,
        'entropy_coefficient': 0.01, 'value_loss_coefficient': 0.001,
        'n_episodes': 50, 'horizon': 2048, 'epochs': 8, 'batch_size': 256, 'batch_mode': 'shuffle',
        'discount_factor': 0.99, 'max_grad_norm': 0.5, 'eps_clip': 0.3, 'advantage_estimator': 'gae',
        'checkpoint_interval': 100, 'evaluation_mode': True, 'eval_episodes': eval_episodes,
        'use_gpu': False, 'render': False, 'print_stats': False, 'action_masking': True,
        'allow_no_op': False, 'use_gnn': True, 'gnn_hidden_dim': 128, 'gnn_heads': 4, 'gnn_layers': 2,
        'load_model_path': model_path, 'save_model_path': None, 'wandb_project': None,
        'wandb_entity': None, 'wandb_tag': None, 'automatic_name_saving': False, 'tensorboard_path': None,
    }

    env_params = Namespace(**env_cfg)
    train_params = Namespace(**tr_cfg)

    predictor = ShortestPathPredictorForRailEnv(env_params.observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=env_params.observation_tree_depth, predictor=predictor)

    env = FlatlandRailEnv(train_params, env_params, tree_observation, False, True, False)
    env.reset()

    action_size = env.get_rail_env().action_space[0]
    ppo = PsPPOPolicy(env.state_size + 1, action_size, train_params, env_params.n_agents)

    max_steps = int(4 * 2 * (env_params.y_dim + env_params.x_dim + (env_params.n_agents / env_params.n_cities)))
    
    # Các metrics cần thu thập qua nhiều episodes
    scores, completions, deadlocks, tasks_finished = [], [], [], []
    action_probs_list = []

    for _ in range(eval_episodes):
        prev_obs, info = env.reset()
        done = {a: False for a in range(env_params.n_agents)}
        done['__all__'] = False
        agent_ids = get_agent_ids(env.get_rail_env().agents, env_params.malfunction_parameters.malfunction_rate)

        for _step in range(max_steps):
            action_dict = {}
            for agent in prev_obs:
                action_mask = get_action_masking(env, agent, action_size, train_params)
                if info['action_required'][agent]:
                    action_dict[agent] = ppo.act(np.append(prev_obs[agent], [agent_ids[agent]]), action_mask, agent_id=agent)

            next_obs, rewards, done, info = env.step(action_dict)
            for a in range(env_params.n_agents):
                if not done[a]: prev_obs[a] = next_obs[a].copy()
            if done['__all__']: break

        scores.append(float(env.env.normalized_score))
        completions.append(float(env.env.completion_percentage) * 100.0)
        deadlocks.append(float(env.env.deadlocks_percentage) * 100.0)
        tasks_finished.append(int(env.env.tasks_finished))
        action_probs_list.append(env.env.action_probs)

    # Chuyển đổi action_probs thành array để tính mean/std cho từng action
    action_probs_arr = np.array(action_probs_list) * 100.0 # Tính theo %
    
    return {
        'score_mean': np.mean(scores), 'score_std': np.std(scores),
        'completion_mean_pct': np.mean(completions), 'completion_std_pct': np.std(completions),
        'deadlock_mean_pct': np.mean(deadlocks), 'deadlock_std_pct': np.std(deadlocks),
        'tasks_finished_mean': np.mean(tasks_finished), 'tasks_finished_std': np.std(tasks_finished),
        'action_wait_mean_pct': np.mean(action_probs_arr[:, 0]), 'action_wait_std_pct': np.std(action_probs_arr[:, 0]),
        'action_left_mean_pct': np.mean(action_probs_arr[:, 1]), 'action_left_std_pct': np.std(action_probs_arr[:, 1]),
        'action_forward_mean_pct': np.mean(action_probs_arr[:, 2]), 'action_forward_std_pct': np.std(action_probs_arr[:, 2]),
        'action_right_mean_pct': np.mean(action_probs_arr[:, 3]), 'action_right_std_pct': np.std(action_probs_arr[:, 3]),
        'action_stop_mean_pct': np.mean(action_probs_arr[:, 4]), 'action_stop_std_pct': np.std(action_probs_arr[:, 4])
    }

if __name__ == "__main__":
    models = {
        "Version 1": "version1/curriculum_v1_stage3.pt",
        "Version 2": "version2/curriculum_v2_stage3.pt",
        "Version 3": "version3/curriculum_v3_stage3.pt",
        "Version 4": "version4/curriculum_v4_stage3.pt"
    }

    results = []
    print("Bắt đầu đánh giá các model... (Mỗi model test 3 episodes để tiết kiệm thời gian)")
    for name, path in models.items():
        if os.path.exists(path):
            print(f"Đang đánh giá {name} ({path})...")
            try:
                res = evaluate_model(path, run_seed=42, eval_episodes=3)
                res['Model'] = name
                results.append(res)
            except Exception as e:
                print(f"Lỗi khi đánh giá {name}: {e}")
        else:
            print(f"Không tìm thấy model: {path}")

    if results:
        print("\n=== KẾT QUẢ SO SÁNH 4 MÔ HÌNH CHI TIẾT (MEAN ± STD) ===")
        print("="*80)
        for r in results:
            name = r['Model']
            print(f"Mô hình: {name}")
            print(f"  - Điểm số (Score):          {r['score_mean']:8.3f} ± {r['score_std']:.3f}")
            print(f"  - Tỉ lệ hoàn thành (%):     {r['completion_mean_pct']:8.2f} ± {r['completion_std_pct']:.2f}")
            print(f"  - Tỉ lệ kẹt tàu (Deadlock): {r['deadlock_mean_pct']:8.2f} ± {r['deadlock_std_pct']:.2f}%")
            print(f"  - Số tàu tới đích:          {r['tasks_finished_mean']:8.2f} ± {r['tasks_finished_std']:.2f} tàu")
            print("  - Phân bổ hành động (Action Probs):")
            print(f"      + Chờ / Không làm gì (Wait): {r['action_wait_mean_pct']:6.2f}% ± {r['action_wait_std_pct']:.2f}%")
            print(f"      + Rẽ Trái (Left):            {r['action_left_mean_pct']:6.2f}% ± {r['action_left_std_pct']:.2f}%")
            print(f"      + Đi Thẳng (Forward):        {r['action_forward_mean_pct']:6.2f}% ± {r['action_forward_std_pct']:.2f}%")
            print(f"      + Rẽ Phải (Right):           {r['action_right_mean_pct']:6.2f}% ± {r['action_right_std_pct']:.2f}%")
            print(f"      + Dừng Lại (Stop):           {r['action_stop_mean_pct']:6.2f}% ± {r['action_stop_std_pct']:.2f}%")
            print("-" * 80)
    else:
        print("Không có kết quả nào!")