from argparse import Namespace
from datetime import datetime
from flatland.envs.malfunction_generators import MalfunctionParameters
from src.psppo.ps_ppo_flatland import train_multiple_agents
import os

def curriculum_train():
    print("=" * 60)
    print("CURRICULUM LEARNING - GNN + Action Masking")
    print("=" * 60)

    base_training = {
        # Network
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
        # Training
        "n_episodes": 500,
        "horizon": 2048, "epochs": 8,
        "batch_size": 256, "batch_mode": "shuffle",
        "discount_factor": 0.99, "max_grad_norm": 0.5,
        "eps_clip": 0.3, "advantage_estimator": "gae",
        # Misc
        "checkpoint_interval": 100,
        "evaluation_mode": False, "eval_episodes": 25,
        "use_gpu": False, "render": False, "print_stats": True,
        "wandb_project": "flatland-challenge-ps-ppo-test",
        "wandb_entity": "fiorenzoparascandolo",
        "wandb_tag": "curriculum-gnn",
        "automatic_name_saving": False,
        "tensorboard_path": "logs/curriculum/",
        # Masking + GNN
        "action_masking": True, "allow_no_op": False,
        "use_gnn": True, "gnn_hidden_dim": 128,
        "gnn_heads": 4, "gnn_layers": 2,
    }

    base_env = {
        "seed": 14,
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 30,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "malfunction_parameters": MalfunctionParameters(
            malfunction_rate=0.005,
            min_duration=15,
            max_duration=50),
        "speed_profiles": {
            1.: 0.25, 1./2.: 0.25,
            1./3.: 0.25, 1./4.: 0.25},
        "custom_observations": False,
        "reward_shaping": True, "uniform_reward": True,
        "stop_penalty": -0.2, "invalid_action_penalty": -0.0,
        "deadlock_penalty": -10.0,
        "shortest_path_penalty_coefficient": 1.2,
        "done_bonus": 0.5,
    }

    # ==========================================
    # Stage 1: 3 tàu, map 30x30
    # ==========================================
    print("\n>>> STAGE 1: 3 trains, 30x30 map, 600 episodes")
    env_s1 = {**base_env, "n_agents": 3, "x_dim": 30, "y_dim": 30, "n_cities": 3}
    train_s1 = {**base_training,
                "n_episodes": 500,
                "load_model_path": "",
                "save_model_path": "curriculum_stage1.pt",
                "wandb_tag": "curriculum-stage1"}
    train_multiple_agents(Namespace(**env_s1), Namespace(**train_s1))
    print(">>> Stage 1 done! Saved: curriculum_stage1.pt")

    # ==========================================
    # Stage 2: 5 tàu, map 40x40
    # ==========================================
    print("\n>>> STAGE 2: 5 trains, 40x40 map, 600 episodes")
    env_s2 = {**base_env, "n_agents": 5, "x_dim": 40, "y_dim": 40, "n_cities": 4}
    train_s2 = {**base_training,
                "n_episodes": 500,
                "load_model_path": "curriculum_stage1.pt",
                "save_model_path": "curriculum_stage2.pt",
                "wandb_tag": "curriculum-stage2"}
    train_multiple_agents(Namespace(**env_s2), Namespace(**train_s2))
    print(">>> Stage 2 done! Saved: curriculum_stage2.pt")

    # # ==========================================
    # # Stage 2.5: 6 tàu, map 45x30
    # # ==========================================
    # print("\n>>> STAGE 2.5: 6 trains, 45x30 map, 600 episodes")
    # env_s2_5 = {**base_env, "n_agents": 6, "x_dim": 45, "y_dim": 30, "n_cities": 5}
    # train_s2_5 = {**base_training,
    #               "n_episodes": 600,
    #               "load_model_path": "curriculum_stage2.pt",
    #               "save_model_path": "curriculum_stage2_5.pt",
    #               "wandb_tag": "curriculum-stage2-5"}
    # train_multiple_agents(Namespace(**env_s2_5), Namespace(**train_s2_5))
    # print(">>> Stage 2.5 done! Saved: curriculum_stage2_5.pt")

    # ==========================================
    # Stage 3: 8 tàu, map 48x27
    # ==========================================
    print("\n>>> STAGE 3: 8 trains, 48x27 map, 600 episodes")
    env_s3 = {**base_env, "n_agents": 8, "x_dim": 48, "y_dim": 27, "n_cities": 5}
    train_s3 = {**base_training,
                "n_episodes": 500,
                "load_model_path": "curriculum_stage2.pt",
                "save_model_path": "curriculum_stage3.pt",
                "wandb_tag": "curriculum-stage3"}
    train_multiple_agents(Namespace(**env_s3), Namespace(**train_s3))
    print(">>> Stage 3 done! Saved: curriculum_stage3.pt")

    # ==========================================
    # Stage 4: Fine-tuning - tăng done_bonus, giảm learning rate
    # ==========================================
    print("\n>>> STAGE 4: Fine-tune 8 trains, 48x27, 500 episodes")
    env_s4 = {**base_env, "n_agents": 8, "x_dim": 48, "y_dim": 27, "n_cities": 5,
              "deadlock_penalty": -10.0,
              "done_bonus": 2.0}
    train_s4 = {**base_training,
                "n_episodes": 500,
                # "learning_rate": 0.002,  # original (from base_training)
                "learning_rate": 0.0005,   # ← giảm lr để fine-tune ổn định
                # "entropy_coefficient": 0.01,  # original (from base_training)
                "entropy_coefficient": 0.005,  # ← giảm entropy để exploit nhiều hơn
                "load_model_path": "curriculum_stage3.pt",
                "save_model_path": "curriculum_stage4.pt",
                "wandb_tag": "curriculum-stage4"}
    train_multiple_agents(Namespace(**env_s4), Namespace(**train_s4))
    print(">>> Stage 4 done! Saved: curriculum_stage4.pt")

    # print("\n" + "=" * 60)
    # print("CURRICULUM TRAINING COMPLETE!")
    # print("Final model: curriculum_stage4.pt")
    # print("=" * 60)

    # ==========================================
    # Auto-save lên Google Drive
    # ==========================================
    print("\n>>> Saving models to Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        
        import shutil
        save_dir = '/content/drive/MyDrive/flatland_models'
        os.makedirs(save_dir, exist_ok=True)
        
        for f in ['curriculum_stage1.pt', 'curriculum_stage2.pt',
                  'curriculum_stage3.pt', 'curriculum_stage4.pt']:
            src = f'/content/FlatlandChallenge/{f}'
            dst = f'{save_dir}/{f}'
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"  ✅ Saved {f}")
            else:
                print(f"  ⚠️ Not found: {f}")
        
        print(f">>> All models saved to {save_dir}")
    except Exception as e:
        print(f"  ⚠️ Drive save failed: {e}")
        print("  → Models still available at /content/FlatlandChallenge/")

    print("\n" + "=" * 60)
    print("CURRICULUM TRAINING COMPLETE!")
    print("Final model: curriculum_stage4.pt")
    print("=" * 60)

if __name__ == "__main__":
    curriculum_train()
