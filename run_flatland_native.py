"""
Flatland Native Visualizer
===========================
Uses Flatland's built-in RenderTool with a Tkinter config GUI.
Load your trained GNN PS-PPO model, set parameters, and click Run.

Usage:
    python run_flatland_native.py
"""

import os
import sys
import math
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import torch
from PIL import Image, ImageTk

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fantasy_visualizer.model_loader import StandaloneGNNPsPPO

# Flatland
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool


# ── Observation normalizer (same as fantasy_visualizer) ──────────────

class SimpleObservationNormalizer:
    def __init__(self, tree_depth=2):
        self.tree_depth = tree_depth
        self.n_features = 11
        self.n_nodes = sum(4 ** i for i in range(tree_depth + 1))
        self.state_size = self.n_features * self.n_nodes

    def normalize(self, obs):
        if obs is None:
            return np.zeros(self.state_size)
        flat = self._flatten(obs, self.tree_depth)
        if len(flat) < self.state_size:
            flat = np.concatenate([flat, np.zeros(self.state_size - len(flat))])
        return flat[:self.state_size]

    def _flatten(self, node, depth):
        if node is None or depth < 0:
            size = self.n_features * sum(4 ** i for i in range(depth + 1))
            return np.zeros(size)
        feats = self._extract(node)
        if depth == 0:
            return feats
        children = []
        for key in ['L', 'F', 'R', 'B']:
            child = getattr(node, 'childs', {})
            if isinstance(child, dict) and key in child:
                children.append(self._flatten(child[key], depth - 1))
            else:
                child_size = self.n_features * sum(4 ** i for i in range(depth))
                children.append(np.zeros(child_size))
        return np.concatenate([feats] + children)

    def _extract(self, node):
        f = np.zeros(self.n_features)
        if node is None or isinstance(node, (int, float)):
            return f
        try:
            f[0] = 0 if math.isinf(node.dist_own_target_encountered) else node.dist_own_target_encountered
            f[1] = 0 if math.isinf(node.dist_other_target_encountered) else node.dist_other_target_encountered
            f[2] = 0 if math.isinf(node.dist_other_agent_encountered) else node.dist_other_agent_encountered
            f[3] = 0 if math.isinf(node.dist_potential_conflict) else node.dist_potential_conflict
            f[4] = 0 if math.isinf(node.dist_unusable_switch) else node.dist_unusable_switch
            f[5] = 0 if math.isinf(node.dist_to_next_branch) else node.dist_to_next_branch
            f[6] = 0 if math.isinf(node.dist_min_to_target) else node.dist_min_to_target
            f[7] = node.num_agents_same_direction
            f[8] = node.num_agents_opposite_direction
            f[9] = node.num_agents_malfunctioning
            f[10] = node.speed_min_fractional
        except (AttributeError, TypeError):
            pass
        return np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=-1.0)


def simple_action_mask(rail_env, handle, action_size=5):
    mask = [1] * action_size
    mask[0] = 0  # DO_NOTHING disabled
    agent = rail_env.agents[handle]
    if agent.position is None:
        return mask
    pos, d = agent.position, agent.direction
    for act in range(1, 4):  # MOVE_LEFT=1, FORWARD=2, RIGHT=3
        new_d = (d + [-1, 0, 1][act - 1]) % 4
        trans = rail_env.rail.get_transitions((pos, d))
        if trans[new_d] == 0:
            mask[act] = 0
    return mask


# ── Simulation runner ────────────────────────────────────────────────

class SimulationRunner:
    """Runs the Flatland simulation with the trained model."""

    def __init__(self, config, log_fn=None, render_fn=None):
        self.config = config
        self.log = log_fn or print
        self.render_fn = render_fn
        self.running = False
        self.paused = False

    def run(self):
        cfg = self.config
        self.running = True
        self.paused = False

        self.log(f"Creating {cfg['x_dim']}x{cfg['y_dim']} grid, "
                 f"{cfg['n_agents']} agents, seed={cfg['seed']}...")

        # Observation
        predictor = ShortestPathPredictorForRailEnv(cfg['obs_max_path_depth'])
        tree_obs = TreeObsForRailEnv(max_depth=cfg['obs_tree_depth'], predictor=predictor)

        # Environment
        env = RailEnv(
            width=cfg['x_dim'],
            height=cfg['y_dim'],
            rail_generator=sparse_rail_generator(
                max_num_cities=cfg['n_cities'],
                grid_mode=False,
                max_rails_between_cities=cfg['max_rails_between_cities'],
                max_rail_pairs_in_city=cfg['max_rails_in_city'],
                seed=cfg['seed'],
            ),
            line_generator=sparse_line_generator(),
            number_of_agents=cfg['n_agents'],
            malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(
                malfunction_rate=cfg['malfunction_rate'],
                min_duration=cfg['malfunction_min'],
                max_duration=cfg['malfunction_max'],
            )),
            obs_builder_object=tree_obs,
            random_seed=cfg['seed'],
        )

        obs, info = env.reset()
        self.log(f"Grid ready: {env.width}x{env.height}, "
                 f"{len(env.agents)} agents, "
                 f"{int((env.rail.grid != 0).sum())} rail cells")

        # Renderer (Flatland native)
        # Use PILSVG backend for high-quality SVG rendering in the GUI
        try:
            renderer = RenderTool(
                env, gl="PILSVG",
                screen_width=cfg['screen_width'],
                screen_height=cfg['screen_height'],
                show_debug=cfg['show_debug'],
            )
            renderer.set_new_rail()
        except Exception as e:
            self.log(f"Renderer Error: {e}")
            renderer = None

        # Model
        normalizer = SimpleObservationNormalizer(cfg['obs_tree_depth'])
        model = StandaloneGNNPsPPO(action_size=5, hidden_dim=128, n_heads=4, n_layers=2)

        model_path = cfg['model_path']
        if os.path.exists(model_path):
            model.load_checkpoint(model_path)
            self.log(f"Model loaded: {os.path.basename(model_path)}")
        else:
            self.log(f"WARNING: Model not found at {model_path}, using random policy")

        # Agent IDs (PS-PPO)
        max_val = 1.0 + 0.1 * 10
        agent_ids = {
            a.handle: (a.speed_counter.speed + cfg['malfunction_rate'] * 10) / max_val
            for a in env.agents
        }

        max_steps = int(4 * 2 * (cfg['y_dim'] + cfg['x_dim'] + cfg['n_agents'] / max(1, cfg['n_cities'])))
        self.log(f"Max steps: {max_steps}  |  Step delay: {cfg['step_delay']}s")

        dones = {a: False for a in range(len(env.agents))}
        dones['__all__'] = False

        n_episodes = cfg['n_episodes']

        for ep in range(1, n_episodes + 1):
            if not self.running:
                break

            if ep > 1:
                obs, info = env.reset()
                renderer.set_new_rail()
                dones = {a: False for a in range(len(env.agents))}
                dones['__all__'] = False

            self.log(f"\n--- Episode {ep}/{n_episodes} ---")

            total_reward = 0.0

            for step in range(max_steps):
                if not self.running:
                    break

                # Pause loop
                while self.paused and self.running:
                    time.sleep(0.1)

                action_dict = {}

                for h in range(len(env.agents)):
                    if dones.get(h, False):
                        continue
                    if obs.get(h) is None:
                        continue

                    norm = normalizer.normalize(obs[h])
                    full = np.append(norm, [agent_ids.get(h, 0.5)])
                    mask = simple_action_mask(env, h)
                    action_dict[h] = model.act(full, mask)

                obs, rewards, dones, info = env.step(action_dict)

                for a, r in rewards.items():
                    if isinstance(a, int):
                        total_reward += r

                if renderer:
                    try:
                        renderer.render_env(
                            show=False,
                            show_observations=cfg['show_observations'],
                            show_predictions=cfg['show_predictions'],
                            show_rowcols=cfg['show_rowcols'],
                        )
                        if self.render_fn:
                            img = renderer.get_image()
                            if img is not None:
                                self.render_fn(img)
                    except Exception as e:
                        self.log(f"Rendering frame failed: {e}")

                time.sleep(cfg['step_delay'])

                if dones.get('__all__', False):
                    completed = sum(1 for i in range(len(env.agents)) if dones.get(i, False))
                    self.log(f"  All done at step {step + 1}! "
                             f"Completed: {completed}/{len(env.agents)}, "
                             f"Reward: {total_reward:.2f}")
                    # Show final frame a bit longer
                    time.sleep(1.5)
                    break

            if not dones.get('__all__', False):
                completed = sum(1 for i in range(len(env.agents)) if dones.get(i, False))
                self.log(f"  Timeout at step {max_steps}. "
                         f"Completed: {completed}/{len(env.agents)}, "
                         f"Reward: {total_reward:.2f}")

        try:
            renderer.close_window()
        except Exception:
            pass

        self.running = False
        self.log("\nSimulation finished.")

    def stop(self):
        self.running = False

    def toggle_pause(self):
        self.paused = not self.paused


class SimulatorWindow:
    """A separate window to display the simulation frames."""
    def __init__(self, parent, title="Flatland Simulation", width=1200, height=800):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry(f"{width}x{height}")
        self.canvas = tk.Canvas(self.top, bg='#0c061e', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.photo = None
        self.closed = False
        self.top.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.closed = True
        self.top.destroy()

    def update_image(self, pil_img):
        if self.closed:
            return
        
        # Ensure we have a PIL Image
        if isinstance(pil_img, np.ndarray):
            pil_img = Image.fromarray(pil_img)
            
        # Determine dimensions
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = pil_img.size
            
        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        # Center the image
        self.canvas.create_image(max(cw // 2, 0), max(ch // 2, 0), image=self.photo, anchor='center')


# ── Tkinter Config GUI ──────────────────────────────────────────────

class ConfigGUI:
    """Tkinter GUI for configuring and launching the Flatland visualizer."""

    DEFAULTS = {
        'model_path': os.path.join(PROJECT_ROOT, 'curriculum_stage3.pt'),
        'n_agents': 2,
        'x_dim': 35,
        'y_dim': 35,
        'n_cities': 2,
        'max_rails_between_cities': 2,
        'max_rails_in_city': 3,
        'seed': 42,
        'obs_tree_depth': 2,
        'obs_max_path_depth': 30,
        'malfunction_rate': 0.005,
        'malfunction_min': 20,
        'malfunction_max': 50,
        'step_delay': 0.1,
        'n_episodes': 3,
        'screen_width': 1200,
        'screen_height': 800,
        'show_debug': True,
        'show_observations': True,
        'show_predictions': True,
        'show_rowcols': False,
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Flatland Visualizer - Configuration")
        self.root.resizable(True, True)
        self.root.minsize(580, 700)

        self.vars = {}
        self.runner = None
        self.sim_thread = None
        self.sim_window = None

        self._apply_theme()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_theme(self):
        style = ttk.Style()
        style.theme_use('clam')

        BG = '#1a1a2e'
        FG = '#e0d8c8'
        ACCENT = '#e6c84c'
        BTN_BG = '#16213e'
        ENTRY_BG = '#0f3460'
        SELECT = '#533483'

        self.root.configure(bg=BG)

        style.configure('.', background=BG, foreground=FG, fieldbackground=ENTRY_BG)
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=FG, font=('Segoe UI', 10))
        style.configure('Header.TLabel', background=BG, foreground=ACCENT, font=('Segoe UI', 13, 'bold'))
        style.configure('Title.TLabel', background=BG, foreground=ACCENT, font=('Georgia', 18, 'bold'))
        style.configure('Subtitle.TLabel', background=BG, foreground='#9a8c70', font=('Segoe UI', 9))
        style.configure('TEntry', fieldbackground=ENTRY_BG, foreground=FG,
                         insertcolor=FG, font=('Consolas', 10))
        style.configure('TCheckbutton', background=BG, foreground=FG, font=('Segoe UI', 10))
        style.configure('TLabelframe', background=BG, foreground=ACCENT, font=('Segoe UI', 10, 'bold'))
        style.configure('TLabelframe.Label', background=BG, foreground=ACCENT)

        style.configure('Run.TButton', background='#2d6a4f', foreground='white',
                         font=('Segoe UI', 12, 'bold'), padding=(20, 10))
        style.map('Run.TButton',
                   background=[('active', '#40916c'), ('disabled', '#333')],
                   foreground=[('disabled', '#666')])

        style.configure('Stop.TButton', background='#9b2226', foreground='white',
                         font=('Segoe UI', 11, 'bold'), padding=(15, 8))
        style.map('Stop.TButton',
                   background=[('active', '#ae2012'), ('disabled', '#333')],
                   foreground=[('disabled', '#666')])

        style.configure('Browse.TButton', background=BTN_BG, foreground=ACCENT,
                         font=('Segoe UI', 9), padding=(8, 2))
        style.map('Browse.TButton', background=[('active', SELECT)])

        style.configure('Pause.TButton', background='#e9c46a', foreground='#1a1a2e',
                         font=('Segoe UI', 10, 'bold'), padding=(12, 6))
        style.map('Pause.TButton', background=[('active', '#f4a261')])

        self._colors = {'BG': BG, 'FG': FG, 'ACCENT': ACCENT,
                         'ENTRY_BG': ENTRY_BG, 'LOG_BG': '#0a0a1a'}

    def _build_ui(self):
        # Main scrollable frame
        canvas = tk.Canvas(self.root, bg=self._colors['BG'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient='vertical', command=canvas.yview)
        self.main_frame = ttk.Frame(canvas)

        self.main_frame.bind('<Configure>',
                              lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.main_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Mouse wheel scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _on_mousewheel)

        frame = self.main_frame
        pad = {'padx': 12, 'pady': 2}

        # ── Title ──
        ttk.Label(frame, text="Flatland Visualizer", style='Title.TLabel').pack(pady=(15, 0))
        ttk.Label(frame, text="Configure parameters and run your trained model",
                   style='Subtitle.TLabel').pack(pady=(0, 10))

        # ── Model ──
        grp = ttk.LabelFrame(frame, text="  Model  ", padding=10)
        grp.pack(fill='x', **pad)
        row = ttk.Frame(grp)
        row.pack(fill='x')
        self.vars['model_path'] = tk.StringVar(value=self.DEFAULTS['model_path'])
        ttk.Label(row, text="Model path:").pack(side='left')
        ttk.Entry(row, textvariable=self.vars['model_path'], width=45).pack(side='left', padx=5, expand=True, fill='x')
        ttk.Button(row, text="Browse...", style='Browse.TButton',
                    command=self._browse_model).pack(side='left')

        # ── Environment ──
        grp = ttk.LabelFrame(frame, text="  Environment  ", padding=10)
        grp.pack(fill='x', **pad)

        self._add_row(grp, 'x_dim', 'Grid Width', int)
        self._add_row(grp, 'y_dim', 'Grid Height', int)
        self._add_row(grp, 'n_agents', 'Number of Agents', int)
        self._add_row(grp, 'n_cities', 'Number of Cities', int)
        self._add_row(grp, 'max_rails_between_cities', 'Max Rails Between Cities', int)
        self._add_row(grp, 'max_rails_in_city', 'Max Rails In City', int)
        self._add_row(grp, 'seed', 'Random Seed', int)

        # ── Malfunctions ──
        grp = ttk.LabelFrame(frame, text="  Malfunctions  ", padding=10)
        grp.pack(fill='x', **pad)

        self._add_row(grp, 'malfunction_rate', 'Malfunction Rate', float)
        self._add_row(grp, 'malfunction_min', 'Min Duration', int)
        self._add_row(grp, 'malfunction_max', 'Max Duration', int)

        # ── Observation ──
        grp = ttk.LabelFrame(frame, text="  Observation  ", padding=10)
        grp.pack(fill='x', **pad)

        self._add_row(grp, 'obs_tree_depth', 'Tree Depth', int)
        self._add_row(grp, 'obs_max_path_depth', 'Max Path Depth', int)

        # ── Simulation ──
        grp = ttk.LabelFrame(frame, text="  Simulation  ", padding=10)
        grp.pack(fill='x', **pad)

        self._add_row(grp, 'step_delay', 'Step Delay (sec)', float)
        self._add_row(grp, 'n_episodes', 'Number of Episodes', int)

        # ── Display ──
        grp = ttk.LabelFrame(frame, text="  Display  ", padding=10)
        grp.pack(fill='x', **pad)

        self._add_row(grp, 'screen_width', 'Window Width', int)
        self._add_row(grp, 'screen_height', 'Window Height', int)

        self.vars['show_debug'] = tk.BooleanVar(value=self.DEFAULTS['show_debug'])
        self.vars['show_observations'] = tk.BooleanVar(value=self.DEFAULTS['show_observations'])
        self.vars['show_predictions'] = tk.BooleanVar(value=self.DEFAULTS['show_predictions'])
        self.vars['show_rowcols'] = tk.BooleanVar(value=self.DEFAULTS['show_rowcols'])

        checks_frame = ttk.Frame(grp)
        checks_frame.pack(fill='x', pady=5)
        ttk.Checkbutton(checks_frame, text="Debug info", variable=self.vars['show_debug']).pack(side='left', padx=10)
        ttk.Checkbutton(checks_frame, text="Observations", variable=self.vars['show_observations']).pack(side='left', padx=10)
        ttk.Checkbutton(checks_frame, text="Predictions", variable=self.vars['show_predictions']).pack(side='left', padx=10)
        ttk.Checkbutton(checks_frame, text="Row/Col", variable=self.vars['show_rowcols']).pack(side='left', padx=10)

        # ── Buttons ──
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', padx=12, pady=10)

        self.btn_run = ttk.Button(btn_frame, text="  Run Simulation  ",
                                   style='Run.TButton', command=self._run)
        self.btn_run.pack(side='left', padx=5)

        self.btn_pause = ttk.Button(btn_frame, text="  Pause  ",
                                     style='Pause.TButton', command=self._pause, state='disabled')
        self.btn_pause.pack(side='left', padx=5)

        self.btn_stop = ttk.Button(btn_frame, text="  Stop  ",
                                    style='Stop.TButton', command=self._stop, state='disabled')
        self.btn_stop.pack(side='left', padx=5)

        # ── Log ──
        log_frame = ttk.LabelFrame(frame, text="  Log  ", padding=5)
        log_frame.pack(fill='both', expand=True, padx=12, pady=(5, 15))

        self.log_text = tk.Text(log_frame, height=10, wrap='word',
                                 bg=self._colors['LOG_BG'], fg=self._colors['FG'],
                                 insertbackground=self._colors['FG'],
                                 font=('Consolas', 9), relief='flat', borderwidth=0)
        log_scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scroll.pack(side='right', fill='y')

        self._log("Ready. Configure parameters and click Run.")

    def _add_row(self, parent, key, label, dtype):
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=2)
        ttk.Label(row, text=f"{label}:", width=24, anchor='e').pack(side='left')
        var = tk.StringVar(value=str(self.DEFAULTS[key]))
        self.vars[key] = (var, dtype)
        entry = ttk.Entry(row, textvariable=var, width=15)
        entry.pack(side='left', padx=(8, 0))

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch model", "*.pt *.pth"), ("All files", "*.*")],
            initialdir=PROJECT_ROOT,
        )
        if path:
            self.vars['model_path'].set(path)

    def _get_config(self):
        """Read all GUI values into a config dict."""
        cfg = {}
        for key, val in self.vars.items():
            if key == 'model_path':
                cfg[key] = val.get()
            elif isinstance(val, tk.BooleanVar):
                cfg[key] = val.get()
            else:
                var, dtype = val
                try:
                    cfg[key] = dtype(var.get())
                except ValueError:
                    messagebox.showerror("Invalid input",
                                         f"Invalid value for '{key}': {var.get()}")
                    return None
        return cfg

    def _log(self, msg):
        """Append message to log (thread-safe)."""
        def _append():
            self.log_text.insert('end', msg + '\n')
            self.log_text.see('end')
        self.root.after(0, _append)

    def _render_callback(self, pil_img):
        """Thread-safe callback to update simulation window."""
        def _update():
            if self.sim_window:
                self.sim_window.update_image(pil_img)
        self.root.after(0, _update)

    def _run(self):
        cfg = self._get_config()
        if cfg is None:
            return

        self.btn_run.configure(state='disabled')
        self.btn_pause.configure(state='normal')
        self.btn_stop.configure(state='normal')

        # Create simulation display window
        if self.sim_window:
            self.sim_window.top.destroy()
        
        self.sim_window = SimulatorWindow(
            self.root, 
            width=cfg['screen_width'], 
            height=cfg['screen_height']
        )

        self.runner = SimulationRunner(cfg, log_fn=self._log, render_fn=self._render_callback)

        def _thread():
            try:
                self.runner.run()
            except Exception as e:
                self._log(f"ERROR: {e}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                self.root.after(0, self._on_sim_done)

        self.sim_thread = threading.Thread(target=_thread, daemon=True)
        self.sim_thread.start()
        self._log("Simulation started...")

    def _pause(self):
        if self.runner:
            self.runner.toggle_pause()
            label = "Resume" if self.runner.paused else "Pause"
            self.btn_pause.configure(text=f"  {label}  ")
            self._log("PAUSED" if self.runner.paused else "RESUMED")

    def _stop(self):
        if self.runner:
            self.runner.stop()
            self._log("Stopping simulation...")

    def _on_sim_done(self):
        self.btn_run.configure(state='normal')
        self.btn_pause.configure(state='disabled', text="  Pause  ")
        self.btn_stop.configure(state='disabled')

    def _on_close(self):
        if self.runner:
            self.runner.stop()
        if self.sim_window:
            self.sim_window.top.destroy()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ConfigGUI()
    app.run()
