"""
╔══════════════════════════════════════════════════════════════╗
║       ENCHANTED RAILWAY KINGDOM - Flatland Visualizer       ║
║                   Fantasy 2D Edition                         ║
║                                                              ║
║  Visualize your trained PS-PPO / GNN model navigating        ║
║  trains through an enchanted railway network!                ║
╚══════════════════════════════════════════════════════════════╝

Controls:
  SPACE     - Play / Pause
  RIGHT     - Step forward (when paused)
  R         - Reset episode
  +/-       - Adjust speed
  SCROLL    - Zoom in/out
  DRAG      - Pan camera (middle mouse or hold SHIFT + left)
  F         - Fit to screen
  1-5       - Follow agent camera
  ESC/0     - Free camera
  TAB       - Toggle info panel
  Q         - Quit
"""

import os
import sys
import math
import time
import random
from collections import defaultdict

import numpy as np
import torch
import pygame

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fantasy_visualizer.assets import (
    FantasyAssets, ParticleSystem, COLORS, AGENT_COLORS,
    lerp_color, glow_surface, DIR_OFFSETS
)
from fantasy_visualizer.model_loader import StandaloneGNNPsPPO, TreeObsToGraph

# Flatland imports
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.step_utils.states import TrainState


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

CONFIG = {
    # Window
    'screen_width': 1400,
    'screen_height': 850,
    'fps': 60,

    # Cell rendering
    'cell_size': 48,
    'min_zoom': 0.3,
    'max_zoom': 3.0,

    # Environment
    'n_agents': 5,
    'x_dim': 25,
    'y_dim': 25,
    'n_cities': 3,
    'max_rails_between_cities': 2,
    'max_rails_in_city': 2,
    'seed': 42,
    'observation_tree_depth': 2,
    'observation_max_path_depth': 30,
    'malfunction_rate': 0.005,
    'malfunction_min_duration': 15,
    'malfunction_max_duration': 50,

    # Model
    'model_path': os.path.join(PROJECT_ROOT, 'curriculum_stage3.pt'),

    # Simulation
    'default_step_delay': 0.3,  # seconds between steps
    'min_step_delay': 0.05,
    'max_step_delay': 2.0,

    # UI Panel
    'panel_width': 320,
}


# ═══════════════════════════════════════════════════════════
# OBSERVATION NORMALIZATION (Simplified)
# ═══════════════════════════════════════════════════════════

class SimpleObservationNormalizer:
    """Simplified observation normalizer for inference."""

    def __init__(self, tree_depth=2):
        self.tree_depth = tree_depth
        self.n_features_per_node = 11
        self.n_nodes = sum([4 ** i for i in range(tree_depth + 1)])
        self.state_size = self.n_features_per_node * self.n_nodes

    def normalize(self, obs, rail_env=None, handle=None):
        """Normalize a tree observation to a flat vector."""
        if obs is None:
            return np.zeros(self.state_size)

        normalized = self._normalize_tree(obs, self.tree_depth)

        # Ensure correct size
        if len(normalized) < self.state_size:
            normalized = np.concatenate([normalized, np.zeros(self.state_size - len(normalized))])
        elif len(normalized) > self.state_size:
            normalized = normalized[:self.state_size]

        return normalized

    def _normalize_tree(self, node, depth):
        """Recursively normalize tree observation."""
        if node is None or depth < 0:
            return np.zeros(self.n_features_per_node * sum([4 ** i for i in range(depth + 1)]))

        features = self._extract_features(node)

        if depth == 0:
            return features

        child_features = []
        for child_key in ['L', 'F', 'R', 'B']:
            child = getattr(node, 'childs', {})
            if isinstance(child, dict) and child_key in child:
                child_features.append(self._normalize_tree(child[child_key], depth - 1))
            else:
                child_size = self.n_features_per_node * sum([4 ** i for i in range(depth)])
                child_features.append(np.zeros(child_size))

        return np.concatenate([features] + child_features)

    def _extract_features(self, node):
        """Extract features from a tree observation node."""
        features = np.zeros(self.n_features_per_node)
        if node is None or node == -np.inf or isinstance(node, (int, float)):
            return features

        try:
            features[0] = node.dist_own_target_encountered if not math.isinf(node.dist_own_target_encountered) else 0
            features[1] = node.dist_other_target_encountered if not math.isinf(node.dist_other_target_encountered) else 0
            features[2] = node.dist_other_agent_encountered if not math.isinf(node.dist_other_agent_encountered) else 0
            features[3] = node.dist_potential_conflict if not math.isinf(node.dist_potential_conflict) else 0
            features[4] = node.dist_unusable_switch if not math.isinf(node.dist_unusable_switch) else 0
            features[5] = node.dist_to_next_branch if not math.isinf(node.dist_to_next_branch) else 0
            features[6] = node.dist_min_to_target if not math.isinf(node.dist_min_to_target) else 0
            features[7] = node.num_agents_same_direction
            features[8] = node.num_agents_opposite_direction
            features[9] = node.num_agents_malfunctioning
            features[10] = node.speed_min_fractional
        except (AttributeError, TypeError):
            pass

        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)


# ═══════════════════════════════════════════════════════════
# ACTION MASKING (Simplified)
# ═══════════════════════════════════════════════════════════

def simple_action_mask(rail_env, agent_handle, action_size=5):
    """Simple action masking based on valid transitions."""
    # Action indices: DO_NOTHING=0, MOVE_LEFT=1, MOVE_FORWARD=2, MOVE_RIGHT=3, STOP_MOVING=4
    mask = [1] * action_size
    mask[0] = 0  # DO_NOTHING generally not allowed

    agent = rail_env.agents[agent_handle]
    if agent.position is None:
        return mask

    pos = agent.position
    direction = agent.direction

    for action in range(action_size):
        if action in [0, 4]:  # DO_NOTHING, STOP_MOVING
            continue

        if action == 1:  # MOVE_LEFT
            new_dir = (direction - 1) % 4
        elif action == 2:  # MOVE_FORWARD
            new_dir = direction
        elif action == 3:  # MOVE_RIGHT
            new_dir = (direction + 1) % 4
        else:
            continue

        # Flatland 4.x API: get_transitions takes ((row, col), direction)
        transitions = rail_env.rail.get_transitions((pos, direction))
        if transitions[new_dir] == 0:
            mask[action] = 0

    return mask


# ═══════════════════════════════════════════════════════════
# UI PANEL
# ═══════════════════════════════════════════════════════════

class UIPanel:
    """Fantasy-themed information panel."""

    def __init__(self, width, screen_height):
        self.width = width
        self.height = screen_height
        self.visible = True
        self.scroll_y = 0

    def toggle(self):
        self.visible = not self.visible

    def draw(self, surface, game_state):
        if not self.visible:
            return

        x = surface.get_width() - self.width
        panel = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Background
        panel.fill((12, 8, 25, 230))

        # Border
        pygame.draw.line(panel, COLORS['ui_border'], (0, 0), (0, self.height), 2)
        # Decorative top border
        pygame.draw.line(panel, COLORS['ui_accent'], (0, 0), (self.width, 0), 2)

        y = 15
        y = self._draw_title(panel, y)
        y = self._draw_separator(panel, y)
        y = self._draw_episode_info(panel, y, game_state)
        y = self._draw_separator(panel, y)
        y = self._draw_agents_info(panel, y, game_state)
        y = self._draw_separator(panel, y)
        y = self._draw_controls(panel, y)
        y = self._draw_separator(panel, y)
        y = self._draw_model_info(panel, y, game_state)

        surface.blit(panel, (x, 0))

    def _draw_title(self, panel, y):
        font_title = pygame.font.SysFont('Georgia', 18, bold=True)
        font_sub = pygame.font.SysFont('Georgia', 11)

        title = font_title.render("~ Enchanted Railway ~", True, COLORS['ui_accent'])
        panel.blit(title, (self.width // 2 - title.get_width() // 2, y))
        y += 25

        subtitle = font_sub.render("Kingdom of Flatland", True, COLORS['ui_text_dim'])
        panel.blit(subtitle, (self.width // 2 - subtitle.get_width() // 2, y))
        y += 20
        return y

    def _draw_separator(self, panel, y):
        y += 5
        sep_width = self.width - 20
        for i in range(sep_width):
            intensity = 1.0 - abs(i - sep_width / 2) / (sep_width / 2)
            color = lerp_color(COLORS['bg_dark'], COLORS['ui_border'], intensity * 0.7)
            try:
                panel.set_at((10 + i, y), color)
            except (IndexError, TypeError):
                pass
        y += 10
        return y

    def _draw_episode_info(self, panel, y, gs):
        font = pygame.font.SysFont('Consolas', 13)
        font_bold = pygame.font.SysFont('Consolas', 13, bold=True)

        # Status indicator
        status_text = "> RUNNING" if gs.get('playing') else "|| PAUSED"
        status_color = (100, 255, 100) if gs.get('playing') else (255, 200, 100)
        text = font_bold.render(status_text, True, status_color)
        panel.blit(text, (15, y))
        y += 20

        info_lines = [
            (f"Episode:    {gs.get('episode', 0)}", COLORS['ui_text']),
            (f"Step:       {gs.get('step', 0)} / {gs.get('max_steps', 0)}", COLORS['ui_text']),
            (f"Speed:      {gs.get('speed_label', '1x')}", COLORS['ui_text']),
            (f"Grid:       {gs.get('grid_w', 0)}×{gs.get('grid_h', 0)}", COLORS['ui_text_dim']),
            (f"Agents:     {gs.get('n_agents', 0)}", COLORS['ui_text_dim']),
        ]

        # Score
        score = gs.get('score', 0)
        score_color = (100, 255, 100) if score > 0.5 else (255, 200, 100) if score > 0 else (255, 100, 100)
        info_lines.append((f"Score:      {score:.2f}", score_color))

        completed = gs.get('completed', 0)
        total = gs.get('n_agents', 1)
        comp_color = (100, 255, 100) if completed == total else COLORS['ui_text']
        info_lines.append((f"Completed:  {completed}/{total}", comp_color))

        for text_str, color in info_lines:
            text = font.render(text_str, True, color)
            panel.blit(text, (15, y))
            y += 17

        return y

    def _draw_agents_info(self, panel, y, gs):
        font_header = pygame.font.SysFont('Consolas', 12, bold=True)
        font = pygame.font.SysFont('Consolas', 11)

        header = font_header.render("-- Agents --", True, COLORS['ui_accent'])
        panel.blit(header, (15, y))
        y += 18

        agents_info = gs.get('agents', [])
        for i, agent in enumerate(agents_info):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]

            # Color indicator
            pygame.draw.circle(panel, color, (22, y + 6), 5)
            pygame.draw.circle(panel, lerp_color(color, (255, 255, 255), 0.3), (22, y + 6), 3)

            # Status
            status = agent.get('status', 'unknown')
            status_icons = {
                'moving': ('>', (100, 255, 100)),
                'waiting': ('~', (255, 200, 80)),
                'done': ('*', (100, 255, 200)),
                'malfunction': ('!', (255, 80, 80)),
                'stopped': ('#', (200, 200, 200)),
                'ready': ('o', (100, 180, 255)),
            }
            icon, s_color = status_icons.get(status, ('?', COLORS['ui_text_dim']))

            info = f" Agent {i}: {status.upper():12s}"
            if agent.get('speed', 1.0) < 1.0:
                info += f" Spd:{agent['speed']:.1f}"

            text = font.render(info, True, s_color)
            panel.blit(text, (32, y))

            # Progress bar (distance to target)
            if 'progress' in agent:
                bar_x = 200
                bar_w = 90
                bar_h = 6
                progress = max(0, min(1, agent['progress']))
                pygame.draw.rect(panel, (40, 40, 40), (bar_x, y + 4, bar_w, bar_h), border_radius=2)
                if progress > 0:
                    prog_color = lerp_color((255, 80, 80), (80, 255, 80), progress)
                    pygame.draw.rect(panel, prog_color,
                                     (bar_x, y + 4, int(bar_w * progress), bar_h), border_radius=2)

            y += 18

        return y

    def _draw_controls(self, panel, y):
        font_header = pygame.font.SysFont('Consolas', 12, bold=True)
        font = pygame.font.SysFont('Consolas', 10)

        header = font_header.render("-- Controls --", True, COLORS['ui_accent'])
        panel.blit(header, (15, y))
        y += 16

        controls = [
            ("SPACE", "Play / Pause"),
            ("→", "Step forward"),
            ("R", "Reset episode"),
            ("+/-", "Speed up/down"),
            ("Scroll", "Zoom"),
            ("Shift+Drag", "Pan"),
            ("F", "Fit to screen"),
            ("1-5", "Follow agent"),
            ("TAB", "Toggle panel"),
            ("Q/ESC", "Quit"),
        ]

        for key, desc in controls:
            key_text = font.render(f"{key:>12s}", True, COLORS['ui_accent'])
            desc_text = font.render(f"  {desc}", True, COLORS['ui_text_dim'])
            panel.blit(key_text, (10, y))
            panel.blit(desc_text, (10 + key_text.get_width(), y))
            y += 14

        return y

    def _draw_model_info(self, panel, y, gs):
        font_header = pygame.font.SysFont('Consolas', 12, bold=True)
        font = pygame.font.SysFont('Consolas', 10)

        header = font_header.render("-- Model --", True, COLORS['ui_accent'])
        panel.blit(header, (15, y))
        y += 16

        model_info = [
            f"Type: GNN PS-PPO (GAT)",
            f"GAT Layers: 2",
            f"Heads: 4, Hidden: 128",
            f"Actions: 5 (L/F/R/Stop/Noop)",
            f"File: curriculum_stage3.pt",
        ]

        for line in model_info:
            text = font.render(line, True, COLORS['ui_text_dim'])
            panel.blit(text, (15, y))
            y += 14

        return y


# ═══════════════════════════════════════════════════════════
# CAMERA
# ═══════════════════════════════════════════════════════════

class Camera:
    """Smooth camera with zoom and pan."""

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.x = 0.0
        self.y = 0.0
        self.zoom = 1.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_zoom = 1.0
        self.follow_agent = -1  # -1 = free camera
        self.smoothness = 0.1

    def fit_to_grid(self, grid_w, grid_h, cell_size, panel_width=0):
        """Fit the camera to show the entire grid."""
        available_w = self.screen_w - panel_width
        zoom_x = available_w / (grid_w * cell_size + 40)
        zoom_y = self.screen_h / (grid_h * cell_size + 40)
        self.target_zoom = min(zoom_x, zoom_y, CONFIG['max_zoom'])

        total_grid_w = grid_w * cell_size * self.target_zoom
        total_grid_h = grid_h * cell_size * self.target_zoom

        self.target_x = (available_w - total_grid_w) / 2
        self.target_y = (self.screen_h - total_grid_h) / 2

    def update(self, dt, agents=None, cell_size=48):
        """Update camera with smooth interpolation."""
        if self.follow_agent >= 0 and agents is not None and self.follow_agent < len(agents):
            agent = agents[self.follow_agent]
            if agent.position is not None:
                ax = agent.position[1] * cell_size * self.zoom + cell_size * self.zoom / 2
                ay = agent.position[0] * cell_size * self.zoom + cell_size * self.zoom / 2
                self.target_x = self.screen_w / 2 - ax
                self.target_y = self.screen_h / 2 - ay

        t = min(1.0, self.smoothness * dt * 60)
        self.x += (self.target_x - self.x) * t
        self.y += (self.target_y - self.y) * t
        self.zoom += (self.target_zoom - self.zoom) * t

    def world_to_screen(self, wx, wy):
        return (int(wx * self.zoom + self.x),
                int(wy * self.zoom + self.y))

    def screen_to_world(self, sx, sy):
        return ((sx - self.x) / self.zoom,
                (sy - self.y) / self.zoom)

    def apply_zoom(self, delta, mouse_x, mouse_y):
        old_zoom = self.target_zoom
        self.target_zoom *= 1.1 ** delta
        self.target_zoom = max(CONFIG['min_zoom'], min(CONFIG['max_zoom'], self.target_zoom))

        # Zoom toward mouse position
        ratio = self.target_zoom / old_zoom
        self.target_x = mouse_x - (mouse_x - self.target_x) * ratio
        self.target_y = mouse_y - (mouse_y - self.target_y) * ratio


# ═══════════════════════════════════════════════════════════
# MAIN VISUALIZER
# ═══════════════════════════════════════════════════════════

class FlatlandFantasyVisualizer:
    """Main visualizer application."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("⚔ Enchanted Railway Kingdom ⚔ - Flatland Visualizer")

        self.screen = pygame.display.set_mode(
            (CONFIG['screen_width'], CONFIG['screen_height']),
            pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()

        # Assets
        self.assets = FantasyAssets(CONFIG['cell_size'])
        self.particles = ParticleSystem()
        self.ui_panel = UIPanel(CONFIG['panel_width'], CONFIG['screen_height'])
        self.camera = Camera(CONFIG['screen_width'], CONFIG['screen_height'])

        # State
        self.running = True
        self.playing = False
        self.step_delay = CONFIG['default_step_delay']
        self.last_step_time = 0
        self.step_count = 0
        self.episode_count = 0
        self.done_all = False
        self.total_reward = 0

        # Dragging
        self.dragging = False
        self.drag_start = (0, 0)

        # Initialize environment and model
        self._init_environment()
        self._init_model()

        # Fit camera
        panel_w = CONFIG['panel_width'] if self.ui_panel.visible else 0
        self.camera.fit_to_grid(self.env.width, self.env.height, CONFIG['cell_size'], panel_w)
        # Snap camera immediately
        self.camera.x = self.camera.target_x
        self.camera.y = self.camera.target_y
        self.camera.zoom = self.camera.target_zoom

        # Agent tracking
        self.agent_prev_pos = {}
        self.agent_actions = {}

    def _init_environment(self):
        """Initialize the Flatland environment."""
        print("[ENV] Initializing Flatland environment...")

        predictor = ShortestPathPredictorForRailEnv(CONFIG['observation_max_path_depth'])
        self.tree_obs = TreeObsForRailEnv(max_depth=CONFIG['observation_tree_depth'], predictor=predictor)

        self.env = RailEnv(
            width=CONFIG['x_dim'],
            height=CONFIG['y_dim'],
            rail_generator=sparse_rail_generator(
                max_num_cities=CONFIG['n_cities'],
                grid_mode=False,
                max_rails_between_cities=CONFIG['max_rails_between_cities'],
                max_rail_pairs_in_city=CONFIG['max_rails_in_city'],
                seed=CONFIG['seed']
            ),
            line_generator=sparse_line_generator(),
            number_of_agents=CONFIG['n_agents'],
            malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(
                malfunction_rate=CONFIG['malfunction_rate'],
                min_duration=CONFIG['malfunction_min_duration'],
                max_duration=CONFIG['malfunction_max_duration']
            )),
            obs_builder_object=self.tree_obs,
            random_seed=CONFIG['seed']
        )

        self.obs_normalizer = SimpleObservationNormalizer(CONFIG['observation_tree_depth'])
        self.max_steps = int(4 * 2 * (CONFIG['y_dim'] + CONFIG['x_dim'] + CONFIG['n_agents'] / CONFIG['n_cities']))

        self._reset_episode()
        print(f"[ENV] Grid: {self.env.width}x{self.env.height}, Agents: {len(self.env.agents)}, Max steps: {self.max_steps}")

    def _init_model(self):
        """Initialize and load the GNN model."""
        print("[MODEL] Loading GNNPsPPO model...")

        self.model = StandaloneGNNPsPPO(
            action_size=5,
            hidden_dim=128,
            n_heads=4,
            n_layers=2
        )

        model_path = CONFIG['model_path']
        if os.path.exists(model_path):
            self.model.load_checkpoint(model_path)
            print("[MODEL] Model loaded successfully!")
        else:
            print(f"[MODEL] WARNING: Model file not found at {model_path}")
            print("[MODEL] Running with random policy!")

    def _reset_episode(self):
        """Reset the environment for a new episode."""
        self.obs, self.info = self.env.reset()
        self.dones = {a: False for a in range(len(self.env.agents))}
        self.dones['__all__'] = False
        self.step_count = 0
        self.episode_count += 1
        self.done_all = False
        self.total_reward = 0
        self.agent_prev_pos = {}
        self.agent_actions = {}

        # Compute agent IDs for PS-PPO
        self.agent_ids = {}
        max_speed = 1.0
        max_mal = 0.1 * 10
        max_val = max_speed + max_mal
        for a in self.env.agents:
            self.agent_ids[a.handle] = (a.speed_counter.speed + CONFIG['malfunction_rate'] * 10) / max_val

        print(f"[ENV] Episode {self.episode_count} started")

    def _step_environment(self):
        """Perform one environment step using the model."""
        if self.done_all or self.step_count >= self.max_steps:
            return

        action_dict = {}

        for agent_handle in range(len(self.env.agents)):
            if self.dones.get(agent_handle, False):
                continue

            if self.obs.get(agent_handle) is None:
                continue

            # Normalize observation
            normalized_obs = self.obs_normalizer.normalize(self.obs[agent_handle])

            # Append agent ID (PS-PPO style)
            agent_id_val = self.agent_ids.get(agent_handle, 0.5)
            full_obs = np.append(normalized_obs, [agent_id_val])

            # Get action mask
            mask = simple_action_mask(self.env, agent_handle)

            # Get action from model
            action = self.model.act(full_obs, mask)
            action_dict[agent_handle] = action
            self.agent_actions[agent_handle] = action

        # Track previous positions
        for agent in self.env.agents:
            if agent.position is not None:
                self.agent_prev_pos[agent.handle] = agent.position

        # Step environment
        self.obs, rewards, self.dones, self.info = self.env.step(action_dict)

        # Update rewards
        for a, r in rewards.items():
            if isinstance(a, int):
                self.total_reward += r

        self.step_count += 1

        # Emit particles for moving agents
        for agent in self.env.agents:
            if agent.position is not None and agent.handle in self.agent_prev_pos:
                prev = self.agent_prev_pos[agent.handle]
                if prev != agent.position:
                    px = agent.position[1] * CONFIG['cell_size'] + CONFIG['cell_size'] // 2
                    py = agent.position[0] * CONFIG['cell_size'] + CONFIG['cell_size'] // 2
                    color = AGENT_COLORS[agent.handle % len(AGENT_COLORS)]
                    self.particles.emit(
                        px * self.camera.zoom + self.camera.x,
                        py * self.camera.zoom + self.camera.y,
                        color, count=5
                    )

        if self.dones.get('__all__', False):
            self.done_all = True
            print(f"[ENV] Episode {self.episode_count} finished at step {self.step_count}")

    def _handle_events(self):
        """Handle user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    self.camera.apply_zoom(2, event.pos[0], event.pos[1])
                elif event.button == 5:  # Scroll down
                    self.camera.apply_zoom(-2, event.pos[0], event.pos[1])
                elif event.button == 2 or (event.button == 1 and pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    self.dragging = True
                    self.drag_start = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in [1, 2]:
                    self.dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    dx = event.pos[0] - self.drag_start[0]
                    dy = event.pos[1] - self.drag_start[1]
                    self.camera.target_x += dx
                    self.camera.target_y += dy
                    self.camera.x += dx
                    self.camera.y += dy
                    self.drag_start = event.pos

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                self.camera.apply_zoom(event.y * 2, mx, my)

            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.camera.screen_w = event.w
                self.camera.screen_h = event.h
                self.ui_panel.height = event.h

    def _handle_keydown(self, event):
        """Handle keyboard shortcuts."""
        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
            self.running = False

        elif event.key == pygame.K_SPACE:
            self.playing = not self.playing

        elif event.key == pygame.K_RIGHT:
            if not self.playing:
                self._step_environment()

        elif event.key == pygame.K_r:
            self._reset_episode()

        elif event.key in [pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS]:
            self.step_delay = max(CONFIG['min_step_delay'], self.step_delay * 0.7)

        elif event.key in [pygame.K_MINUS, pygame.K_KP_MINUS]:
            self.step_delay = min(CONFIG['max_step_delay'], self.step_delay * 1.4)

        elif event.key == pygame.K_f:
            panel_w = CONFIG['panel_width'] if self.ui_panel.visible else 0
            self.camera.fit_to_grid(self.env.width, self.env.height, CONFIG['cell_size'], panel_w)
            self.camera.follow_agent = -1

        elif event.key == pygame.K_TAB:
            self.ui_panel.toggle()

        elif event.key == pygame.K_0:
            self.camera.follow_agent = -1

        elif pygame.K_1 <= event.key <= pygame.K_9:
            idx = event.key - pygame.K_1
            if idx < len(self.env.agents):
                self.camera.follow_agent = idx

    def _get_game_state(self):
        """Collect current game state for UI."""
        speed_factor = CONFIG['default_step_delay'] / self.step_delay
        if speed_factor >= 1:
            speed_label = f"{speed_factor:.1f}x"
        else:
            speed_label = f"{speed_factor:.2f}x"

        completed = sum(1 for d in self.dones.items()
                        if isinstance(d[0], int) and d[1])

        agents_info = []
        for i, agent in enumerate(self.env.agents):
            info = {'speed': agent.speed_counter.speed}

            if agent.state == TrainState.DONE:
                info['status'] = 'done'
                info['progress'] = 1.0
            elif agent.state == TrainState.MOVING:
                if agent.malfunction_handler.malfunction_down_counter > 0:
                    info['status'] = 'malfunction'
                else:
                    info['status'] = 'moving'
                # Estimate progress
                if agent.position is not None and agent.target is not None:
                    dist = abs(agent.position[0] - agent.target[0]) + abs(agent.position[1] - agent.target[1])
                    max_dist = self.env.width + self.env.height
                    info['progress'] = max(0, 1 - dist / max_dist)
                else:
                    info['progress'] = 0
            elif agent.state == TrainState.WAITING:
                info['status'] = 'waiting'
                info['progress'] = 0
            elif agent.state == TrainState.READY_TO_DEPART:
                info['status'] = 'ready'
                info['progress'] = 0
            else:
                info['status'] = 'stopped'
                info['progress'] = 0

            agents_info.append(info)

        return {
            'playing': self.playing,
            'episode': self.episode_count,
            'step': self.step_count,
            'max_steps': self.max_steps,
            'speed_label': speed_label,
            'grid_w': self.env.width,
            'grid_h': self.env.height,
            'n_agents': len(self.env.agents),
            'score': self.total_reward / max(1, self.step_count * len(self.env.agents)),
            'completed': completed,
            'agents': agents_info,
        }

    def _render(self):
        """Render the entire scene."""
        sw, sh = self.screen.get_size()
        cs = CONFIG['cell_size']

        # Draw background
        self.assets.draw_background(self.screen, self.camera.x, self.camera.y, sw, sh)

        # Create world surface for grid rendering
        # Only render visible cells for performance
        cam = self.camera

        # Determine visible cell range
        wx1, wy1 = cam.screen_to_world(0, 0)
        wx2, wy2 = cam.screen_to_world(sw, sh)

        col_start = max(0, int(wx1 / cs) - 1)
        col_end = min(self.env.width, int(wx2 / cs) + 2)
        row_start = max(0, int(wy1 / cs) - 1)
        row_end = min(self.env.height, int(wy2 / cs) + 2)

        # Draw ground tiles
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                sx, sy = cam.world_to_screen(col * cs, row * cs)
                scaled_cs = int(cs * cam.zoom)
                if scaled_cs < 2:
                    continue

                has_rail = self.env.rail.grid[row, col] != 0

                # Draw ground
                ground_surf = pygame.Surface((scaled_cs, scaled_cs))
                tile_idx = (col * 7 + row * 13) % len(self.assets._ground_tiles)
                pygame.transform.scale(self.assets._ground_tiles[tile_idx], (scaled_cs, scaled_cs), ground_surf)
                self.screen.blit(ground_surf, (sx, sy))

                # Rail glow under tracks
                if has_rail:
                    glow = glow_surface(int(scaled_cs * 1.2), COLORS['rail_glow'], 0.15)
                    glow_rect = glow.get_rect(center=(sx + scaled_cs // 2, sy + scaled_cs // 2))
                    self.screen.blit(glow, glow_rect, special_flags=pygame.BLEND_ADD)

        # Draw rails
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                transitions = self.env.rail.grid[row, col]
                if transitions == 0:
                    continue

                sx, sy = cam.world_to_screen(col * cs, row * cs)
                scaled_cs = int(cs * cam.zoom)
                if scaled_cs < 4:
                    continue

                self._draw_rail_cell(sx, sy, scaled_cs, transitions, row + col * 0.3)

        # Draw targets (below agents)
        for i, agent in enumerate(self.env.agents):
            if agent.target is not None:
                is_done = self.dones.get(i, False)
                tx, ty = cam.world_to_screen(agent.target[1] * cs, agent.target[0] * cs)
                scaled_cs = int(cs * cam.zoom)
                if scaled_cs >= 4:
                    self._draw_target_scaled(tx, ty, scaled_cs, i, is_done)

        # Draw agents
        for i, agent in enumerate(self.env.agents):
            pos = agent.position
            if pos is None:
                # Show at initial position if waiting
                if agent.state in [TrainState.WAITING, TrainState.READY_TO_DEPART]:
                    pos = agent.initial_position
                if pos is None:
                    continue

            ax, ay = cam.world_to_screen(pos[1] * cs, pos[0] * cs)
            scaled_cs = int(cs * cam.zoom)
            if scaled_cs < 4:
                continue

            is_done = agent.state == TrainState.DONE
            is_malfunction = agent.malfunction_handler.malfunction_down_counter > 0
            is_moving = agent.state == TrainState.MOVING

            self._draw_agent_scaled(ax, ay, scaled_cs, agent.direction, i,
                                    is_moving, is_malfunction, is_done, agent.speed_counter.speed)

        # Draw particles
        self.particles.draw(self.screen)

        # Draw "ALL DONE" banner if episode finished
        if self.done_all:
            self._draw_completion_banner()

        # Draw UI panel
        game_state = self._get_game_state()
        self.ui_panel.draw(self.screen, game_state)

        # Draw mini status bar
        self._draw_status_bar()

        pygame.display.flip()

    def _draw_rail_cell(self, sx, sy, scaled_cs, transitions, time_seed):
        """Draw rail tracks for a cell at screen coordinates."""
        half = scaled_cs // 2
        center = (sx + half, sy + half)

        dir_points = {
            0: (sx + half, sy),            # North
            1: (sx + scaled_cs, sy + half), # East
            2: (sx + half, sy + scaled_cs), # South
            3: (sx, sy + half),             # West
        }

        drawn = set()
        connected_dirs = set()

        for from_dir in range(4):
            for to_dir in range(4):
                bit_index = from_dir * 4 + to_dir
                if transitions & (1 << (15 - bit_index)):
                    pair = tuple(sorted([from_dir, to_dir]))
                    connected_dirs.add(from_dir)
                    connected_dirs.add(to_dir)

                    if pair in drawn:
                        continue
                    drawn.add(pair)

                    p1 = dir_points[from_dir]
                    p2 = dir_points[to_dir] if from_dir != to_dir else center

                    # Rail bed
                    width_bed = max(3, scaled_cs // 7)
                    pygame.draw.line(self.screen, COLORS['rail_sleeper'], p1, p2, width_bed)

                    # Rail core with shimmer
                    glow_phase = 0.7 + 0.3 * math.sin(self.assets._time * 2.0 + time_seed)
                    rail_color = lerp_color(COLORS['rail_dim'], COLORS['rail_core'], glow_phase)
                    width_core = max(2, scaled_cs // 12)
                    pygame.draw.line(self.screen, rail_color, p1, p2, width_core)

                    # Highlight
                    highlight = lerp_color(COLORS['rail_core'], COLORS['rail_glow'], glow_phase * 0.4)
                    pygame.draw.line(self.screen, highlight, p1, p2, max(1, scaled_cs // 24))

        # Junction crystal
        if len(connected_dirs) > 2:
            glow_phase = 0.5 + 0.5 * math.sin(self.assets._time * 3.0 + time_seed)
            junc_color = lerp_color(COLORS['rail_core'], COLORS['station_crystal'], glow_phase)
            junc_r = max(2, scaled_cs // 10)
            pygame.draw.circle(self.screen, junc_color, center, junc_r)
            pygame.draw.circle(self.screen, COLORS['rail_glow'], center, max(1, junc_r // 2))

    def _draw_target_scaled(self, tx, ty, scaled_cs, agent_idx, is_reached):
        """Draw a scaled target marker."""
        half = scaled_cs // 2
        cx, cy = tx + half, ty + half
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        pulse = 0.5 + 0.5 * math.sin(self.assets._time * 2.5 + agent_idx * 1.3)

        if not is_reached:
            # Glow
            glow_size = int(scaled_cs * 0.9 + scaled_cs * 0.15 * pulse)
            glow = glow_surface(glow_size, color, 0.35 + 0.15 * pulse)
            glow_rect = glow.get_rect(center=(cx, cy))
            self.screen.blit(glow, glow_rect, special_flags=pygame.BLEND_ADD)

            # Crystal
            crystal_h = int(scaled_cs * 0.35)
            crystal_w = int(scaled_cs * 0.2)
            points = [
                (cx, cy - crystal_h),
                (cx + crystal_w, cy),
                (cx, cy + crystal_h // 3),
                (cx - crystal_w, cy),
            ]
            shadow = [(p[0] + 1, p[1] + 1) for p in points]
            pygame.draw.polygon(self.screen, (0, 0, 0), shadow)
            pygame.draw.polygon(self.screen, color, points)

            highlight = lerp_color(color, (255, 255, 255), 0.5)
            inner = [(cx, cy - crystal_h + 3), (cx + crystal_w - 3, cy),
                     (cx, cy + crystal_h // 3 - 2), (cx - crystal_w + 3, cy)]
            if crystal_w > 4:
                pygame.draw.polygon(self.screen, highlight, inner)

            # Floating number
            bounce = math.sin(self.assets._time * 2 + agent_idx) * 3
            font_size = max(8, scaled_cs // 3)
            font = pygame.font.SysFont('Arial', font_size, bold=True)
            text = font.render(str(agent_idx), True, COLORS['spark_white'])
            text_rect = text.get_rect(center=(cx, cy - crystal_h - 8 + bounce))
            self.screen.blit(text, text_rect)
        else:
            glow = glow_surface(scaled_cs, lerp_color(color, COLORS['spark_gold'], 0.5), 0.2)
            self.screen.blit(glow, glow.get_rect(center=(cx, cy)), special_flags=pygame.BLEND_ADD)

    def _draw_agent_scaled(self, ax, ay, scaled_cs, direction, idx,
                           is_moving, is_malfunction, is_done, speed):
        """Draw a scaled agent."""
        half = scaled_cs // 2
        cx, cy = ax + half, ay + half
        color = AGENT_COLORS[idx % len(AGENT_COLORS)]

        if is_done:
            # Completion sparkles
            n = 8
            for i in range(n):
                angle = self.assets._time * 2 + i * (math.pi * 2 / n)
                dist = scaled_cs * 0.3 + scaled_cs * 0.1 * math.sin(self.assets._time * 4 + i)
                sx = cx + math.cos(angle) * dist
                sy = cy + math.sin(angle) * dist
                c = lerp_color(color, COLORS['spark_gold'], 0.5 + 0.5 * math.sin(self.assets._time * 5 + i))
                size = max(1, int(2 + math.sin(self.assets._time * 6 + i * 0.7)))
                pygame.draw.circle(self.screen, c, (int(sx), int(sy)), size)
            # Center star
            star_pulse = 0.5 + 0.5 * math.sin(self.assets._time * 3)
            star_c = lerp_color(COLORS['spark_gold'], COLORS['spark_white'], star_pulse)
            pygame.draw.circle(self.screen, star_c, (cx, cy), max(2, scaled_cs // 8))
            return

        # Glow
        glow_size = int(scaled_cs * 0.9)
        glow = glow_surface(glow_size, color, 0.4)
        self.screen.blit(glow, glow.get_rect(center=(cx, cy)), special_flags=pygame.BLEND_ADD)

        # Body
        body_w = int(scaled_cs * 0.7)
        body_h = int(scaled_cs * 0.4)

        angle_deg = {0: 90, 1: 0, 2: 270, 3: 180}.get(direction, 0)

        train_surf = pygame.Surface((body_w + 10, body_h + 10), pygame.SRCALPHA)
        tw, th = train_surf.get_size()
        tcx, tcy = tw // 2, th // 2

        # Shadow
        shadow_rect = pygame.Rect(tcx - body_w // 2 + 1, tcy - body_h // 2 + 1, body_w, body_h)
        pygame.draw.rect(train_surf, (0, 0, 0, 80), shadow_rect, border_radius=4)

        # Main body
        body_rect = pygame.Rect(tcx - body_w // 2, tcy - body_h // 2, body_w, body_h)
        pygame.draw.rect(train_surf, color, body_rect, border_radius=4)

        # Highlight
        hl = lerp_color(color, (255, 255, 255), 0.3)
        hl_rect = pygame.Rect(tcx - body_w // 2 + 2, tcy - body_h // 2 + 1, body_w - 4, body_h // 3)
        pygame.draw.rect(train_surf, hl, hl_rect, border_radius=2)

        # Front light
        front_x = tcx + body_w // 2 - 3
        pygame.draw.circle(train_surf, COLORS['rail_glow'], (front_x, tcy), max(2, body_h // 6))

        # Windows
        win_c = (200, 230, 255, 220)
        for i in range(2):
            wx = tcx - body_w // 4 + i * (body_w // 3)
            pygame.draw.rect(train_surf, win_c, (wx - 2, tcy - 2, 5, 4), border_radius=1)

        # Malfunction overlay
        if is_malfunction:
            pulse = 0.5 + 0.5 * math.sin(self.assets._time * 8.0)
            mal_c = (255, 50, 50, int(pulse * 120))
            mal_surf = pygame.Surface((body_w + 6, body_h + 6), pygame.SRCALPHA)
            pygame.draw.rect(mal_surf, mal_c, (0, 0, body_w + 6, body_h + 6), border_radius=5)
            train_surf.blit(mal_surf, (tcx - body_w // 2 - 3, tcy - body_h // 2 - 3))

            # Lightning bolt icon
            bolt_color = (255, 255, 100)
            bolt_points = [(tcx, tcy - body_h // 2 - 4),
                           (tcx - 3, tcy - 1),
                           (tcx + 1, tcy - 1),
                           (tcx - 2, tcy + body_h // 2 + 2)]
            pygame.draw.lines(train_surf, bolt_color, False, bolt_points, 2)

        # Rotate
        rotated = pygame.transform.rotate(train_surf, angle_deg)
        rot_rect = rotated.get_rect(center=(cx, cy))
        self.screen.blit(rotated, rot_rect)

        # Badge
        badge_r = max(6, scaled_cs // 5)
        badge_y = cy - half + badge_r // 2
        pygame.draw.circle(self.screen, (0, 0, 0), (cx, badge_y), badge_r + 1)
        pygame.draw.circle(self.screen, color, (cx, badge_y), badge_r)
        font = pygame.font.SysFont('Arial', max(8, badge_r + 2), bold=True)
        text = font.render(str(idx), True, (255, 255, 255))
        self.screen.blit(text, text.get_rect(center=(cx, badge_y)))

        # Speed indicator for slow agents
        if speed < 1.0:
            spd_font = pygame.font.SysFont('Arial', max(7, scaled_cs // 6))
            spd_text = spd_font.render(f"x{speed:.1f}", True, (255, 200, 100))
            self.screen.blit(spd_text, spd_text.get_rect(center=(cx, cy + half - 4)))

    def _draw_completion_banner(self):
        """Draw the episode completion banner."""
        sw, sh = self.screen.get_size()
        banner_h = 80
        banner_surf = pygame.Surface((sw, banner_h), pygame.SRCALPHA)

        # Gradient background
        for y in range(banner_h):
            t = 1.0 - abs(y - banner_h / 2) / (banner_h / 2)
            alpha = int(t * 200)
            pulse = 0.5 + 0.5 * math.sin(self.assets._time * 2)
            bg_color = lerp_color((20, 10, 40), (40, 20, 60), pulse)
            banner_surf.fill((*bg_color, alpha), (0, y, sw, 1))

        # Border lines
        gold = COLORS['ui_accent']
        pygame.draw.line(banner_surf, gold, (0, 0), (sw, 0), 2)
        pygame.draw.line(banner_surf, gold, (0, banner_h - 1), (sw, banner_h - 1), 2)

        # Text
        font_big = pygame.font.SysFont('Georgia', 28, bold=True)
        font_sub = pygame.font.SysFont('Georgia', 14)

        completed = sum(1 for i in range(len(self.env.agents)) if self.dones.get(i, False))
        total = len(self.env.agents)

        if completed == total:
            title = "*** ALL TRAINS ARRIVED ***"
            title_color = COLORS['spark_gold']
        else:
            title = f"Episode Complete - {completed}/{total} Arrived"
            title_color = COLORS['ui_text']

        title_surf = font_big.render(title, True, title_color)
        banner_surf.blit(title_surf, (sw // 2 - title_surf.get_width() // 2, 15))

        sub = f"Steps: {self.step_count}  |  Press R to restart  |  Press SPACE to auto-restart"
        sub_surf = font_sub.render(sub, True, COLORS['ui_text_dim'])
        banner_surf.blit(sub_surf, (sw // 2 - sub_surf.get_width() // 2, 50))

        self.screen.blit(banner_surf, (0, sh // 2 - banner_h // 2))

    def _draw_status_bar(self):
        """Draw a minimal status bar at the bottom."""
        sw, sh = self.screen.get_size()
        bar_h = 24
        bar = pygame.Surface((sw, bar_h), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 160))

        font = pygame.font.SysFont('Consolas', 11)

        # Left: step info
        step_text = f" Step {self.step_count}/{self.max_steps} | Episode {self.episode_count}"
        text = font.render(step_text, True, COLORS['ui_text_dim'])
        bar.blit(text, (5, 4))

        # Center: agent status dots
        dot_x = sw // 2 - len(self.env.agents) * 14
        for i, agent in enumerate(self.env.agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            if self.dones.get(i, False):
                pygame.draw.circle(bar, COLORS['spark_gold'], (dot_x, bar_h // 2), 5)
                pygame.draw.circle(bar, color, (dot_x, bar_h // 2), 3)
            elif agent.state == TrainState.MOVING:
                pygame.draw.circle(bar, color, (dot_x, bar_h // 2), 5)
            else:
                pygame.draw.circle(bar, lerp_color(color, (60, 60, 60), 0.5), (dot_x, bar_h // 2), 5, 1)
            dot_x += 28

        # Right: zoom info
        zoom_text = f"Zoom: {self.camera.zoom:.1f}x "
        text = font.render(zoom_text, True, COLORS['ui_text_dim'])
        bar.blit(text, (sw - text.get_width() - 5, 4))

        self.screen.blit(bar, (0, sh - bar_h))

    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("   ENCHANTED RAILWAY KINGDOM - Flatland Visualizer")
        print("   Press SPACE to start, TAB for help panel")
        print("=" * 60 + "\n")

        while self.running:
            dt = self.clock.tick(CONFIG['fps']) / 1000.0

            self._handle_events()

            # Auto-step when playing
            if self.playing and not self.done_all:
                now = time.time()
                if now - self.last_step_time >= self.step_delay:
                    self._step_environment()
                    self.last_step_time = now

            # Auto-reset if done and playing
            if self.playing and self.done_all:
                now = time.time()
                if now - self.last_step_time >= 2.0:  # 2s pause before reset
                    self._reset_episode()
                    self.last_step_time = now

            # Update
            self.camera.update(dt, self.env.agents, CONFIG['cell_size'])
            self.assets._time += dt
            self.particles.update(dt)

            # Render
            self._render()

        pygame.quit()
        print("[APP] Goodbye!")


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = FlatlandFantasyVisualizer()
    app.run()
