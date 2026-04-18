
import os
import sys
import math
import io
import base64
import numpy as np
import torch
from flask import Flask, render_template, jsonify, request

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fantasy_visualizer.model_loader import StandaloneGNNPsPPO
from fantasy_visualizer.main import SimpleObservationNormalizer, simple_action_mask

# Flatland imports
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool

# Fantasy Visualizer Assets
import pygame
from fantasy_visualizer.assets import FantasyAssets, COLORS, AGENT_COLORS, lerp_color, glow_surface

app = Flask(__name__)

# Global state for the simulation
class SimulationState:
    def __init__(self):
        self.env = None
        self.model = None
        self.obs = None
        self.dones = {}
        self.step_count = 0
        self.max_steps = 0
        self.total_reward = 0
        self.agent_ids = {}
        self.obs_normalizer = None
        self.renderer = None
        
        # Performance: Pre-initialize Pygame for headless rendering
        if not pygame.get_init():
            pygame.init()
        self.assets = None
        self.render_surf = None
        self.cell_size = 48
        self.render_time = 0
        self.config = {
            'n_agents': 2,
            'x_dim': 35,
            'y_dim': 35,
            'n_cities': 2,
            'max_rails_between_cities': 2,
            'max_rails_in_city': 3,
            'seed': 42,
            'malfunction_rate': 0.005,
            'malfunction_min': 15,
            'malfunction_max': 50,
            'obs_tree_depth': 2,
            'obs_max_path_depth': 30,
            'model_path': os.path.join(PROJECT_ROOT, 'curriculum_stage3.pt')
        }

    def init_env(self, config=None):
        if config:
            self.config.update(config)
        
        predictor = ShortestPathPredictorForRailEnv(self.config['obs_max_path_depth'])
        tree_obs = TreeObsForRailEnv(max_depth=self.config['obs_tree_depth'], predictor=predictor)
        
        self.env = RailEnv(
            width=self.config['x_dim'],
            height=self.config['y_dim'],
            rail_generator=sparse_rail_generator(
                max_num_cities=self.config['n_cities'],
                grid_mode=False,
                max_rails_between_cities=self.config['max_rails_between_cities'],
                max_rail_pairs_in_city=self.config['max_rails_in_city'],
                seed=self.config['seed']
            ),
            line_generator=sparse_line_generator(),
            number_of_agents=self.config['n_agents'],
            malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(
                malfunction_rate=self.config['malfunction_rate'],
                min_duration=self.config['malfunction_min'],
                max_duration=self.config['malfunction_max']
            )),
            obs_builder_object=tree_obs,
            random_seed=self.config['seed']
        )
        
        self.obs_normalizer = SimpleObservationNormalizer(self.config['obs_tree_depth'])
        self.max_steps = int(4 * 2 * (self.config['y_dim'] + self.config['x_dim'] + self.config['n_agents'] / max(1, self.config['n_cities'])))
        
        # Initialize assets and surface
        self.assets = FantasyAssets(self.cell_size)
        surf_w = self.config['x_dim'] * self.cell_size
        surf_h = self.config['y_dim'] * self.cell_size
        self.render_surf = pygame.Surface((surf_w, surf_h))
        
        # Reset environment to initialize internal attributes
        self.reset()
        
        # Initialize Native Renderer after reset
        try:
            self.renderer = RenderTool(self.env, gl="PIL")
        except Exception as e:
            print(f"Failed to initialize RenderTool: {e}")
            self.renderer = None
        
    def reset(self):
        self.obs, _ = self.env.reset()
        
        # Ensure self.dones and env.dones are in sync
        # Some versions of Flatland initialize dones in reset, others don't
        if not hasattr(self.env, 'dones') or self.env.dones is None:
            self.env.dones = {a: False for a in range(len(self.env.agents))}
        
        if '__all__' not in self.env.dones:
            self.env.dones['__all__'] = False
        
        self.dones = self.env.dones
        self.step_count = 0
        self.total_reward = 0
        
        if self.renderer:
            try:
                self.renderer.set_new_rail()
            except:
                pass
        
        # Agent IDs for PS-PPO
        self.agent_ids = {}
        for a in self.env.agents:
            self.agent_ids[a.handle] = (a.speed_counter.speed + 0.005 * 10) / 1.05
            
        if self.assets:
            self.render_fantasy()

    def load_model(self):
        self.model = StandaloneGNNPsPPO(action_size=5, hidden_dim=128, n_heads=4, n_layers=2)
        if os.path.exists(self.config['model_path']):
            self.model.load_checkpoint(self.config['model_path'])
            return True
        return False

    def step(self):
        if self.dones.get('__all__', False) or self.step_count >= self.max_steps:
            return self.get_state()

        action_dict = {}
        for agent_handle in range(len(self.env.agents)):
            if self.dones.get(agent_handle, False) or self.obs.get(agent_handle) is None:
                continue

            norm_obs = self.obs_normalizer.normalize(self.obs[agent_handle])
            agent_id_val = self.agent_ids.get(agent_handle, 0.5)
            full_obs = np.append(norm_obs, [agent_id_val])
            
            mask = simple_action_mask(self.env, agent_handle)
            action = self.model.act(full_obs, mask)
            action_dict[agent_handle] = action

        # Defensive: ensure env.dones exists before step()
        if not hasattr(self.env, 'dones') or self.env.dones is None:
            self.env.dones = self.dones
        
        if '__all__' not in self.env.dones:
            self.env.dones['__all__'] = False

        self.obs, rewards, self.dones, _ = self.env.step(action_dict)
        
        for a, r in rewards.items():
            if isinstance(a, int):
                self.total_reward += r
        self.step_count += 1
        
        # Update render time
        self.render_time += 1.0
        
        # Render native view
        if self.renderer:
            try:
                # Ensure env.dones is updated for the renderer
                self.env.dones = self.dones
                self.renderer.render_env(show=False)
            except Exception as e:
                print(f"Rendering error: {e}")
            
        if self.assets:
            self.render_fantasy()
            
        return self.get_state()

    def render_fantasy(self):
        if not self.assets or not self.render_surf or not self.env or not self.env.rail:
            return
            
        # Draw background
        sw = self.render_surf.get_width()
        sh = self.render_surf.get_height()
        self.assets.draw_background(self.render_surf, 0, 0, sw, sh)
        
        cs = self.cell_size
        
        # Draw ground and rails
        for row in range(self.env.height):
            for col in range(self.env.width):
                sx, sy = col * cs, row * cs
                transitions = self.env.rail.grid[row, col]
                self.assets.draw_ground_tile(self.render_surf, sx, sy, has_rail=(transitions != 0))
                if transitions != 0:
                    self._draw_rail_cell_pygame(sx, sy, cs, transitions, row + col * 0.3)
                    
        # Draw targets
        for i, agent in enumerate(self.env.agents):
            if agent.target is not None:
                is_done = self.dones.get(i, False)
                tx, ty = agent.target[1] * cs, agent.target[0] * cs
                self._draw_target_pygame(tx, ty, cs, i, is_done)
                
        # Draw agents
        for i, agent in enumerate(self.env.agents):
            pos = agent.position
            if pos is None:
                if agent.state in [TrainState.WAITING, TrainState.READY_TO_DEPART]:
                    pos = agent.initial_position
                if pos is None:
                    continue
            
            ax, ay = pos[1] * cs, pos[0] * cs
            is_done = agent.state == TrainState.DONE
            is_malfunction = agent.malfunction_handler.malfunction_down_counter > 0
            is_moving = agent.state == TrainState.MOVING
            
            self._draw_agent_pygame(ax, ay, cs, agent.direction, i, 
                                   is_moving, is_malfunction, is_done, agent.speed_counter.speed)

    def _draw_rail_cell_pygame(self, sx, sy, cs, transitions, time_seed):
        half = cs // 2
        center = (sx + half, sy + half)
        dir_points = {
            0: (sx + half, sy),
            1: (sx + cs, sy + half),
            2: (sx + half, sy + cs),
            3: (sx, sy + half),
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
                    if pair in drawn: continue
                    drawn.add(pair)
                    p1 = dir_points[from_dir]
                    p2 = dir_points[to_dir] if from_dir != to_dir else center
                    pygame.draw.line(self.render_surf, COLORS['rail_sleeper'], p1, p2, max(3, cs // 7))
                    glow_phase = 0.7 + 0.3 * math.sin(self.assets._time * 2.0 + time_seed)
                    rail_color = lerp_color(COLORS['rail_dim'], COLORS['rail_core'], glow_phase)
                    pygame.draw.line(self.render_surf, rail_color, p1, p2, max(2, cs // 12))
                    highlight = lerp_color(COLORS['rail_core'], COLORS['rail_glow'], glow_phase * 0.4)
                    pygame.draw.line(self.render_surf, highlight, p1, p2, max(1, cs // 24))
        if len(connected_dirs) > 2:
            glow_phase = 0.5 + 0.5 * math.sin(self.assets._time * 3.0 + time_seed)
            junc_color = lerp_color(COLORS['rail_core'], COLORS['station_crystal'], glow_phase)
            junc_r = max(2, cs // 10)
            pygame.draw.circle(self.render_surf, junc_color, center, junc_r)
            pygame.draw.circle(self.render_surf, COLORS['rail_glow'], center, max(1, junc_r // 2))

    def _draw_target_pygame(self, tx, ty, cs, agent_idx, is_reached):
        half = cs // 2
        cx, cy = tx + half, ty + half
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        pulse = 0.5 + 0.5 * math.sin(self.assets._time * 2.5 + agent_idx * 1.3)
        if not is_reached:
            glow_size = int(cs * 0.9 + cs * 0.15 * pulse)
            glow = glow_surface(glow_size, color, 0.35 + 0.15 * pulse)
            self.render_surf.blit(glow, glow.get_rect(center=(cx, cy)), special_flags=pygame.BLEND_ADD)
            crystal_h, crystal_w = int(cs * 0.35), int(cs * 0.2)
            pts = [(cx, cy-crystal_h), (cx+crystal_w, cy), (cx, cy+crystal_h//3), (cx-crystal_w, cy)]
            pygame.draw.polygon(self.render_surf, (0,0,0), [(p[0]+1, p[1]+1) for p in pts])
            pygame.draw.polygon(self.render_surf, color, pts)
        else:
            glow = glow_surface(cs, lerp_color(color, (255,215,0), 0.5), 0.2)
            self.render_surf.blit(glow, glow.get_rect(center=(cx, cy)), special_flags=pygame.BLEND_ADD)

    def _draw_agent_pygame(self, ax, ay, cs, direction, agent_idx, is_moving, is_mal, is_done, speed):
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        half = cs // 2
        cx, cy = ax + half, ay + half
        if is_done: return
        body_w, body_h = int(cs * 0.7), int(cs * 0.4)
        angle = {0: 90, 1: 0, 2: 270, 3: 180}.get(direction, 0)
        train_surf = pygame.Surface((body_w + 10, body_h + 10), pygame.SRCALPHA)
        tcx, tcy = (body_w+10)//2, (body_h+10)//2
        pygame.draw.rect(train_surf, color, (tcx-body_w//2, tcy-body_h//2, body_w, body_h), border_radius=4)
        highlight = lerp_color(color, (255,255,255), 0.3)
        pygame.draw.rect(train_surf, highlight, (tcx-body_w//2+2, tcy-body_h//2+1, body_w-4, body_h//3), border_radius=2)
        if is_mal:
            pulse = 0.5 + 0.5 * math.sin(self.assets._time * 8.0)
            pygame.draw.rect(train_surf, (255,50,50,int(pulse*100)), (tcx-body_w//2-3, tcy-body_h//2-3, body_w+6, body_h+6), border_radius=5)
        rotated = pygame.transform.rotate(train_surf, angle)
        self.render_surf.blit(rotated, rotated.get_rect(center=(cx, cy)))
        
        # Badge
        badge_size = max(8, cs // 4)
        pygame.draw.circle(self.render_surf, (0,0,0,180), (cx, cy-half//2-4), badge_size+1)
        pygame.draw.circle(self.render_surf, color, (cx, cy-half//2-4), badge_size)
        font = pygame.font.SysFont('Arial', max(8, badge_size))
        text = font.render(str(agent_idx), True, (255,255,255))
        self.render_surf.blit(text, text.get_rect(center=(cx, cy-half//2-4)))

    def get_state(self):
        agents_info = []
        for i, agent in enumerate(self.env.agents):
            status = 'stopped'
            if agent.state == TrainState.DONE: status = 'done'
            elif agent.state == TrainState.MOVING: status = 'moving'
            elif agent.state == TrainState.WAITING: status = 'waiting'
            elif agent.state == TrainState.READY_TO_DEPART: status = 'ready'
            
            pos = [int(p) for p in agent.position] if agent.position is not None else None
            target = [int(p) for p in agent.target] if agent.target is not None else None
            
            agents_info.append({
                'handle': int(agent.handle),
                'position': pos,
                'direction': int(agent.direction) if agent.direction is not None else None,
                'target': target,
                'status': status,
                'speed': float(agent.speed_counter.speed),
                'malfunction': bool(agent.malfunction_handler.malfunction_down_counter > 0)
            })

        # Get grid transitions (compact format)
        grid = []
        if self.env and self.env.rail:
            for r in range(self.env.height):
                row = []
                for c in range(self.env.width):
                    row.append(int(self.env.rail.get_full_transitions(r, c)))
                grid.append(row)
        else:
            # Empty grid fallback
            grid = [[0 for _ in range(self.config['x_dim'])] for _ in range(self.config['y_dim'])]

        # Get native rendering if available
        native_image = None
        if self.assets and self.render_surf:
            try:
                # Convert Pygame surface to Base64
                img_data = pygame.image.tostring(self.render_surf, "RGB")
                from PIL import Image
                img = Image.frombytes("RGB", self.render_surf.get_size(), img_data)
                
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                native_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Error capturing fantasy image: {e}")
        elif self.renderer:
            try:
                img = self.renderer.get_image()
                if img is not None:
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    native_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Error capturing native image: {e}")

        return {
            'agents': agents_info,
            'grid': grid,
            'step': int(self.step_count),
            'max_steps': int(self.max_steps),
            'total_reward': float(self.total_reward),
            'done_all': bool(self.dones.get('__all__', False)),
            'width': int(self.env.width),
            'height': int(self.env.height),
            'native_image': native_image
        }

sim = SimulationState()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init', methods=['POST'])
def init_sim():
    config = request.json
    print(f"[API] Initializing with config: {config}")
    sim.init_env(config)
    if sim.model is None:
        sim.load_model()
    return jsonify(sim.get_state())

@app.route('/step', methods=['POST'])
def step_sim():
    state = sim.step()
    return jsonify(state)

@app.route('/reset', methods=['POST'])
def reset_sim():
    sim.reset()
    return jsonify(sim.get_state())

if __name__ == '__main__':
    # Using False for debug to avoid reloader issues with complex objects like simulators
    app.run(debug=False, port=5000)
