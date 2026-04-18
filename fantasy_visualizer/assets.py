"""
Fantasy 2D Asset Generator for Flatland Visualization
Procedurally generates all sprites and visual assets using Pygame.
Theme: Enchanted Railway Kingdom
"""

import math
import random
import pygame

# ═══════════════════════════════════════════════════════════
# COLOR PALETTE - Enchanted Fantasy Theme
# ═══════════════════════════════════════════════════════════

COLORS = {
    # Background & ground
    'bg_dark':          (12, 6, 30),
    'bg_mid':           (22, 14, 46),
    'bg_light':         (32, 22, 62),
    'ground_dark':      (15, 35, 15),
    'ground_mid':       (20, 50, 20),
    'ground_light':     (28, 62, 28),
    'ground_accent':    (18, 45, 22),

    # Rails - Golden magical
    'rail_core':        (255, 215, 0),
    'rail_glow':        (255, 235, 100),
    'rail_dim':         (180, 150, 40),
    'rail_sleeper':     (90, 60, 30),

    # Agent colors - Gemstone theme
    'agent_ruby':       (220, 40, 60),
    'agent_sapphire':   (30, 100, 220),
    'agent_emerald':    (30, 200, 80),
    'agent_amethyst':   (160, 50, 220),
    'agent_topaz':      (255, 165, 0),
    'agent_diamond':    (180, 220, 255),
    'agent_opal':       (255, 120, 180),
    'agent_jade':       (0, 180, 140),

    # Station/Target
    'station_core':     (100, 180, 255),
    'station_glow':     (140, 200, 255),
    'station_crystal':  (200, 230, 255),

    # UI
    'ui_bg':            (15, 10, 30, 220),
    'ui_border':        (180, 150, 50),
    'ui_text':          (220, 210, 190),
    'ui_text_bright':   (255, 245, 220),
    'ui_text_dim':      (140, 130, 110),
    'ui_accent':        (255, 215, 0),

    # Particles
    'spark_gold':       (255, 230, 100),
    'spark_white':      (255, 255, 240),
    'magic_purple':     (180, 100, 255),
    'magic_blue':       (100, 160, 255),
}

AGENT_COLORS = [
    COLORS['agent_ruby'],
    COLORS['agent_sapphire'],
    COLORS['agent_emerald'],
    COLORS['agent_amethyst'],
    COLORS['agent_topaz'],
    COLORS['agent_diamond'],
    COLORS['agent_opal'],
    COLORS['agent_jade'],
]

# Direction constants: N=0, E=1, S=2, W=3
DIR_NORTH, DIR_EAST, DIR_SOUTH, DIR_WEST = 0, 1, 2, 3
DIR_OFFSETS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
DIR_NAMES = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}


def lerp_color(c1, c2, t):
    """Linear interpolation between two colors."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(min(len(c1), len(c2))))


def glow_surface(size, color, intensity=1.0):
    """Create a radial glow surface."""
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    center = size // 2
    for r in range(center, 0, -1):
        alpha = int(intensity * 255 * (1 - r / center) ** 1.5)
        alpha = max(0, min(255, alpha))
        c = (*color[:3], alpha)
        pygame.draw.circle(surf, c, (center, center), r)
    return surf


class FantasyAssets:
    """Generates and caches all visual assets for the fantasy railway theme."""

    def __init__(self, cell_size=40):
        self.cell_size = cell_size
        self.half = cell_size // 2
        self._cache = {}
        self._stars = []
        self._time = 0

        # Pre-generate ground tile variations
        self._ground_tiles = self._generate_ground_tiles()
        # Pre-generate glow textures
        self._rail_glow = glow_surface(cell_size, COLORS['rail_glow'], 0.3)
        self._station_glow = glow_surface(cell_size * 2, COLORS['station_glow'], 0.6)
        # Pre-generate star field
        self._generate_stars(200)

    def _generate_stars(self, count):
        """Generate background star positions and properties."""
        self._stars = []
        for _ in range(count):
            self._stars.append({
                'x': random.random(),
                'y': random.random(),
                'size': random.uniform(0.5, 2.5),
                'brightness': random.uniform(0.3, 1.0),
                'twinkle_speed': random.uniform(1.0, 4.0),
                'twinkle_offset': random.uniform(0, math.pi * 2),
            })

    def _generate_ground_tiles(self):
        """Generate varied ground tiles for a natural look."""
        tiles = []
        for i in range(8):
            surf = pygame.Surface((self.cell_size, self.cell_size))
            base_r = random.randint(12, 22)
            base_g = random.randint(38, 55)
            base_b = random.randint(12, 22)
            surf.fill((base_r, base_g, base_b))

            # Add subtle texture
            for _ in range(random.randint(3, 8)):
                x = random.randint(0, self.cell_size - 1)
                y = random.randint(0, self.cell_size - 1)
                r = random.randint(1, 3)
                c = (base_r + random.randint(-5, 8),
                     base_g + random.randint(-5, 10),
                     base_b + random.randint(-5, 5))
                c = tuple(max(0, min(255, v)) for v in c)
                pygame.draw.circle(surf, c, (x, y), r)

            # Subtle grass dots
            for _ in range(random.randint(1, 4)):
                x = random.randint(2, self.cell_size - 3)
                y = random.randint(2, self.cell_size - 3)
                c = (base_r - 3, base_g + random.randint(5, 20), base_b - 3)
                c = tuple(max(0, min(255, v)) for v in c)
                pygame.draw.line(surf, c, (x, y), (x, y - random.randint(2, 4)), 1)

            tiles.append(surf)
        return tiles

    def draw_background(self, surface, camera_x, camera_y, width, height):
        """Draw the enchanted night sky background with stars."""
        # Gradient sky
        for y in range(0, height, 4):
            t = y / height
            color = lerp_color(COLORS['bg_dark'], COLORS['bg_mid'], t)
            pygame.draw.line(surface, color, (0, y), (width, y))

        # Draw twinkling stars
        self._time += 0.016  # ~60fps
        for star in self._stars:
            sx = int(star['x'] * width)
            sy = int(star['y'] * height * 0.4)  # Stars in top 40%
            twinkle = 0.5 + 0.5 * math.sin(self._time * star['twinkle_speed'] + star['twinkle_offset'])
            brightness = int(star['brightness'] * twinkle * 255)
            if brightness < 30:
                continue
            color = (brightness, brightness, min(255, brightness + 30))
            size = max(1, int(star['size'] * twinkle))
            if size <= 1:
                surface.set_at((sx, sy), color)
            else:
                pygame.draw.circle(surface, color, (sx, sy), size)

    def draw_ground_tile(self, surface, x, y, has_rail=False):
        """Draw a ground tile at pixel position."""
        # Use deterministic tile selection based on position
        tile_idx = (x * 7 + y * 13) % len(self._ground_tiles)
        surface.blit(self._ground_tiles[tile_idx], (x, y))

        if has_rail:
            # Subtle glow under rails
            glow_rect = self._rail_glow.get_rect(center=(x + self.half, y + self.half))
            surface.blit(self._rail_glow, glow_rect, special_flags=pygame.BLEND_ADD)

    def draw_rail_segment(self, surface, px, py, transitions, time_offset=0.0):
        """
        Draw fantasy-styled rail tracks for a cell.
        transitions: 16-bit int encoding all transitions for this cell.
        """
        cs = self.cell_size
        half = self.half
        center = (px + half, py + half)

        # Parse which directions connect
        connections = set()
        for from_dir in range(4):
            for to_dir in range(4):
                bit_index = from_dir * 4 + to_dir
                if transitions & (1 << (15 - bit_index)):
                    connections.add((from_dir, to_dir))

        if not connections:
            return

        # Get unique direction pairs
        drawn = set()
        dir_points = {
            DIR_NORTH: (px + half, py),
            DIR_EAST:  (px + cs, py + half),
            DIR_SOUTH: (px + half, py + cs),
            DIR_WEST:  (px, py + half),
        }

        connected_dirs = set()
        for f, t in connections:
            connected_dirs.add(f)
            connected_dirs.add(t)

        # Draw rail connections as golden paths
        for from_d, to_d in connections:
            pair = tuple(sorted([from_d, to_d]))
            if pair in drawn:
                continue
            drawn.add(pair)

            p1 = dir_points[from_d]
            p2 = dir_points[to_d]

            if from_d == to_d:
                # Dead-end: draw from edge to center
                p2 = center

            # Draw rail bed (darker)
            pygame.draw.line(surface, COLORS['rail_sleeper'], p1, p2, max(5, cs // 6))

            # Draw rail core (golden)
            glow_phase = 0.7 + 0.3 * math.sin(self._time * 2.0 + time_offset + from_d)
            rail_color = lerp_color(COLORS['rail_dim'], COLORS['rail_core'], glow_phase)
            pygame.draw.line(surface, rail_color, p1, p2, max(3, cs // 10))

            # Draw rail highlight (brighter center line)
            highlight_color = lerp_color(COLORS['rail_core'], COLORS['rail_glow'], glow_phase * 0.5)
            pygame.draw.line(surface, highlight_color, p1, p2, max(1, cs // 20))

        # Draw junction point if multiple connections
        if len(connected_dirs) > 2:
            # Crystal junction marker
            glow_phase = 0.5 + 0.5 * math.sin(self._time * 3.0 + time_offset)
            junc_color = lerp_color(COLORS['rail_core'], COLORS['station_crystal'], glow_phase)
            pygame.draw.circle(surface, junc_color, center, max(3, cs // 8))
            pygame.draw.circle(surface, COLORS['rail_glow'], center, max(2, cs // 12))

    def draw_agent(self, surface, px, py, direction, agent_idx, is_moving=True,
                   is_malfunction=False, is_done=False, speed=1.0):
        """Draw a fantasy train/carriage at the given pixel position."""
        cs = self.cell_size
        half = self.half
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        cx, cy = px + half, py + half

        if is_done:
            # Draw completion sparkle
            self._draw_completion_effect(surface, cx, cy, cs, agent_idx)
            return

        # Train body dimensions
        body_w = int(cs * 0.7)
        body_h = int(cs * 0.4)

        # Rotate based on direction
        angle = {DIR_NORTH: 90, DIR_EAST: 0, DIR_SOUTH: 270, DIR_WEST: 180}.get(direction, 0)

        # Create train surface
        train_surf = pygame.Surface((body_w + 10, body_h + 10), pygame.SRCALPHA)
        tw, th = train_surf.get_size()
        tcx, tcy = tw // 2, th // 2

        # Glow effect behind train
        glow_size = int(cs * 0.9)
        glow = glow_surface(glow_size, color, 0.4)
        glow_rect = glow.get_rect(center=(cx, cy))
        surface.blit(glow, glow_rect, special_flags=pygame.BLEND_ADD)

        # --- Draw the train carriage ---
        # Body shadow
        body_rect = pygame.Rect(tcx - body_w // 2 + 1, tcy - body_h // 2 + 1, body_w, body_h)
        pygame.draw.rect(train_surf, (0, 0, 0, 80), body_rect, border_radius=4)

        # Main body
        body_rect = pygame.Rect(tcx - body_w // 2, tcy - body_h // 2, body_w, body_h)
        pygame.draw.rect(train_surf, color, body_rect, border_radius=4)

        # Body highlight (lighter top)
        highlight_color = lerp_color(color, (255, 255, 255), 0.3)
        highlight_rect = pygame.Rect(tcx - body_w // 2 + 2, tcy - body_h // 2 + 1,
                                     body_w - 4, body_h // 3)
        pygame.draw.rect(train_surf, highlight_color, highlight_rect, border_radius=2)

        # Front indicator (direction)
        front_x = tcx + body_w // 2 - 3
        front_y = tcy
        pygame.draw.circle(train_surf, COLORS['rail_glow'], (front_x, front_y), max(2, cs // 12))

        # Windows
        win_color = (200, 230, 255, 200)
        for i in range(2):
            wx = tcx - body_w // 4 + i * (body_w // 3)
            wy = tcy - 1
            pygame.draw.rect(train_surf, win_color,
                             (wx - 2, wy - 2, 5, 4), border_radius=1)

        # Malfunction effect (red pulse)
        if is_malfunction:
            pulse = 0.5 + 0.5 * math.sin(self._time * 8.0)
            mal_color = (255, 50, 50, int(pulse * 100))
            mal_surf = pygame.Surface((body_w + 6, body_h + 6), pygame.SRCALPHA)
            pygame.draw.rect(mal_surf, mal_color,
                             (0, 0, body_w + 6, body_h + 6), border_radius=5)
            train_surf.blit(mal_surf, (tcx - body_w // 2 - 3, tcy - body_h // 2 - 3))

        # Speed indicator (wheels/magic trail)
        if speed < 1.0:
            # Slow speed indicator - smaller wheels
            for i in range(2):
                wx = tcx - body_w // 4 + i * (body_w // 2)
                wy = tcy + body_h // 2
                pygame.draw.circle(train_surf, COLORS['rail_dim'], (wx, wy), max(2, cs // 14))

        # Rotate train
        rotated = pygame.transform.rotate(train_surf, angle)
        rot_rect = rotated.get_rect(center=(cx, cy))
        surface.blit(rotated, rot_rect)

        # Agent number badge
        self._draw_agent_badge(surface, cx, cy - half // 2 - 4, agent_idx, color)

    def _draw_agent_badge(self, surface, cx, cy, agent_idx, color):
        """Draw a small numbered badge above the agent."""
        badge_size = max(8, self.cell_size // 4)
        # Background circle
        pygame.draw.circle(surface, (0, 0, 0, 180), (cx, cy), badge_size + 1)
        pygame.draw.circle(surface, color, (cx, cy), badge_size)
        # Number
        font = pygame.font.SysFont('Arial', max(8, badge_size))
        text = font.render(str(agent_idx), True, (255, 255, 255))
        text_rect = text.get_rect(center=(cx, cy))
        surface.blit(text, text_rect)

    def _draw_completion_effect(self, surface, cx, cy, cs, agent_idx):
        """Draw sparkle effect when agent reaches target."""
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        n_sparkles = 8
        for i in range(n_sparkles):
            angle = self._time * 2 + i * (math.pi * 2 / n_sparkles)
            dist = cs * 0.3 + cs * 0.1 * math.sin(self._time * 4 + i)
            sx = cx + math.cos(angle) * dist
            sy = cy + math.sin(angle) * dist
            spark_color = lerp_color(color, COLORS['spark_gold'], 0.5 + 0.5 * math.sin(self._time * 5 + i))
            size = max(1, int(2 + math.sin(self._time * 6 + i * 0.7)))
            pygame.draw.circle(surface, spark_color, (int(sx), int(sy)), size)

        # Center star
        star_pulse = 0.5 + 0.5 * math.sin(self._time * 3)
        star_color = lerp_color(COLORS['spark_gold'], COLORS['spark_white'], star_pulse)
        pygame.draw.circle(surface, star_color, (cx, cy), max(2, cs // 8))

    def draw_target(self, surface, px, py, agent_idx, is_reached=False):
        """Draw a fantasy target marker (crystal beacon)."""
        cs = self.cell_size
        half = self.half
        cx, cy = px + half, py + half
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]

        # Pulsing glow
        pulse = 0.5 + 0.5 * math.sin(self._time * 2.5 + agent_idx * 1.3)

        if not is_reached:
            # Glow ring
            glow_size = int(cs * 0.8 + cs * 0.1 * pulse)
            glow = glow_surface(glow_size, color, 0.35 + 0.15 * pulse)
            glow_rect = glow.get_rect(center=(cx, cy))
            surface.blit(glow, glow_rect, special_flags=pygame.BLEND_ADD)

            # Crystal base
            crystal_h = int(cs * 0.35)
            crystal_w = int(cs * 0.2)
            points = [
                (cx, cy - crystal_h),       # Top
                (cx + crystal_w, cy),        # Right
                (cx, cy + crystal_h // 3),   # Bottom
                (cx - crystal_w, cy),        # Left
            ]
            # Shadow
            shadow_points = [(p[0] + 1, p[1] + 1) for p in points]
            pygame.draw.polygon(surface, (0, 0, 0), shadow_points)
            # Crystal body
            pygame.draw.polygon(surface, color, points)
            # Crystal highlight
            highlight = lerp_color(color, (255, 255, 255), 0.5)
            inner_points = [
                (cx, cy - crystal_h + 3),
                (cx + crystal_w - 3, cy),
                (cx, cy + crystal_h // 3 - 2),
                (cx - crystal_w + 3, cy),
            ]
            pygame.draw.polygon(surface, highlight, inner_points)

            # Floating rune (agent number)
            bounce = math.sin(self._time * 2 + agent_idx) * 3
            font = pygame.font.SysFont('Arial', max(10, cs // 3))
            text = font.render(str(agent_idx), True, COLORS['spark_white'])
            text_rect = text.get_rect(center=(cx, cy - crystal_h - 6 + bounce))
            surface.blit(text, text_rect)
        else:
            # Completed target - subtle glow
            glow = glow_surface(cs, lerp_color(color, COLORS['spark_gold'], 0.5), 0.2)
            glow_rect = glow.get_rect(center=(cx, cy))
            surface.blit(glow, glow_rect, special_flags=pygame.BLEND_ADD)

    def draw_spawn_point(self, surface, px, py, agent_idx):
        """Draw the spawn point (initial position) of an agent."""
        cs = self.cell_size
        half = self.half
        cx, cy = px + half, py + half
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]

        # Draw a faded ring
        ring_color = lerp_color(color, COLORS['bg_dark'], 0.5)
        pygame.draw.circle(surface, ring_color, (cx, cy), cs // 3, 2)

        # Small portal effect
        portal_pulse = 0.3 + 0.2 * math.sin(self._time * 1.5 + agent_idx * 2)
        portal_glow = glow_surface(cs // 2, color, portal_pulse)
        glow_rect = portal_glow.get_rect(center=(cx, cy))
        surface.blit(portal_glow, glow_rect, special_flags=pygame.BLEND_ADD)


class ParticleSystem:
    """Magical particle effects for agent movement trails."""

    def __init__(self):
        self.particles = []

    def emit(self, x, y, color, count=3, speed=1.0):
        """Emit particles at a position."""
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2)
            vel = random.uniform(0.5, 2.0) * speed
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * vel,
                'vy': math.sin(angle) * vel,
                'life': random.uniform(0.5, 1.5),
                'max_life': random.uniform(0.5, 1.5),
                'color': color,
                'size': random.uniform(1, 3),
            })

    def update(self, dt=0.016):
        """Update all particles."""
        alive = []
        for p in self.particles:
            p['life'] -= dt
            if p['life'] > 0:
                p['x'] += p['vx'] * dt * 30
                p['y'] += p['vy'] * dt * 30
                p['vy'] += 0.5 * dt  # Slight gravity
                alive.append(p)
        self.particles = alive

    def draw(self, surface):
        """Draw all particles."""
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            size = max(1, int(p['size'] * alpha))
            color = tuple(int(c * alpha) for c in p['color'][:3])
            color = tuple(max(0, min(255, c)) for c in color)
            pygame.draw.circle(surface, color, (int(p['x']), int(p['y'])), size)
