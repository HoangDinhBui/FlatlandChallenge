
/**
 * Enchanted Railway Kingdom - Web Visualizer JS
 */

class Renderer {
    constructor(canvas, ctx) {
        this.canvas = canvas;
        this.ctx = ctx;
        this.cellSize = 40;
        this.zoom = 1.0;
        this.offsetX = 0;
        this.offsetY = 0;
        this.stars = this._generateStars(150);
        this.time = 0;
        this.agentColors = ['#dc283c', '#1e64dc', '#1ec850', '#a032dc', '#ffa500', '#b4dcff', '#ff78b4', '#00b48c'];
        this.particles = [];
        this.viewMode = 'fantasy'; // 'fantasy' or 'native'
        this.nativeImage = new Image();
    }

    _generateStars(count) {
        const stars = [];
        for (let i = 0; i < count; i++) {
            stars.push({
                x: Math.random(),
                y: Math.random(),
                size: Math.random() * 2 + 0.5,
                brightness: Math.random() * 0.7 + 0.3,
                offset: Math.random() * Math.PI * 2
            });
        }
        return stars;
    }

    draw(state) {
        this.time += 0.016;
        const { agents, grid, width, height, native_image } = state;
        
        if (this.viewMode === 'native' && native_image) {
            this._drawNative(native_image, width, height);
            return;
        }

        // Clear background
        this._drawBackground();
        
        this.ctx.save();
        this.ctx.translate(this.offsetX, this.offsetY);
        this.ctx.scale(this.zoom, this.zoom);

        // Draw grid/ground
        this._drawGrid(width, height, grid);
        
        // Draw targets
        agents.forEach((agent, i) => {
            if (agent.target) {
                this._drawTarget(agent.target[1], agent.target[0], i, agent.status === 'done');
            }
        });

        // Draw agents
        agents.forEach((agent, i) => {
            if (agent.position) {
                this._drawAgent(agent.position[1], agent.position[0], agent.direction, i, agent);
            }
        });

        if (this.viewMode === 'fantasy') {
            this._updateParticles();
            this._drawParticles();
        }

        this.ctx.restore();
    }

    _drawNative(base64Data, w, h) {
        // Clear background with dark color to match fantasy theme
        this.ctx.fillStyle = '#0c061e';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        const img = this.nativeImage;
        img.src = 'data:image/png;base64,' + base64Data;
        
        if (img.complete) {
            this.ctx.save();
            this.ctx.translate(this.offsetX, this.offsetY);
            this.ctx.scale(this.zoom, this.zoom);
            
            // The native image is already rendered with appropriate aspect ratio and padding by RenderTool
            // We just draw it starting from 0,0 relative to our zoom/offset
            this.ctx.drawImage(img, 0, 0);
            
            this.ctx.restore();
        } else {
            img.onload = () => {
                // Re-draw will happen next frame
            };
        }
    }

    _drawBackground() {
        if (this.viewMode === 'fantasy') {
            const grad = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
            grad.addColorStop(0, '#0c061e');
            grad.addColorStop(1, '#160e2e');
            this.ctx.fillStyle = grad;
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

            // Twinkling stars
            this.stars.forEach(star => {
                const opacity = 0.3 + 0.7 * Math.abs(Math.sin(this.time * 2 + star.offset));
                this.ctx.fillStyle = `rgba(255, 255, 255, ${star.brightness * opacity})`;
                this.ctx.beginPath();
                this.ctx.arc(star.x * this.canvas.width, star.y * this.canvas.height, star.size * opacity, 0, Math.PI * 2);
                this.ctx.fill();
            });
        } else {
            this.ctx.fillStyle = '#f0f0f0';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    _drawGrid(w, h, grid) {
        if (!grid) return;
        const cs = this.cellSize;
        
        for (let r = 0; r < h; r++) {
            for (let c = 0; c < w; c++) {
                const x = c * cs;
                const y = r * cs;
                
                if (this.viewMode === 'fantasy') {
                    this.ctx.fillStyle = 'rgba(20, 50, 20, 0.3)';
                    this.ctx.fillRect(x, y, cs, cs);
                } else {
                    this.ctx.strokeStyle = '#ddd';
                    this.ctx.setLineDash([2, 4]);
                    this.ctx.strokeRect(x, y, cs, cs);
                    this.ctx.setLineDash([]);
                }
                
                // Rail
                const trans = grid[r][c];
                if (trans > 0) {
                    this._drawRail(x, y, trans);
                }
            }
        }
    }

    _drawRail(x, y, trans) {
        const cs = this.cellSize;
        const half = cs / 2;
        const center = { x: x + half, y: y + half };
        const dirPoints = [
            { x: x + half, y: y },        // N
            { x: x + cs, y: y + half },   // E
            { x: x + half, y: y + cs },   // S
            { x: x, y: y + half }         // W
        ];

        let connections = [];
        for (let f = 0; f < 4; f++) {
            for (let t = 0; t < 4; t++) {
                const bit = f * 4 + t;
                if (trans & (1 << (15 - bit))) {
                    connections.push([f, t]);
                }
            }
        }

        if (this.viewMode === 'fantasy') {
            this.ctx.lineWidth = 4;
            this.ctx.lineCap = 'round';
            connections.forEach(([f, t]) => {
                const p1 = dirPoints[f];
                const p2 = (f === t) ? center : dirPoints[t];
                const glow = 0.7 + 0.3 * Math.sin(this.time * 2 + f);
                this.ctx.strokeStyle = `rgba(255, 215, 0, ${glow * 0.3})`;
                this.ctx.lineWidth = 6;
                this.ctx.beginPath();
                this.ctx.moveTo(p1.x, p1.y);
                this.ctx.lineTo(p2.x, p2.y);
                this.ctx.stroke();
                this.ctx.strokeStyle = '#ffd700';
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.moveTo(p1.x, p1.y);
                this.ctx.lineTo(p2.x, p2.y);
                this.ctx.stroke();
            });
        } else {
            this.ctx.strokeStyle = '#333';
            this.ctx.lineWidth = 2;
            connections.forEach(([f, t]) => {
                const p1 = dirPoints[f];
                const p2 = (f === t) ? center : dirPoints[t];
                this.ctx.beginPath();
                this.ctx.moveTo(p1.x, p1.y);
                this.ctx.lineTo(p2.x, p2.y);
                this.ctx.stroke();
            });
        }
    }

    _drawTarget(c, r, i, reached) {
        const cs = this.cellSize;
        const x = c * cs + cs / 2;
        const y = r * cs + cs / 2;
        const color = this.agentColors[i % this.agentColors.length];

        if (this.viewMode === 'fantasy') {
            if (reached) {
                this.ctx.fillStyle = color;
                this.ctx.globalAlpha = 0.3;
                this.ctx.beginPath();
                this.ctx.arc(x, y, cs/4, 0, Math.PI * 2);
                this.ctx.fill();
                this.ctx.globalAlpha = 1.0;
                return;
            }
            const pulse = 0.8 + 0.2 * Math.sin(this.time * 3 + i);
            this.ctx.shadowBlur = 10 * pulse;
            this.ctx.shadowColor = color;
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.moveTo(x, y - cs/3);
            this.ctx.lineTo(x + cs/5, y);
            this.ctx.lineTo(x, y + cs/4);
            this.ctx.lineTo(x - cs/5, y);
            this.ctx.closePath();
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
            this.ctx.fillStyle = 'white';
            this.ctx.font = 'bold 10px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(i, x, y - cs/3 - 5);
        } else {
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(x, y, cs/6, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.strokeStyle = 'white';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            this.ctx.fillStyle = '#333';
            this.ctx.font = '8px Arial';
            this.ctx.fillText(i, x, y + cs/3);
        }
    }

    _drawAgent(c, r, dir, i, data) {
        const cs = this.cellSize;
        const cx = c * cs + cs / 2;
        const cy = r * cs + cs / 2;
        const color = this.agentColors[i % this.agentColors.length];
        
        this.ctx.save();
        this.ctx.translate(cx, cy);
        const rotation = [270, 0, 90, 180][dir] || 0;
        this.ctx.rotate(rotation * Math.PI / 180);

        if (this.viewMode === 'fantasy') {
            this.ctx.fillStyle = color;
            this.ctx.shadowBlur = 15;
            this.ctx.shadowColor = color;
            this.ctx.beginPath();
            this.ctx.roundRect(-cs*0.35, -cs*0.2, cs*0.7, cs*0.4, 4);
            this.ctx.fill();
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            this.ctx.fillRect(-cs*0.3, -cs*0.15, cs*0.6, cs*0.1);
            if (data.malfunction) {
                const pulse = 0.5 + 0.5 * Math.sin(this.time * 8);
                this.ctx.strokeStyle = `rgba(255, 0, 0, ${pulse})`;
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(-cs*0.4, -cs*0.25, cs*0.8, cs*0.5);
            }
        } else {
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.moveTo(cs * 0.3, 0);
            this.ctx.lineTo(-cs * 0.2, -cs * 0.2);
            this.ctx.lineTo(-cs * 0.2, cs * 0.2);
            this.ctx.closePath();
            this.ctx.fill();
            if (data.malfunction) {
                this.ctx.strokeStyle = 'red';
                this.ctx.lineWidth = 2;
                this.ctx.stroke();
            }
        }
        this.ctx.restore();
        
        if (this.viewMode === 'fantasy') {
            this.ctx.fillStyle = 'white';
            this.ctx.font = '10px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(i, cx, cy - cs/2);
            if (data.status === 'moving' && Math.random() < 0.4) {
                this.emitParticle(cx, cy, color);
            }
        }
    }

    emitParticle(x, y, color) {
        this.particles.push({
            x, y,
            vx: (Math.random() - 0.5) * 1.5,
            vy: (Math.random() - 0.5) * 1.5,
            life: 1.0,
            color
        });
    }

    _updateParticles() {
        this.particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            p.life -= 0.02;
        });
        this.particles = this.particles.filter(p => p.life > 0);
    }

    _drawParticles() {
        this.particles.forEach(p => {
            this.ctx.fillStyle = p.color;
            this.ctx.globalAlpha = p.life;
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, 2 * p.life, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.globalAlpha = 1.0;
    }

    fitToScreen(w, h) {
        const availableW = this.canvas.width;
        const availableH = this.canvas.height;
        const zoomX = (availableW - 100) / (w * this.cellSize);
        const zoomY = (availableH - 100) / (h * this.cellSize);
        this.zoom = Math.min(zoomX, zoomY, 2.0);
        this.offsetX = (availableW - (w * this.cellSize * this.zoom)) / 2;
        this.offsetY = (availableH - (h * this.cellSize * this.zoom)) / 2;
    }
}

class SimulationEngine {
    constructor() {
        this.canvas = document.getElementById('visualizer-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.renderer = new Renderer(this.canvas, this.ctx);
        this.state = null;
        this.isPlaying = false;
        this.speed = 5;
        this.lastStep = 0;

        this._setupEvents();
        this._resize();
        this.init();
    }

    async init() {
        const config = {
            n_agents: parseInt(document.getElementById('n_agents').value),
            x_dim: parseInt(document.getElementById('x_dim').value),
            y_dim: parseInt(document.getElementById('y_dim').value),
            n_cities: parseInt(document.getElementById('n_cities').value),
            seed: parseInt(document.getElementById('seed').value),
            malfunction_rate: parseFloat(document.getElementById('malfunction_rate').value),
            malfunction_min: parseInt(document.getElementById('malfunction_min').value),
            malfunction_max: parseInt(document.getElementById('malfunction_max').value),
            obs_tree_depth: parseInt(document.getElementById('obs_tree_depth').value),
            obs_max_path_depth: parseInt(document.getElementById('obs_max_path_depth').value),
            max_rails_between_cities: parseInt(document.getElementById('max_rails_between_cities').value),
            max_rails_in_city: parseInt(document.getElementById('max_rails_in_city').value)
        };

        const res = await fetch('/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        this.state = await res.json();
        this.renderer.fitToScreen(this.state.width, this.state.height);
        this._updateStats();
        if (!this.renderInterval) {
            this._render();
        }
    }

    async step() {
        const res = await fetch('/step', { method: 'POST' });
        this.state = await res.json();
        this._updateStats();
        if (this.state.done_all) this.isPlaying = false;
    }

    async reset() {
        const res = await fetch('/reset', { method: 'POST' });
        this.state = await res.json();
        this._updateStats();
    }

    _render() {
        if (this.state) {
            this.renderer.draw(this.state);
        }

        if (this.isPlaying) {
            const now = Date.now();
            const delay = 1000 / this.speed;
            if (now - this.lastStep > delay) {
                this.step();
                this.lastStep = now;
            }
        }

        this.renderInterval = requestAnimationFrame(() => this._render());
    }

    _updateStats() {
        if (!this.state) return;
        const agents = this.state.agents;
        const completed = agents.filter(a => a.status === 'done').length;
        
        document.getElementById('stat-step').textContent = `${this.state.step} / ${this.state.max_steps}`;
        document.getElementById('stat-completed').textContent = `${completed} / ${agents.length}`;
        const rate = (completed / agents.length * 100).toFixed(0);
        document.getElementById('stat-success').textContent = `${rate}%`;
        document.getElementById('stat-reward').textContent = this.state.total_reward.toFixed(2);
        
        const btnPlay = document.getElementById('play-icon');
        btnPlay.textContent = this.isPlaying ? '⏸' : '▶';
    }

    _setupEvents() {
        window.addEventListener('resize', () => this._resize());
        
        document.getElementById('btn-init').onclick = () => this.init();
        document.getElementById('btn-play').onclick = () => this.isPlaying = !this.isPlaying;
        document.getElementById('btn-step').onclick = () => this.step();
        document.getElementById('btn-reset').onclick = () => this.reset();
        document.getElementById('btn-fit').onclick = () => this.renderer.fitToScreen(this.state.width, this.state.height);
        
        document.querySelector('.toggle-advanced').onclick = () => {
            const fields = document.querySelector('.advanced-fields');
            const icon = document.querySelector('.toggle-advanced .icon');
            fields.classList.toggle('open');
            icon.textContent = fields.classList.contains('open') ? '▲' : '▼';
        };

        document.getElementById('view-mode').onchange = (e) => {
            this.renderer.viewMode = e.target.value;
        };

        const slider = document.getElementById('speed-slider');
        slider.oninput = (e) => {
            this.speed = parseInt(e.target.value);
            document.getElementById('speed-value').textContent = `${this.speed}x`;
        };

        // Zooming
        this.canvas.onwheel = (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            this.renderer.zoom *= delta;
        };

        // Panning
        let isDragging = false;
        let lastX, lastY;
        this.canvas.onmousedown = (e) => {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
        };
        window.onmousemove = (e) => {
            if (!isDragging) return;
            this.renderer.offsetX += e.clientX - lastX;
            this.renderer.offsetY += e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
        };
        window.onmouseup = () => isDragging = false;
    }

    _resize() {
        const wrapper = document.getElementById('canvas-wrapper');
        this.canvas.width = wrapper.clientWidth;
        this.canvas.height = wrapper.clientHeight;
    }
}

// Start app
window.onload = () => {
    new SimulationEngine();
};
