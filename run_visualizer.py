"""
╔══════════════════════════════════════════════════════════════╗
║       ENCHANTED RAILWAY KINGDOM - Flatland Visualizer       ║
║───────────────────────────────────────────────────────────── ║
║  Launch this to visualize your trained model!                ║
║                                                              ║
║  Usage:                                                      ║
║    python run_visualizer.py                                  ║
║    python run_visualizer.py --model path/to/model.pt         ║
║    python run_visualizer.py --agents 3 --width 20 --height 20║
║    python run_visualizer.py --seed 123                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Enchanted Railway Kingdom - Flatland Fantasy Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  SPACE       Play / Pause simulation
  RIGHT       Step forward (when paused)
  R           Reset to new episode
  +/-         Speed up / slow down
  Scroll      Zoom in / out
  Shift+Drag  Pan camera
  F           Fit to screen
  1-5         Follow agent camera
  0/ESC       Free camera / Quit
  TAB         Toggle info panel
        """
    )

    parser.add_argument('--model', type=str, default='curriculum_stage3.pt',
                        help='Path to model checkpoint (default: curriculum_stage3.pt)')
    parser.add_argument('--agents', type=int, default=5,
                        help='Number of agents (default: 5)')
    parser.add_argument('--width', type=int, default=25,
                        help='Grid width (default: 25)')
    parser.add_argument('--height', type=int, default=25,
                        help='Grid height (default: 25)')
    parser.add_argument('--cities', type=int, default=3,
                        help='Number of cities (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--cell-size', type=int, default=48,
                        help='Cell size in pixels (default: 48)')
    parser.add_argument('--speed', type=float, default=0.3,
                        help='Step delay in seconds (default: 0.3)')
    parser.add_argument('--fullscreen', action='store_true',
                        help='Start in fullscreen mode')

    args = parser.parse_args()

    # Update config
    from fantasy_visualizer.main import CONFIG
    CONFIG['model_path'] = os.path.abspath(args.model)
    CONFIG['n_agents'] = args.agents
    CONFIG['x_dim'] = args.width
    CONFIG['y_dim'] = args.height
    CONFIG['n_cities'] = args.cities
    CONFIG['seed'] = args.seed
    CONFIG['cell_size'] = args.cell_size
    CONFIG['default_step_delay'] = args.speed

    print("""
    +======================================================+
    |     ENCHANTED RAILWAY KINGDOM                        |
    |         Flatland Fantasy Visualizer                  |
    |                                                      |
    |   Model: {:40s} |
    |   Grid:  {}x{} with {} agents                        |
    |   Seed:  {:42d} |
    +======================================================+
    """.format(
        os.path.basename(args.model),
        args.width, args.height, args.agents,
        args.seed
    ))

    from fantasy_visualizer.main import FlatlandFantasyVisualizer
    app = FlatlandFantasyVisualizer()
    app.run()


if __name__ == "__main__":
    main()
