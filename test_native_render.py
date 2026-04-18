
import io
import base64
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.utils.rendertools import RenderTool

def test_native_render():
    print("Initializing environment...")
    env = RailEnv(
        width=25,
        height=25,
        rail_generator=sparse_rail_generator(max_num_cities=3),
        line_generator=sparse_line_generator(),
        number_of_agents=1
    )
    env.reset()
    
    print("Initializing RenderTool with PIL...")
    try:
        renderer = RenderTool(env, gl="PIL")
        renderer.render_env()
        img = renderer.get_image()
        if img:
            print(f"Success! Image size: {img.size}")
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
            print(f"Base64 length: {len(encoded)}")
        else:
            print("Failed: No image returned.")
    except Exception as e:
        print(f"Failed with error: {e}")

if __name__ == "__main__":
    test_native_render()
