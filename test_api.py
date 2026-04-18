
import requests
import time

def test_api():
    base_url = "http://localhost:5000"
    
    print("Testing /init...")
    config = {
        "n_agents": 2,
        "x_dim": 20,
        "y_dim": 20,
        "n_cities": 2,
        "seed": 42
    }
    try:
        res = requests.post(f"{base_url}/init", json=config)
        res.raise_for_status()
        state = res.json()
        print(f"Success: Grid {state['width']}x{state['height']}, Agents: {len(state['agents'])}")
        
        print("Testing /step...")
        res = requests.post(f"{base_url}/step")
        res.raise_for_status()
        state = res.json()
        print(f"Success: Step count {state['step']}")
        
        print("Testing /reset...")
        res = requests.post(f"{base_url}/reset")
        res.raise_for_status()
        state = res.json()
        print(f"Success: Step reset to {state['step']}")
        
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

if __name__ == "__main__":
    test_api()
