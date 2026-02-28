import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnvActions


def find_decision_cells(env):
    """

    :param env: The RailEnv to inspect
    :return: A set containing decision cells, made by switches and their neighbors
    """

    switches = []
    switches_neighbors = []
    directions = list(range(4))
    for h in range(env.height):
        for w in range(env.width):
            pos = (h, w)
            is_switch = False
            # Check for switch counting the outgoing transition
            for orientation in directions:
                possible_transitions = env.rail.get_transitions(*pos, orientation)
                num_transitions = np.count_nonzero(possible_transitions)
                if num_transitions > 1:
                    switches.append(pos)
                    is_switch = True
                    break
            if is_switch:
                # Add all neighbouring rails, if pos is a switch
                for orientation in directions:
                    possible_transitions = env.rail.get_transitions(*pos, orientation)
                    for movement in directions:
                        if possible_transitions[movement]:
                            switches_neighbors.append(get_new_position(pos, movement))

    return set(switches).union(set(switches_neighbors))


def get_action_masking(env, agent, action_size, train_params):
    """

    :param env: the environment
    :param agent: the agent index/handler
    :param action_size: the environment's number of available actions
    :param train_params: training parameters to customize the mask
    :return: the action mask for the passed agent
    """

    # Mask initialization
    action_mask = [1 * (0 if action == RailEnvActions.DO_NOTHING and not train_params.allow_no_op else 1)
                   for action in range(action_size)]

    # Mask filling
    if train_params.action_masking:
        rail_env = env.get_rail_env()
        agent_obj = rail_env.agents[agent]

        if agent_obj.position is not None:
            pos = agent_obj.position
            direction = agent_obj.direction

            for action in range(action_size):
                if action in [RailEnvActions.DO_NOTHING, RailEnvActions.STOP_MOVING]:
                    continue

                # Tính hướng mới
                if action == RailEnvActions.MOVE_LEFT:
                    new_dir = (direction - 1) % 4
                elif action == RailEnvActions.MOVE_FORWARD:
                    new_dir = direction
                elif action == RailEnvActions.MOVE_RIGHT:
                    new_dir = (direction + 1) % 4
                else:
                    continue

                # Kiểm tra transition hợp lệ
                transitions = rail_env.rail.get_transitions(*pos, direction)
                if transitions[new_dir] == 0:
                    action_mask[action] = 0
                    continue

                # Tính vị trí tiếp theo
                next_pos = get_new_position(pos, new_dir)
                r, c = next_pos

                # Out of bounds
                if not (0 <= r < rail_env.height and 0 <= c < rail_env.width):
                    action_mask[action] = 0

    return action_mask

# ============================================================
# CONTRIBUTION 1: Enhanced 3-Level Action Masking
# Mở rộng từ physical masking gốc
# ============================================================

def get_enhanced_action_masking(env, agent_handle, action_size, train_params):
    """
    3-level action masking cho railway safety:
      Level 1: Physical constraints (dùng lại hàm gốc)
      Level 2: Collision avoidance (MỚI)
      Level 3: Deadlock prevention fallback (MỚI)
    """
    from flatland.core.grid.grid4_utils import get_new_position
    from flatland.envs.rail_env import RailEnvActions

    rail_env = env.get_rail_env()
    agent = rail_env.agents[agent_handle]

    # Level 1: dùng lại hàm gốc
    action_mask = get_action_masking(env, agent_handle, action_size, train_params)

    # Nếu không bật masking hoặc agent chưa vào bản đồ thì trả về luôn
    if not train_params.action_masking or agent.position is None:
        return action_mask

    # Level 2: Collision Avoidance
    # Lấy tập vị trí của tất cả agents khác đang active
    other_positions = set()
    for i, other in enumerate(rail_env.agents):
        if i != agent_handle and other.position is not None:
            other_positions.add(other.position)

    for action in range(action_size):
        if action_mask[action] == 0:
            continue
        if action in [RailEnvActions.DO_NOTHING, RailEnvActions.STOP_MOVING]:
            continue

        # Tính hướng mới
        direction = agent.direction
        if action == RailEnvActions.MOVE_LEFT:
            new_direction = (direction - 1) % 4
        elif action == RailEnvActions.MOVE_FORWARD:
            new_direction = direction
        elif action == RailEnvActions.MOVE_RIGHT:
            new_direction = (direction + 1) % 4
        else:
            continue

        next_pos = get_new_position(agent.position, new_direction)

        if next_pos in other_positions:
            action_mask[action] = 0  # Block collision

    # Level 3: Deadlock Prevention
    # Nếu tất cả move actions đều bị block → cho phép DO_NOTHING
    move_actions = [
        RailEnvActions.MOVE_LEFT,
        RailEnvActions.MOVE_FORWARD,
        RailEnvActions.MOVE_RIGHT
    ]
    all_moves_blocked = all(action_mask[a] == 0 for a in move_actions)
    if all_moves_blocked:
        action_mask[RailEnvActions.DO_NOTHING] = 1

    return action_mask


def get_priority_based_masking(env, agent_id, action_size, train_params):
    """
    Level 4: Priority-based conflict resolution
    Tàu id nhỏ hơn có priority cao hơn.
    Chỉ STOP khi:
      - dist == 1 (sát nhau), HOẶC
      - dist == 2 VÀ đang tiến về phía tàu priority cao (moving toward)
    Không STOP nếu tàu kia đang STOPPED (tránh mutual deadlock dây chuyền).
    """
    action_mask = get_enhanced_action_masking(env, agent_id, action_size, train_params)

    rail_env = env.get_rail_env()
    agent = rail_env.agents[agent_id]

    from flatland.envs.agent_utils import TrainState
    if agent.state not in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
        return action_mask
    if agent.position is None:
        return action_mask

    # Vector hướng di chuyển: N=0, E=1, S=2, W=3
    dir_vectors = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    for other_id, other_agent in enumerate(rail_env.agents):
        if other_id >= agent_id:
            continue  # chỉ nhường tàu priority cao hơn (id nhỏ hơn)
        if other_agent.position is None:
            continue
        if other_agent.state not in [TrainState.MOVING, TrainState.MALFUNCTION]:
            continue  # không nhường tàu đang STOPPED → tránh deadlock dây chuyền

        dist = abs(agent.position[0] - other_agent.position[0]) + \
               abs(agent.position[1] - other_agent.position[1])

        should_stop = False

        if dist == 1:
            # Sát nhau → luôn nhường
            should_stop = True
        elif dist == 2:
            # Chỉ nhường nếu đang tiến về phía tàu kia
            dv = dir_vectors.get(agent.direction, (0, 0))
            next_pos = (agent.position[0] + dv[0], agent.position[1] + dv[1])
            next_dist = abs(next_pos[0] - other_agent.position[0]) + \
                        abs(next_pos[1] - other_agent.position[1])
            if next_dist < dist:
                should_stop = True

        if should_stop:
            priority_mask = [False] * action_size
            priority_mask[4] = True  # chỉ STOP
            return priority_mask

    return action_mask