"""
Coordination Logger - In ra lộ trình và kết quả điều phối của từng tàu
"""
from flatland.envs.agent_utils import TrainState


class CoordinationLogger:
    def __init__(self, rail_env, n_agents):
        self.rail_env = rail_env
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        """Reset logs cho episode mới"""
        self.paths = {i: [] for i in range(self.n_agents)}          # Lộ trình từng tàu
        self.conflicts = []                                           # Danh sách xung đột
        self.wait_events = {i: [] for i in range(self.n_agents)}     # Lần tàu phải chờ
        self.completion = {i: None for i in range(self.n_agents)}    # Bước hoàn thành
        self.prev_positions = {i: None for i in range(self.n_agents)}
        self.prev_actions = {i: None for i in range(self.n_agents)}

    def log_step(self, step, action_dict):
        """Ghi log sau mỗi bước"""
        agents = self.rail_env.agents

        for agent_id in range(self.n_agents):
            agent = agents[agent_id]

            # Ghi vị trí hiện tại vào lộ trình
            if agent.position is not None:
                pos = agent.position
                if not self.paths[agent_id] or self.paths[agent_id][-1] != pos:
                    self.paths[agent_id].append(pos)

            # Ghi action
            if agent_id in action_dict:
                self.prev_actions[agent_id] = action_dict[agent_id]

            # Phát hiện tàu phải STOP (action = 4) khi có tàu khác gần
            if agent_id in action_dict and action_dict[agent_id] == 4:
                if agent.position is not None:
                    # Kiểm tra có tàu khác gần không
                    for other_id in range(self.n_agents):
                        if other_id == agent_id:
                            continue
                        other = agents[other_id]
                        if other.position is None:
                            continue
                        dist = abs(agent.position[0] - other.position[0]) + \
                               abs(agent.position[1] - other.position[1])
                        if dist <= 3:
                            # Phát hiện xung đột
                            conflict = {
                                "step": step,
                                "agent": agent_id,
                                "yielded_to": other_id,
                                "position": agent.position,
                            }
                            # Tránh log trùng
                            if not self.conflicts or self.conflicts[-1] != conflict:
                                self.conflicts.append(conflict)
                            self.wait_events[agent_id].append(step)
                            break

            # Ghi bước hoàn thành
            if agent.state == TrainState.DONE and self.completion[agent_id] is None:
                self.completion[agent_id] = step

        self.prev_positions = {i: agents[i].position for i in range(self.n_agents)}

    def print_report(self, episode=None):
        """In báo cáo điều phối sau episode"""
        agents = self.rail_env.agents
        action_names = {0: "↻ DO_NOTHING", 1: "← LEFT", 2: "↑ FORWARD", 3: "→ RIGHT", 4: "◼ STOP"}

        print("\n" + "="*60)
        if episode:
            print(f"COORDINATION REPORT - Episode {episode}")
        else:
            print("COORDINATION REPORT")
        print("="*60)

        # 1. Lộ trình từng tàu
        print("\n📍 LỘ TRÌNH TỪNG TÀU:")
        for agent_id in range(self.n_agents):
            agent = agents[agent_id]
            path = self.paths[agent_id]
            start = agent.initial_position
            target = agent.target
            done_step = self.completion[agent_id]
            n_waits = len(self.wait_events[agent_id])

            status = f"✅ Hoàn thành bước {done_step}" if done_step else "❌ Chưa hoàn thành"

            print(f"\n  Tàu {agent_id}:")
            print(f"    Xuất phát : {start}")
            print(f"    Đích đến  : {target}")
            print(f"    Số bước đi: {len(path)}")
            print(f"    Số lần chờ: {n_waits}")
            print(f"    Trạng thái: {status}")
            if len(path) > 0:
                # In lộ trình rút gọn (mỗi 5 bước)
                sampled = path[::max(1, len(path)//10)]
                print(f"    Lộ trình  : {' → '.join(str(p) for p in sampled)}")

        # 2. Các điểm xung đột
        print(f"\n⚠️  ĐIỂM XUNG ĐỘT ({len(self.conflicts)} lần):")
        if not self.conflicts:
            print("  Không có xung đột!")
        else:
            # Group by position
            from collections import Counter
            positions = Counter(c["position"] for c in self.conflicts)
            hotspots = positions.most_common(5)
            print(f"  Top điểm xung đột:")
            for pos, count in hotspots:
                print(f"    Vị trí {pos}: {count} lần xung đột")

            # In chi tiết 5 xung đột đầu
            print(f"\n  Chi tiết (5 xung đột đầu):")
            for c in self.conflicts[:5]:
                print(f"    Bước {c['step']:3d}: Tàu {c['agent']} nhường Tàu {c['yielded_to']} tại {c['position']}")

        # 3. Tổng kết
        n_done = sum(1 for v in self.completion.values() if v is not None)
        total_waits = sum(len(v) for v in self.wait_events.values())

        print(f"\n📊 TỔNG KẾT:")
        print(f"  Tàu hoàn thành : {n_done}/{self.n_agents}")
        print(f"  Tổng xung đột  : {len(self.conflicts)}")
        print(f"  Tổng lần chờ   : {total_waits}")
        print("="*60 + "\n")
