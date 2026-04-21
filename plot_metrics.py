import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu từ kết quả đánh giá 4 Version
models = ['Version 1', 'Version 2', 'Version 3', 'Version 4']

scores = [-0.103, -0.087, -0.093, -0.103]
completions = [0.00, 4.17, 4.17, 0.00]
deadlocks = [0.00, 25.00, 0.00, 8.33]

wait_probs = [49.00, 78.00, 82.03, 58.63]
left_probs = [14.33, 6.13, 1.60, 11.20]
forward_probs = [18.70, 8.43, 9.10, 11.00]
right_probs = [12.13, 4.53, 4.77, 10.20]
stop_probs = [5.80, 2.83, 2.50, 8.93]

plt.style.use('ggplot' if 'ggplot' in plt.style.available else 'default')

# 1. Biểu đồ: Điểm số (Score)
plt.figure(figsize=(8, 5))
bars = plt.bar(models, scores, color=['grey', 'orange', 'green', 'red'])
plt.title('Điểm số Trung bình (Score) của 4 Models')
plt.ylabel('Score (Càng sát 0 càng tốt)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval - 0.005, round(yval, 3), ha='center', va='top', color='black', fontweight='bold')
plt.tight_layout()
plt.savefig('chart_score.png')
plt.close()

# 2. Biểu đồ kép: Tỉ lệ Hoàn thành vs Tỉ lệ Kẹt tàu
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10, 6))
bar1 = plt.bar(x - width/2, completions, width, label='Hoàn thành (%)\n(Càng CAO càng tốt)', color='dodgerblue')
bar2 = plt.bar(x + width/2, deadlocks, width, label='Kẹt tàu - Deadlock (%)\n(Càng THẤP càng tốt)', color='crimson')

plt.title('Tỉ lệ Hoàn thành và Tỉ lệ Kẹt tàu (Deadlock)')
plt.xticks(x, models)
plt.ylabel('Phần trăm (%)')
plt.legend()

for bar in bar1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval}%', ha='center', va='bottom', fontweight='bold')
for bar in bar2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('chart_completion_vs_deadlock.png')
plt.close()

# 3. Biểu đồ thanh ngang chồng: Phân bổ Hành động (Action Probabilities)
plt.figure(figsize=(10, 6))

p1 = plt.barh(models, wait_probs, color='lightgray', label='Chờ (Wait)')
p2 = plt.barh(models, left_probs, left=wait_probs, color='lightblue', label='Trái (Left)')
p3 = plt.barh(models, forward_probs, left=np.array(wait_probs)+np.array(left_probs), color='limegreen', label='Thẳng (Forward)')
p4 = plt.barh(models, right_probs, left=np.array(wait_probs)+np.array(left_probs)+np.array(forward_probs), color='gold', label='Phải (Right)')
p5 = plt.barh(models, stop_probs, left=np.array(wait_probs)+np.array(left_probs)+np.array(forward_probs)+np.array(right_probs), color='salmon', label='Dừng (Stop)')

plt.title('Phân bổ Tỉ lệ Hành động (Action Probabilities) của 4 Models')
plt.xlabel('Phần trăm (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('chart_action_probs.png')
plt.close()

print("Đã tạo xong các biểu đồ: chart_score.png, chart_completion_vs_deadlock.png, chart_action_probs.png")
