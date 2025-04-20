import time
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 📂 Kayıt klasörünü oluştur (eğer yoksa)
save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

def read_log_file(filename):
    try:
        data = np.loadtxt(filename, delimiter=',')
        return data if data.size > 0 else np.array([])
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return np.array([])

def moving_average(data, window_size):
    if len(data) < window_size:
        return np.zeros(len(data))
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_and_save():
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Log dosyasını oku
        total_reward_per_episode = read_log_file("model_total_reward_per_episode.txt")

        # 📊 Total Reward per Episode
        if total_reward_per_episode.size > 0:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            episodes = total_reward_per_episode[:, 0]
            total_rewards = total_reward_per_episode[:, 1]
            mva_total_rewards = moving_average(total_rewards, window_size=1500)

            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Total Reward", color="tab:blue")
            ax1.plot(episodes, total_rewards, label="Total Reward per Episode", alpha=0.5, color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.set_ylabel("Moving Avg (1500)", color="red")
            ax2.plot(episodes[:len(mva_total_rewards)], mva_total_rewards, label="Moving Avg (1500)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            fig.tight_layout()
            save_path = os.path.join(save_dir, f"total_reward_plot_{timestamp}.png")
            plt.savefig(save_path)
            print(f"Total reward plot saved: {save_path}")
            plt.close(fig)

        time.sleep(600)  # ⏳ 10 dakika bekle (600 saniye)

if __name__ == "__main__":
    plot_and_save()
