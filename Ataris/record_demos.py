import gymnasium as gym
import torch
import pygame
import pickle
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import FrameStackObservation
from collections import namedtuple
from datetime import datetime
import ale_py
import os

# 手动有效动作映射表
key_action_map = {
    pygame.K_SPACE: 1,    # FIRE
    pygame.K_UP: 2,       # UP
    pygame.K_RIGHT: 3,    # RIGHT
    pygame.K_LEFT: 4,     # LEFT
    pygame.K_DOWN: 5,     # DOWN
    pygame.K_d: 11,  # RIGHT FIRE
    pygame.K_a: 12,  # LEFT FIRE
}

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def play_and_record_once(env):
    trajectory = []
    obs, _ = env.reset()
    state = torch.tensor(obs, dtype=torch.float32).squeeze(-1).unsqueeze(0) / 255.0

    running = True
    clock = pygame.time.Clock()

    while running:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        for key, mapped_action in key_action_map.items():
            if keys[key]:
                action = mapped_action
                break

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = torch.tensor(next_obs, dtype=torch.float32).squeeze(-1).unsqueeze(0) / 255.0

        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        done_tensor = torch.tensor([done], dtype=torch.float32)

        trajectory.append(Transition(state, action, reward_tensor, next_state, done_tensor))
        state = next_state

        if done:
            break

        clock.tick(30)

    return trajectory

def main():
    pygame.init()
    pygame.display.set_mode((100, 100))  # 创建虚拟窗口接收按键

    env = gym.make("ALE/DonkeyKong-v5", render_mode="human")
    env = AtariWrapper(env, terminal_on_life_loss=True, frame_skip=1)  # 控制灵敏度高，利于示范
    env = FrameStackObservation(env, 4)
    
    num = 5

    all_trajectories = []

    for i in range(num):
        print(f"\n第 {i+1} 次游戏开始。请按方向键/空格/w/q 操作角色，Esc 退出。\n")
        traj = play_and_record_once(env)
        print(f"已记录 {len(traj)} 步。")
        all_trajectories.append(traj)

    filename = f"demo/dk_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    os.makedirs("demo", exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(all_trajectories, f)

    print(f"\n🎉 所有示范已保存到：{filename}")
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
