import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from vizdoom import DoomGame, ScreenFormat, ScreenResolution

# ----------------------------
#       DQN Model
# ----------------------------
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
#      Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool)
        )

    def __len__(self):
        return len(self.buffer)

# ----------------------------
#    VizDoom Environment
# ----------------------------
class VizDoomEnv:
    def __init__(self, config_path, frame_skip=4, frame_stack=4):
        self.game = DoomGame()
        self.game.load_config(config_path)
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_window_visible(True)
        self.game.set_episode_timeout(2100)
        self.game.init()

        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.stack = deque(maxlen=frame_stack)
        self.action_space = [[1,0,0], [0,1,0], [0,0,1]]  # Left, Right, Shoot

    def reset(self):
        self.game.new_episode()
        state = self._get_preprocessed_frame()
        for _ in range(self.frame_stack):
            self.stack.append(state)
        return np.stack(self.stack, axis=0)

    def step(self, action_idx):
        reward = 0.0
        for _ in range(self.frame_skip):
            reward += self.game.make_action(self.action_space[action_idx])
            if self.game.is_episode_finished():
                break

        done = self.game.is_episode_finished()
        if not done:
            next_state = self._get_preprocessed_frame()
            self.stack.append(next_state)
            return np.stack(self.stack, axis=0), reward, done
        else:
            return np.zeros((self.frame_stack, 84, 84)), reward, done

    def _get_preprocessed_frame(self):
        frame = self.game.get_state().screen_buffer
        if frame is None:
            raise RuntimeError("screen_buffer is None. Is the episode started?")

        # Expect HWC RGB (120, 160, 3)
        if frame.shape == (120, 160, 3):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif len(frame.shape) == 2:
            gray = frame
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")

        resized = cv2.resize(gray, (84, 84))
        return resized / 255.0



    def close(self):
        self.game.close()

# ----------------------------
#        Training Loop
# ----------------------------
def train():
    env = VizDoomEnv("ViZDoom/scenarios/basic.cfg")
    input_shape = (4, 84, 84)
    n_actions = len(env.action_space)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(input_shape, n_actions).to(device)
    target_net = DQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(10000)

    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.1
    target_update_freq = 1000
    step_count = 0

    for episode in range(500):
        state = env.reset()
        episode_reward = 0

        while True:
            step_count += 1

            # Epsilon-greedy action
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                    action = q_vals.argmax().item()

            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # Train
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    q_next = target_net(next_states).max(1)[0]
                    q_target = rewards + gamma * q_next * (~dones)

                loss = F.mse_loss(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if step_count % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

    env.close()

if __name__ == "__main__":
    train()
