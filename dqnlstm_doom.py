# Enhanced ViZDoom DQN with LSTM, policy saving, evaluation, and reward graph

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from vizdoom import DoomGame, ScreenFormat, ScreenResolution
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# ----------------------------
#       DQN with LSTM
# ----------------------------
class DQNLSTM(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNLSTM, self).__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_output((c, h, w))
        self.lstm = nn.LSTM(conv_out_size, 512, batch_first=True)
        self.fc = nn.Linear(512, n_actions)

    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, hidden=None):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        batch, seq, c, h, w = x.size()
        x = x.view(batch * seq, c, h, w)
        conv_out = self.conv(x).reshape(batch, seq, -1)
        lstm_out, hidden = self.lstm(conv_out, hidden)
        q_vals = self.fc(lstm_out[:, -1])
        return q_vals, hidden

# ----------------------------
#      Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, sequence):
        self.buffer.append(sequence)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for sequence in samples:
            s_seq, a_seq, r_seq, s2_seq, d_seq = zip(*sequence)
            states.append(np.stack(s_seq))
            actions.append(np.array(a_seq))
            rewards.append(np.array(r_seq))
            next_states.append(np.stack(s2_seq))
            dones.append(np.array(d_seq))
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.bool),
        )

    def __len__(self):
        return len(self.buffer)

# ----------------------------
#      VizDoom Wrapper
# ----------------------------
class VizDoomEnv:
    def __init__(self, config_path, frame_stack=4):
        self.game = DoomGame()
        self.game.load_config(config_path)
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_window_visible(True)
        self.game.init()

        self.frame_stack = frame_stack
        self.stack = deque(maxlen=frame_stack)
        self.action_space = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def reset(self):
        self.game.new_episode()
        frame = self._get_frame()
        for _ in range(self.frame_stack):
            self.stack.append(frame)
        return np.concatenate(self.stack, axis=0)

    def step(self, action_idx):
        reward = self.game.make_action(self.action_space[action_idx])
        done = self.game.is_episode_finished()
        next_state = self._get_frame() if not done else np.zeros_like(self.stack[0])
        self.stack.append(next_state)
        return np.concatenate(self.stack, axis=0), reward, done

    def _get_frame(self):
        frame = self.game.get_state().screen_buffer
        resized = cv2.resize(frame, (84, 84))
        return resized.transpose(2, 0, 1) / 255.0

    def close(self):
        self.game.close()

# ----------------------------
#       Evaluation Function
# ----------------------------
def evaluate(policy_net, env, episodes=3):
    policy_net.eval()
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            with torch.no_grad():
                inp = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(next(policy_net.parameters()).device)
                q_values, _ = policy_net(inp)
                action = q_values.argmax().item()
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        print(f"[EVAL] Episode {ep+1} - Total Reward: {total_reward:.2f}")

# ----------------------------
#       Training Loop
# ----------------------------
def train():
    env = VizDoomEnv("ViZDoom/scenarios/basic.cfg")
    input_shape = (3 * 4, 84, 84)
    frame_stack = 4
    n_actions = len(env.action_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQNLSTM(input_shape, n_actions).to(device)
    target_net = DQNLSTM(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(1000)
    writer = SummaryWriter()

    batch_size = 8
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.1
    target_update = 500
    episode_steps = 100
    global_step = 0
    best_reward = float('-inf')
    all_rewards = []

    for episode in range(1000):
        state = env.reset()
        state_seq, action_seq, reward_seq, next_state_seq, done_seq = [], [], [], [], []
        total_reward = 0
        early_done = False

        for _ in range(episode_steps):
            global_step += 1
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    inp = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    q_values, _ = policy_net(inp)
                    action = q_values.argmax().item()

            next_state, reward, done = env.step(action)
            if action == 2 and reward == 0:
                reward -= 1.0

            state_seq.append(state)
            action_seq.append(action)
            reward_seq.append(reward)
            next_state_seq.append(next_state)
            done_seq.append(done)
            total_reward += reward
            state = next_state

            if done:
                early_done = True
                break

        if not early_done and len(state_seq) == episode_steps:
            replay_buffer.push(list(zip(state_seq, action_seq, reward_seq, next_state_seq, done_seq)))

        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            states = states.to(device)
            next_states = next_states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            q_vals, _ = policy_net(states.unsqueeze(2))
            current_q = q_vals.gather(1, actions[:, -1].unsqueeze(1).long()).squeeze()
            with torch.no_grad():
                next_q, _ = target_net(next_states.unsqueeze(2))
                max_next_q = next_q.max(1)[0]
                target_q = rewards[:, -1] + gamma * max_next_q * (~dones[:, -1])

            loss = F.mse_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss", loss.item(), global_step)

            if global_step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        writer.add_scalar("Reward", total_reward, episode)
        all_rewards.append(total_reward)
        print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(), "best_dqn_lstm_model.pth")
            print(f"[SAVE] New best model saved with reward {best_reward:.2f}")

    env.close()
    writer.close()

    plt.figure(figsize=(10, 4))
    plt.plot(all_rewards)
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig("reward_plot.png")
    plt.show()

    print("\nEvaluating saved model:")
    policy_net.load_state_dict(torch.load("best_dqn_lstm_model.pth"))
    evaluate(policy_net, VizDoomEnv("ViZDoom/scenarios/basic.cfg"))

if __name__ == "__main__":
    train()
