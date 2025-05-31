from vizdoom import *
import numpy as np
import cv2
from collections import deque

# -------- Initialize Game --------
def init_game():
    game = DoomGame()
    game.load_config("ViZDoom/scenarios/basic.cfg")  # Adjust path if needed
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.init()
    return game

# -------- Preprocess Frame --------
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized / 255.0

# -------- Frame Stack --------
class FrameStack:
    def __init__(self, stack_size=4):
        self.stack = deque(maxlen=stack_size)
        self.stack_size = stack_size

    def reset(self, frame):
        self.stack = deque([frame]*self.stack_size, maxlen=self.stack_size)
        return np.stack(self.stack, axis=0)

    def append(self, frame):
        self.stack.append(frame)
        return np.stack(self.stack, axis=0)

# -------- Main Loop --------
if __name__ == "__main__":
    game = init_game()
    stacker = FrameStack(4)

    episodes = 5
    for episode in range(episodes):
        print(f"\nEpisode #{episode + 1}")
        game.new_episode()

        frame = game.get_state().screen_buffer
        state = stacker.reset(preprocess(frame))

        while not game.is_episode_finished():
            # Always shoot (action index 2)
            action = [0, 0, 1]
            reward = game.make_action(action)

            if game.get_state():
                next_frame = preprocess(game.get_state().screen_buffer)
                state = stacker.append(next_frame)

            print("Reward:", reward)

    game.close()
