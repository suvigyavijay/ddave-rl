from stable_baselines3 import  DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env import DangerousDaveEnv
import time

# Create the DangerousDaveEnv environment
env = DangerousDaveEnv(render_mode="human")
env = DummyVecEnv([lambda: env])

# Define and train the DQN agent
model = DQN("CnnPolicy", env, verbose=1, batch_size=64)
model.learn(total_timesteps=50000, progress_bar=True) 

checkpoint_timestamp = int(time.time())

# Save the trained model if desired
model.save("checkpoints/dqn_ddave_{}".format(checkpoint_timestamp))

# Evaluate the trained model
model = DQN.load("checkpoints/dqn_ddave_{}".format(checkpoint_timestamp))

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
