import gymnasium as gym
from stable_baselines3 import PPO
from ale_py import ALEInterface
ale = ALEInterface()
import ale_py

# Create the Donkey Kong environment
# The environment id is typically "ALE/DonkeyKong-v5"; adjust if necessary.
env = gym.make("ALE/DonkeyKong-v5", render_mode="human", frameskip=4)

# Create a PPO agent with a CNN policy (good for image-based inputs)
model = PPO("CnnPolicy", env, verbose=1)

# Train the agent for 100,000 timesteps (adjust timesteps as needed)
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_donkeykong")
print("Training complete and model saved as 'ppo_donkeykong'.")

# OPTIONAL: Demonstrate the trained agent
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    # For debugging, you might want to display or process the frame.
env.close()
