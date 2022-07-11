import roboverse as rv
import numpy as np
from tqdm import tqdm

num_traj = 50
timesteps = 65
env_name = 'VALMultiobjTray-v0'
kwargs = {'test_env': False}
data_save_path = '/iris/u/khazatsky/bridge_codebase/datasets/small_tray_demos.npy'
#do_bc_trick = True



# imlength = 96 * 128 * 3
# a_dim = 7

imlength = 48 * 48 * 3
a_dim = 4

all_returns = 0
env = rv.make(env_name, **kwargs)
env.reset()

dataset = {
    'observations': np.zeros((num_traj, timesteps, imlength), dtype=np.uint8),
    'noisy_actions': np.zeros((num_traj, timesteps, a_dim)),
    'actions': np.zeros((num_traj, timesteps, a_dim)),
}

def collect_demo(timesteps):
	env.demo_reset()
	returns = 0
	traj = {
	    'observations': np.zeros((timesteps, imlength), dtype=np.uint8),
	    'noisy_actions': np.zeros((timesteps, a_dim)),
	    'actions': np.zeros((timesteps, a_dim)),
	}
	for i in range(timesteps):
		traj['observations'][i] = np.uint8(env.render_obs().transpose()).flatten()
		action, noisy_action = env.get_demo_action()
		traj['actions'][i] = action
		traj['noisy_actions'][i] = noisy_action
		next_observation, reward, done, info = env.step(noisy_action)
		returns += reward
	return traj, returns



for j in tqdm(range(num_traj)):
	while True:
		traj, returns = collect_demo(timesteps)
		if env.done:
			dataset['observations'][j] = traj['observations']
			dataset['noisy_actions'][j]  = traj['noisy_actions']
			dataset['actions'][j]  = traj['actions']
			all_returns += returns
			break

print('Average Returns:', all_returns / num_traj)
np.save(data_save_path, dataset)
