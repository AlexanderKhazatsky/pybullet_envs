import roboverse as rv
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
from roboverse.bullet.misc import quat_to_deg
import matplotlib.pyplot as plt
import os
from PIL import Image
from moviepy.editor import *
import argparse
import time
images = []
plt.ion()

from roboverse.devices.xbox_controller import XboxController
controller = XboxController()




# #change to done 2 seperate
#spacemouse = rv.devices.SpaceMouse(DoF=6)
#kwargs = {'test_env': False}
env = rv.make('VALMultiobjTray-v0', DoF=6, gui=True)
# env.reset()
# #env = rv.make('RemoveLid-v0', gui=False)
# #env = rv.make('MugDishRack-v0', gui=False)
# #env = rv.make('FlipPot-v0', gui=True)


start = time.time()
num_traj = 10
for j in range(num_traj):
	env.reset()
	if j > 0: print(returns)
	returns = 0
	for i in range(150):
		img = Image.fromarray(np.uint8(env.render_obs()))
		images.append(np.array(img))
		action = controller.get_action()
		print(action)
		# action, noisy_action = env.get_demo_action()

		next_observation, reward, done, info = env.step(action)
		returns += reward
		#print(info)

print('Simulation Time:', (time.time() - start) / num_traj)

path = '/Users/sasha/Desktop/rollout.mp4'
#path = '/iris/u/khazatsky/bridge_codebase/data/visualizations/rollout.gif'

video = ImageSequenceClip(images, fps=24)
video.write_videofile(path)
