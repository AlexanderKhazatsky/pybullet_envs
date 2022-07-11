import roboverse.bullet as bullet
import numpy as np
import pybullet as p
from gym.spaces import Box, Dict
from collections import OrderedDict
from roboverse.envs.val_envs.val_base_env import VALBaseEnvV0
from roboverse.bullet.misc import load_obj, load_urdf, deg_to_quat, quat_to_deg, bbox_intersecting
from torchvision import transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import os.path as osp
import importlib.util
import random
import pickle
import gym
import os

scale_dict = {'wine': .2, 'car': .15, 'bench': .15, 'camera': .12, 'cola': .13, 'couch': .175, 'dish_soap': .16, 'gatorade': .165, 'plant': .18,
'plastic_bottle': .18, 'crushed_foil': .17, 'crushed_can': .12, 'corn_can': .11, 'carrot': .2, 'bus': .17, 'banana': .2, 'apple': .1}
deg_dict = {'wine': [90, 90, 0], 'car': [90, 0, 90], 'bench': [90, 0, 0], 'camera': [180, 0, 0], 'cola': [180, 0, 90], 'couch': [180, 0, 180], 'dish_soap': [180, 90, 90],
'gatorade': [180, 0, 90], 'plant': [180, 0, 90], 'plastic_bottle': [180, 0, 90], 'crushed_foil': [180, 0, 90], 'crushed_foil': [180, 0, 90], 'crushed_can': [270, 90, 180],
'corn_can': [90, 0, 0], 'carrot': [180, 0, 90], 'bus': [0, 90, 0], 'banana': [0, 90, 0], 'apple': [0, 90, 0]}
train_objects = ['car', 'bench', 'camera', 'cola', 'couch', 'dish_soap', 'gatorade', 'plant', 'plastic_bottle', 'crushed_foil', 'crushed_can', 'corn_can', 'carrot', 'bus', 'banana', 'apple']
test_object = 'wine'


class VALMultiobjTrayV0(VALBaseEnvV0):

    def __init__(self,
                 observation_mode='state',
                 obs_img_dim=48,
                 success_threshold=0.08,
                 transpose_image=False,
                 test_env=False,
                 *args,
                 **kwargs
                 ):
        """
        Grasping env with a single object
        :param goal_pos: xyz coordinate of desired goal
        :param reward_type: one of 'shaped', 'sparse'
        :param reward_min: minimum possible reward per timestep
        :param randomize: whether to randomize the object position or not
        :param observation_mode: state, pixels, pixels_debug
        :param obs_img_dim: image dimensions for the observations
        :param transpose_image: first dimension is channel when true
        :param invisible_robot: the robot arm is invisible when set to True
        """
        # Task Info #
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[.7, 0, -0.2], distance=0.3,
            yaw=90, pitch=-35, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            obs_img_dim, obs_img_dim)
        self.dt = 0.1
        self.height_delta = 0.08
        self.test_env = test_env
        super().__init__(*args, **kwargs)
        self._max_force = 100
        self._action_scale = 0.05
        self._load_environment()

    ### REWARD FUNCTIONS ###

    def get_info(self):
        ee_pos = self.get_end_effector_pos()
        obj_pos = self.get_object_pos('task_obj')
        goal_pos = self.get_object_pos('tray')

        aligned = np.linalg.norm(obj_pos[:2] - ee_pos[:2]) < 0.035
        picked_up = obj_pos[2] > -0.28
        enclosed = np.linalg.norm(obj_pos[2] - ee_pos[2]) < 0.08
        above_goal = np.linalg.norm(obj_pos[:2] - goal_pos[:2]) < 0.05
        obj_goal_distance = np.linalg.norm(obj_pos - goal_pos)
        return {
        'hand_object_aligned': aligned,
        'object_grasped': aligned,
        'object_above_goal': above_goal,
        'picked_up': picked_up,
        'obj_goal_distance': obj_goal_distance,
        'task_achieved': obj_goal_distance < 0.08,
        }

    def get_reward(self, info):
        return info['task_achieved'] - 1

    ### SCRIPTED POLICY ###
    def demo_reset(self):
        reset_obs = self.reset()
        self.timestep = 0
        self.done = False
        self.grip = -1.
        self.default_height = self.get_end_effector_pos()[2]
        return reset_obs

    def get_demo_action(self):

        # Get xyz action
        if self.done:
            xyz_action = self.maintain_hand()
        else:
            xyz_action = self.move_obj()

        action = np.concatenate((xyz_action, [self.grip]))
        action = np.clip(action, a_min=-1, a_max=1)
        noisy_action = np.random.normal(action, 0.05)
        noisy_action = np.clip(noisy_action, a_min=-1., a_max=1.)
        self.timestep += 1

        return action, noisy_action

    def maintain_hand(self):
        action = self.hand_pos - np.array(self.get_end_effector_pos())
        return action

    def move_obj(self):
        goal_pos = self.get_object_pos('tray') + np.array([0.02, 0, 0])
        ee_pos = self.get_end_effector_pos()
        obj_pos = self.get_object_pos('task_obj')
        curr_rot = self.get_object_angle('robot')[2]
        obj_rot = self.get_object_angle('task_obj')[2]
        target_pos = obj_pos + np.array([0., 0.02, 0.])
        aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.035
        enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.03
        above = ee_pos[2] >= self.default_height
        drop_object = np.linalg.norm(obj_pos[:2] - goal_pos[:2]) < 0.05
        self.hand_pos = np.array(ee_pos)

        if not aligned and not above and not drop_object:
            #print('Stage 1')
            action = np.array([0.,0., 0.5])
            self.grip = -1.
        elif not aligned and not drop_object:
            #print('Stage 2')
            action = (target_pos - ee_pos) * 3.0
            action[2] = (self.default_height + 0.05) - ee_pos[2]
            action *= 2.0
            self.grip = -1.
        elif aligned and not enclosed and not drop_object:
            #print('Stage 3')
            action = target_pos - ee_pos
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 1.5
            self.grip = -1.
        elif enclosed and self.grip < 1. and not drop_object:
            #print('Stage 4')
            action = target_pos - ee_pos
            self.grip += 0.2
        elif not above and not drop_object:
            #print('Stage 5')
            action = np.array([0.,0., .5])
            self.grip = 1.
        elif not drop_object:
            #print('Stage 6')
            action = (goal_pos - ee_pos) * 4.0
            action[2] = (self.default_height + 0.08) - ee_pos[2]
            self.grip = 1.
        elif self.grip > -1.:
            #print('Stage 7')
            action = (goal_pos - ee_pos) * 3.0
            action[2] = (self.default_height + 0.08) - ee_pos[2]
            self.grip -= 0.2
        else:
            #print('Stage 8')
            action = (goal_pos - ee_pos) * 3.0
            action[2] = (self.default_height + 0.08) - ee_pos[2]
            self.grip = -1.
            self.done = True

        return action

    ### ENV DESIGN ###

    def _reset_scene(self):
        # Reset Environment Variables
        self.lights_off = False
        objects_present = list(self._objects.keys())
        for key in objects_present:
            p.removeBody(self._objects[key])

        # Reset Sawyer
        p.removeBody(self._sawyer)
        self._sawyer = bullet.objects.sawyer_hand_visual_only()

        while True:
            tray_pos = np.array([np.random.uniform(.63, .7),
               np.random.uniform(-0.1, -0.05),
               -.2915])

            # obj_pos = np.array([np.random.uniform(.65, .75),
            #    np.random.uniform(0.05, 0.15),
            #    -.2])
            obj_pos = np.array([np.random.uniform(.62, .75),
               np.random.uniform(0.05, 0.15),
               -.2])

            if np.linalg.norm(tray_pos[1] - obj_pos[1]) > 0.14:
                break

        # Add Tray
        self._objects['tray'] = bullet.objects.drawer_tray(
            pos=tray_pos, scale=0.18)

        # Add Interaction Object
        if self.test_env:
            task_obj = 'wine'
            rgba = None
        else:
            task_obj = random.choice(train_objects)
            rgba = np.random.uniform(size=4)
            rgba[-1] = 1

        self._objects['task_obj'] = load_obj(
            self.shapenet_func('Objects/{0}/models/model_vhacd.obj'.format(task_obj)),
            self.shapenet_func('Objects/{0}/models/model_normalized.obj'.format(task_obj)),
            pos=obj_pos,
            quat=deg_to_quat(deg_dict[task_obj]),
            scale=scale_dict[task_obj],
            rgba=rgba,
            )

        # Add Distractor Objects
        pos = np.array([np.random.uniform(.4, .5),
               np.random.uniform(0.1, 0.2),
               -.2])
        self._objects['duck'] = bullet.objects.duck(pos=pos, scale=1.1,
            quat=deg_to_quat([90,0,np.random.uniform(-180, 180)]))

        pos = np.array([np.random.uniform(.4, .5),
               np.random.uniform(-0.2, -0.1),
               -.2])
        self._objects['bowl'] = load_obj(
            self.shapenet_func('Objects/bowl/models/model_vhacd.obj'),
            self.shapenet_func('Objects/bowl/models/model_normalized.obj'),
            pos=pos,
            quat=deg_to_quat([90, 0, 0]),
            scale=0.22,
            )

        # Allow the objects to land softly in low gravity
        p.setGravity(0, 0, -1)
        for _ in range(200):
            bullet.step()
        # After landing, bring to stop
        p.setGravity(0, 0, -10)
        for _ in range(100):
            bullet.step()

    def _load_environment(self):
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)


        self._objects = {}
        self._sensors = {}
        self._sawyer = bullet.objects.sawyer_hand_visual_only()
        self._table = bullet.objects.table(pos=[-.0, .0, -1.56], rgba=[.92,.85,.7,1], scale=2.)

        wood_wall = load_urdf(self.shapenet_func('Furniture/wooden_wall/wooden_wall.urdf'),
            pos=[-.5, -0.1, -0.],
            quat=deg_to_quat([180, 180, 90]),
            rgba=[.92,.85,.7,1],
            scale=2.5,
            useFixedBase=True)
        p.setCollisionFilterGroupMask(wood_wall, -1, 1, 0)

        self._workspace = bullet.Sensor(self._sawyer,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])

        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site')