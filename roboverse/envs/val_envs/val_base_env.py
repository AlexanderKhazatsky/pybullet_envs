import roboverse.bullet as bullet
import numpy as np
import pybullet as p
from gym.spaces import Box, Dict
from collections import OrderedDict
from roboverse.envs.sawyer_base import SawyerBaseEnv
from roboverse.bullet.misc import load_obj, deg_to_quat, quat_to_deg, bbox_intersecting
from torchvision import transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import os.path as osp
import importlib.util
import random
import pickle
import gym
import os


cur_path = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(cur_path, '../assets/ShapeNet/')
shapenet_func = lambda name: ASSET_PATH + name

class VALBaseEnvV0(SawyerBaseEnv):

    def __init__(self,
                 observation_mode='state',
                 obs_img_dim=48,
                 success_threshold=0.08,
                 transpose_image=False,
                 DoF=3,
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
        assert DoF in [3, 4, 6]

        self.pickup_eps = -0.3
        self._observation_mode = observation_mode
        self._transpose_image = transpose_image
        self.image_shape = (obs_img_dim, obs_img_dim)
        self.image_length = np.prod(self.image_shape) * 3  # image has 3 channels
        self._ddeg_scale = 5
        self.DoF = DoF

        self._object_position_low = (.65, -0.15, -.25)
        self._object_position_high = (.75, 0.15, -.25)
        self._tray_position_low = (.65, -0.15, -.35)
        self._tray_position_high = (.75, 0.15, -.35)
        self._goal_low = np.array([0.62,-0.17,-.22])
        self._goal_high = np.array([0.78,0.1,-0.22])
        self._fixed_object_position = np.array([.7, 0.1, -.25])
        self._fixed_tray_position = np.array([.65, -0.15, -.35])
        self.start_obj_ind = 4 if (self.DoF == 3) else 8
        self.default_theta = bullet.deg_to_quat([180, 0, 0])
        self._success_threshold = success_threshold
        self.obs_img_dim = obs_img_dim #+.15
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[.7, 0, -0.2], distance=0.3,
            yaw=90, pitch=-35, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)
        self.dt = 0.1
        super().__init__(*args, **kwargs)
        self._max_force = 100
        self._action_scale = 0.05
        self.shapenet_func = shapenet_func
        self._load_environment()
        self._pos_low = [0.0,-0.25,-.36]
        self._pos_high = [0.8,0.25,-0.15]

    def _set_spaces(self):
        act_dim = self.DoF + 1
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)
        observation_dim = 3 if self.DoF == 3 else 7
        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        state_space = gym.spaces.Box(-obs_high, obs_high)
        self.observation_space = Dict([('state_observation', state_space)])

    def _reset_scene(self):
        # Reset Environment Variables
        self.lights_off = False
        objects_present = list(self._objects.keys())
        for key in objects_present:
            p.removeBody(self._objects[key])

        # Reset Sawyer
        p.removeBody(self._sawyer)
        self._sawyer = bullet.objects.sawyer_hand_visual_only()

        # Add Tray
        self._objects['tray'] = bullet.objects.drawer_tray(pos=np.array([.7, -0.1, -.35]), scale=0.18, rgba=[.2,.2,.2,1])

        # Add Interaction Object
        pos = np.array([np.random.uniform(.65, .75),
               np.random.uniform(0.05, 0.15),
               -.2])
        self._objects['wine'] = load_obj(
            self.shapenet_func('Objects/wine/models/model_vhacd.obj'),
            self.shapenet_func('Objects/wine/models/model_normalized.obj'),
            pos=pos,
            quat=deg_to_quat([90, 90, 0]),
            scale=0.2,
            rgba=[.83, .83, .83, 1],
            )

        # Add Distractor Objects
        pos = np.array([np.random.uniform(.4, .5),
               np.random.uniform(0.1, 0.2),
               -.2])
        self._objects['duck'] = bullet.objects.duck(pos=pos, scale=1.,
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
        for _ in range(100):
            bullet.step()
        # After landing, bring to stop
        p.setGravity(0, 0, -10)
        for _ in range(500):
            bullet.step()

    def _load_environment(self):
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)


        self._objects = {}
        self._sensors = {}
        self._sawyer = bullet.objects.sawyer_hand_visual_only()
        self._table = bullet.objects.table(rgba=[.92,.85,.7,1])

        self._workspace = bullet.Sensor(self._sawyer,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])

        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site')

    def render_obs(self):
        resolution_coeff = 2
        w, h = int(48 * resolution_coeff), int(48 * resolution_coeff)
        proj_matrix = bullet.get_projection_matrix(w, h)
        img, depth, segmentation = bullet.render(
            w, h, self._view_matrix_obs,
            proj_matrix, shadow=0, gaussian_width=0)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        img = Image.fromarray(img, mode='RGB')
        img = F.resize(img, (48, 48), T.InterpolationMode.LANCZOS)
        return np.array(img)


    def _format_action(self, *action):
        if self.DoF == 3:
            if len(action) == 1:
                delta_pos, gripper = action[0][:-1], action[0][-1]
            elif len(action) == 2:
                delta_pos, gripper = action[0], action[1]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            return np.array(delta_pos), gripper
        elif self.DoF == 4:
            if len(action) == 1:
                delta_pos, delta_yaw, gripper = action[0][:3], action[0][3:4], action[0][-1]
            elif len(action) == 3:
                delta_pos, delta_yaw, gripper = action[0], action[1], action[2]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            
            delta_angle = [0, 0, delta_yaw[0]]
            return np.array(delta_pos), np.array(delta_angle), gripper
        else:
            if len(action) == 1:
                delta_pos, delta_angle, gripper = action[0][:3], action[0][3:6], action[0][-1]
            elif len(action) == 3:
                delta_pos, delta_angle, gripper = action[0], action[1], action[2]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            return np.array(delta_pos), np.array(delta_angle), gripper

    def step(self, *action):

        # Get positional information
        pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        curr_angle = bullet.get_link_state(self._sawyer, self._end_effector, 'theta')
        default_angle = quat_to_deg(self.default_theta)
    
        # Keep necesary degrees of theta fixed
        if self.DoF == 3:
            angle = default_angle
        elif self.DoF == 4:
            angle = np.append(default_angle[:2], [curr_angle[2]])
        else:
            angle = curr_angle

        # If angle is part of action, use it
        if self.DoF == 3:
            delta_pos, gripper = self._format_action(*action)
        else:
            delta_pos, delta_angle, gripper = self._format_action(*action)
            angle += delta_angle * self._ddeg_scale

        # Update position and theta
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)
        theta = deg_to_quat(angle)
        self._simulate(pos, theta, gripper)

        # Get tuple information
        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = False
        return observation, reward, done, info

    def get_info(self):
        return {}

    def get_object_angle(self, object_name):
        if object_name == 'robot':
            return np.array(bullet.get_link_state(self._sawyer, self._end_effector, 'theta'))
        return bullet.get_body_info(self._objects[object_name], quat_to_deg=True)['theta']

    def get_object_pos(self, obj_name):
        if obj_name == 'button':
            return np.array(get_button_cylinder_pos(self._objects['button']))
        elif obj_name == 'drawer':
            return np.array(get_drawer_handle_pos(self._top_drawer))
        else:
            return np.array(bullet.get_body_info(self._objects[obj_name], quat_to_deg=False)['pos'])

    def get_reward(self, info):
        return 0

    def reset(self):
        self._reset_scene()
        self._format_state_query()

        # Move to starting positions
        action = np.array([0 for i in range(self.DoF)] + [-1])
        for _ in range(3): self.step(action)
        self.default_angle = self.get_object_angle('robot')
        self.default_height = self.get_end_effector_pos()[2]
        
        return self.get_observation()

    def reset(self):
        self._reset_scene()
        self._format_state_query()

        # Sample and load starting positions
        init_pos = np.array(self._pos_init)
        self.goal_pos = np.random.uniform(low=self._goal_low, high=self._goal_high)
        bullet.position_control(self._sawyer, self._end_effector, init_pos, self.default_theta)

        # Move to starting positions
        action = np.array([0 for i in range(self.DoF)] + [-1])
        for _ in range(3): self.step(action)
        return self.get_observation()

    def format_obs(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs


    def get_object_deg(self):
        object_info = bullet.get_body_info(self._objects['obj'],
                                           quat_to_deg=True)
        return object_info['theta']

    def get_hand_deg(self):
        return bullet.get_link_state(self._sawyer, self._end_effector,
            'theta', quat_to_deg=True)

    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_l_finger_joint', keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_r_finger_joint', keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)
        hand_theta = bullet.get_link_state(self._sawyer, self._end_effector,
            'theta', quat_to_deg=False)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()

        if self.DoF > 3:
            observation = np.concatenate((end_effector_pos, hand_theta, gripper_tips_distance))
        else:
            observation = np.concatenate((end_effector_pos, gripper_tips_distance))

        obs_dict = dict(state_observation=observation)

        return obs_dict
