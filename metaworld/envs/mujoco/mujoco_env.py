import abc
import warnings

from numpy.lib.type_check import _imag_dispatcher

import glfw
from gym import error
from gym.utils import seeding
import numpy as np
from os import path
import gym

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def _assert_task_is_set(func):
    def inner(*args, **kwargs):
        env = args[0]
        if not env._set_task_called:
            raise RuntimeError(
                'You must call env.set_task before using env.'
                + func.__name__
            )
        return func(*args, **kwargs)
    return inner


DEFAULT_SIZE = 500

class MujocoEnv(gym.Env, abc.ABC):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """

    max_path_length = 500

    def __init__(self, model_path, frame_skip):
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)

        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', "rgb_array"],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._did_see_sim_exception = False

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abc.abstractmethod
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        pass

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    @_assert_task_is_set
    def reset(self):
        self._did_see_sim_exception = False
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if getattr(self, 'curr_path_length', 0) > self.max_path_length:
            raise ValueError('Maximum path length allowed by the benchmark has been exceeded')
        if self._did_see_sim_exception:
            return

        if n_frames is None:
            n_frames = self.frame_skip
        self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            try:
                self.sim.step()
            except mujoco_py.MujocoException as err:
                warnings.warn(str(err), category=RuntimeWarning)
                self._did_see_sim_exception = True

    def render(
        self,
        mode="human",
        width=480,
        height=480,
        camera_id=None,
        camera_name=["view1", "view2"]
    ):
        # if mode == "rgb_array" or mode == "depth_array":
        #     if camera_id is not None and camera_name is not None:
        #         raise ValueError(
        #             "Both `camera_id` and `camera_name` cannot be"
        #             " specified at the same time."
        #         )

        #     no_camera_specified = camera_name is None and camera_id is None
        #     if no_camera_specified:
        #         camera_name = "track"

        #     if camera_id is None and camera_name in self.model._camera_name2id:
        #         camera_id = self.model.camera_name2id(camera_name)

        #     self._get_viewer(mode).render(width, height, camera_id=camera_id)

        def render_single_view(camera_name):
            data = self.sim.render(
                width, height, mode='offscreen', camera_name=camera_name)

            FOV = self.sim.model.cam_fovy[self.sim.model.camera_name2id(camera_name)]
            print(FOV)

            pt2d_traj = self.project_point_world_to_ego(fov=FOV, img_height=height,img_width=width, camera_name=camera_name)


            # original image is upside-down, so flip it
            # img =  data[::-1, :]
            img = data
            # px = width - int(pt2d[0])
            # py = int(pt2d[1])
            if isinstance(pt2d_traj, np.ndarray):
                for i in range(0, pt2d_traj.shape[0]):
                    pt2d = pt2d_traj[i]
                    py = width - int(pt2d[1])
                    px = height - int(pt2d[0])
                    img[py:py+3, px:px+3, 1] = 250
            return img

        if mode == "rgb_array":
            # window size used for old mujoco-py:
            # data = self._get_viewer(mode).read_pixels(width, height, depth=False)

            if isinstance(camera_name, str):
                img = render_single_view(camera_name)
            elif isinstance(camera_name, list):
                imgs = []
                for name in camera_name:
                    t_img = render_single_view(name)
                    imgs.append(t_img)
                img = np.concatenate(imgs, axis=1)

            return img

        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            # img =  data[::-1, :]
            img = data

            return img

        elif mode == "human":
            self._get_viewer(mode).render()


    # def render(self,
    #            offscreen=True,
    #            camera_name="corner",
    #            resolution=(640, 480)):
    #     assert_string = (
    #         "camera_name should be one of ",
    #         "corner3, corner, corner2, topview, gripperPOV, behindGripper")
    #     assert camera_name in {
    #         "corner3", "corner", "corner2", "topview", "gripperPOV",
    #         "behindGripper"
    #     }, assert_string

    #     print("===render")
    #     if not offscreen:
    #         self._get_viewer('human').render()
    #     else:
    #         return self.sim.render(
    #             *resolution, mode='offscreen', camera_name=camera_name)

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)
