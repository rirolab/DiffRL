from token import ENDMARKER
import numpy as np
import cv2
import taichi as ti
from .losses import Loss ,StateLoss, ChamferLoss, EMDLoss

# TODO: run on GPU, fast_math will cause error on float64's sqrt; removing it cuases compile error..
def init_taichi():
    ti.init(arch=ti.gpu, debug=False, fast_math=True)


@ti.data_oriented
class TaichiEnv:
    def __init__(self, cfg, loss_fn=Loss,nn=False, loss=True):
        """
        A taichi env builds scene according the configuration and the set of manipulators
        """
        # primitives are environment specific parameters ..
        # move it inside can improve speed; don't know why..
        from .mpm_simulator import MPMSimulator
        from .primitive import Primitives
        from .renderer import Renderer
        from .shapes import Shapes
        from .nn.mlp import MLP

        self.cfg = cfg.ENV
        self.primitives = Primitives(cfg.PRIMITIVES)
        self.shapes = Shapes(cfg.SHAPES)
        self.init_particles, self.particle_colors = self.shapes.get()

        cfg.SIMULATOR.defrost()
        self.n_particles = cfg.SIMULATOR.n_particles = len(self.init_particles)
        print("Number of particles: ",self.n_particles)

        self.simulator = MPMSimulator(cfg.SIMULATOR, self.primitives)
        self.renderer = Renderer(cfg.RENDERER, self.primitives)

        if nn:
            self.nn = MLP(self.simulator, self.primitives, (256, 256))

        if loss:
            self.loss = loss_fn(cfg.ENV.loss, self.simulator)
        else:
            self.loss = None
        self._is_copy = True

    def set_copy(self, is_copy: bool):
        self._is_copy = is_copy

    def initialize(self):
        # initialize all taichi variable according to configurations..
        self.primitives.initialize()
        self.simulator.initialize()
        self.renderer.initialize()
        if self.loss:
            self.loss.initialize()
            #self.renderer.set_target_density(self.loss.target_density.to_numpy()/self.simulator.p_mass)

        # call set_state instead of reset..
        self.simulator.reset(self.init_particles)
        if self.loss:
            self.loss.clear()

    def render(self, mode='human', **kwargs):
        assert self._is_copy, "The environment must be in the copy mode for render ..."
        if self.n_particles > 0:
            x = self.simulator.get_x(0)
            self.renderer.set_particles(x, self.particle_colors)
        img = self.renderer.render_frame(shape=1, primitive=1, **kwargs)
        img = np.uint8(img.clip(0, 1) * 255)

        if mode == 'human':
            image = img[..., ::-1]
            image = cv2.resize(image,(512,512))
            cv2.imshow('x', image)
            #cv2.waitKey(0)
        elif mode == 'plt':
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show(block=False)
        else:
            return img

    def get_obs(self, n_observed_particles):
        if self._is_copy:
            t = 0
        else:
            t = self.simulator.cur

        x = self.simulator.get_x(t)
        v = self.simulator.get_v(t)
        outs = []
        for i in self.primitives:
            outs.append(i.get_state(t, needs_grad=False))
        s = np.concatenate(outs)
        step_size = len(x) // n_observed_particles
        return np.concatenate((np.concatenate((x[::step_size], v[::step_size]), axis=-1).reshape(-1), s.reshape(-1)))

    def step(self, action=None):
        if action is not None:
            action = np.array(action)
        self.simulator.step(is_copy=self._is_copy, action=action)

    def compute_loss(self,copy_grad=True,decay=1,taichi_loss=False):
        assert self.loss is not None
        if self._is_copy:
            self.loss.clear()
            return self.loss.compute_loss(0)
        else:
            if isinstance(self.loss,ChamferLoss) or isinstance(self.loss,EMDLoss):
                return self.loss.compute_loss(self.simulator.cur,copy_grad,decay=decay)
            else:
                return self.loss.compute_loss(self.simulator.cur)

    def get_state(self):
        assert self.simulator.cur == 0
        return {
            'state': self.simulator.get_state(0),
            'softness': self.primitives.get_softness(),
            'is_copy': self._is_copy
        }

    def get_current_state(self):
        return self.simulator.get_current_state()

    def save_current_state(self,filename):
        states = self.get_current_state()
        np.savez(filename+'.npz',*states)

    def set_state(self, state, softness, is_copy):
        self.simulator.cur = 0
        self.simulator.set_state(0, state)
        self.primitives.set_softness(softness)
        self._is_copy = is_copy
        if self.loss:
            self.loss.reset()
            self.loss.clear()
    
    def set_torch_nn(self,nn):
        self.simulator.set_nn(nn)

    # obs will be an numpy array
    # obs will be the last step cur
    def act(self,obs,t,obs_type='x'):
        action = np.zeros(self.simulator.primitives.action_dims[-1])
        self.simulator.act(obs,t,self.simulator.cur,action,obs_type)
        return action

    def set_target(self,target):
        self.loss.set_target(target)

    def get_state_grad(self):
        x_grad,v_grad,_, _ = self.simulator.get_state_grad(0)
        return x_grad, v_grad

    def get_x(self):
        return self.simulator.get_x(self.simulator.cur)

    def get_v(self):
        return self.simulator.get_v(self.simulator.cur)

    def set_grad(self):
        assert(isinstance(self.loss,ChamferLoss) or isinstance(self.loss,EMDLoss))
        self.loss.set_grad()
