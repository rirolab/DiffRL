from plb.engine.taichi_env import TaichiEnv
import torch

# We don't know number of particles and number of primitives
# We do know each primitive has a 13-D vector as state
# Assume state is a huge vector
class StateFormat:
    @staticmethod
    def state2frame(state,n_particles,n_primitives):
        x = state[:3*n_particles].view(n_particles,3).numpy()
        v = state[3*n_particles:6*n_particles].view(n_particles,3).numpy()
        F = state[6*n_particles:15*n_particles].view(n_particles,3,3).numpy()
        C = state[15*n_particles:24*n_particles].view(n_particles,3,3).numpy()
        frame = [x,v,F,C]
        base = 24*n_particles
        for i in range(n_primitives):
            frame.append(state[base+i*13:base+(i+1)*13].numpy())
        return frame

    def frame2state(frame,n_particles,n_primitives):
        state = torch.zeros(24*n_particles+13*n_primitives).double()
        state[:3*n_particles] = torch.from_numpy(frame[0].flatten())
        state[3*n_particles:6*n_particles] = torch.from_numpy(frame[1].flatten())
        state[6*n_particles:15*n_particles] = torch.from_numpy(frame[2].flatten())
        state[15*n_particles:24*n_particles] = torch.from_numpy(frame[3].flatten())
        base = 24*n_particles
        for i in range(len(frame)-4):
            state[base+i*13:base+(i+1)*13] = torch.from_numpy(frame[i+4])
        return state


class TaichiLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, taichi_env: TaichiEnv, state, action, cur=None):
        start_cur = taichi_env.simulator.cur if cur == None else cur
        frame = StateFormat.state2frame(state)
        taichi_env.set_state(frame,taichi_env.softness,False,start_cur)
        taichi_env.step(action=action.numpy())
        end_cur = taichi_env.simulator.cur # Point to the last frame in this step
        next_state = taichi_env.get_current_state()['state']
        ctx.save_for_backward(start_cur,end_cur,taichi_env)
        return torch.from_numpy(next_state)

    @staticmethod
    def backward(ctx,next_state_grad):
        start_cur,end_cur,taichi_env = ctx.saved_tensors
        grad_frame = StateFormat.state2frame(next_state_grad)
        taichi_env.set_state_grad(end_cur,grad_frame)
        taichi_env.backprop(end_cur,start_cur)
        frame_grad = taichi_env.get_state_grad(start_cur) # Get gradient for ALL state
        action_grad = taichi_env.get_action_grad(start_cur)
        action_grad = torch.from_numpy(action_grad)
        state_grad = StateFormat.frame2state(frame_grad)
        return state_grad,action_grad
