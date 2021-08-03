from plb.engine.taichi_env import TaichiEnv
import torch


class TaichiLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, taichi_env: TaichiEnv, state, action, cur=None):
        start_cur = taichi_env.simulator.cur if cur == None else cur
        taichi_env.set_state(state.numpy(),taichi_env.softness,False,start_cur)
        taichi_env.step(action=action.numpy())
        end_cur = taichi_env.simulator.cur # Point to the last frame in this step
        next_state = taichi_env.get_current_state()['state']
        ctx.save_for_backward(start_cur,end_cur,taichi_env)
        return torch.from_numpy(next_state)

    @staticmethod
    def backward(ctx,next_state_grad):
        start_cur,end_cur,taichi_env = ctx.saved_tensors
        taichi_env.set_state_grad(end_cur,next_state_grad.numpy())
        taichi_env.backprop(end_cur,start_cur) # @David
        state_grad = taichi_env.get_state_grad(start_cur) # Get gradient for ALL state
        action_grad = taichi_env.get_action_grad(start_cur)
        return state_grad,action_grad