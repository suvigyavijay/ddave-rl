from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
import torch as th

device = 'cuda'
device = th.device(device)

class RLeXploreCallback(BaseCallback):
    """
    A custom callback for the RLeXplore toolkit. 
    """
    def __init__(self, irs, verbose=0):
        super(RLeXploreCallback, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        if isinstance(self.model, OnPolicyAlgorithm):
            self.buffer = self.model.rollout_buffer
        # TODO: support for off-policy algorithms will be added soon!!!

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        observations = self.locals["obs_tensor"]
        device = observations.device
        device = 'cuda'
        device = th.device(device)
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:
        # ===================== compute the intrinsic rewards ===================== #
        obs = th.as_tensor(self.buffer.observations, device=device)
        actions = th.as_tensor(self.buffer.actions, device=device)
        rewards = th.as_tensor(self.buffer.rewards, device=device)
        dones = th.as_tensor(self.buffer.episode_starts, device=device)
        print(obs.shape, actions.shape, rewards.shape, dones.shape, obs.shape)
        intrinsic_rewards = self.irs.compute(samples=dict(observations=obs, 
                                                     actions=actions, 
                                                     rewards=rewards, 
                                                     terminateds=dones,
                                                     truncateds=dones, 
                                                     next_observations=obs
                                                     ))
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #