import torch

class BaseRewardFunction:
    """Base class for reward functions"""
    def compute_reward(self, rollout_idx: int, rollout_episode: dict, 
                      ground_truth_batch: dict) -> float:
        """
        Args:
            rollout_idx: Index of the rollout in the batch
            rollout_episode: Dictionary containing rollout trajectory data
            ground_truth_batch: Batch containing ground-truth demonstration data
        Returns:
            Computed reward value
        """
        raise NotImplementedError

class SuccessReward(BaseRewardFunction):
    """Binary success reward (1 for successful, 0 otherwise)"""
    def compute_reward(self, rollout_idx: int, rollout_episode: dict, 
                      ground_truth_batch: dict) -> float:
        return 1.0 if rollout_episode.get('success', False) else 0.0