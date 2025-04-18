import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class ActTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, action_target, dones, rtg, _, attention_mask, _ = self.get_batch(self.batch_size)
        state_target, reward_target = torch.clone(states), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=rtg[:,0],
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:,-1].reshape(-1, act_dim)

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target, action_target, reward_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

class StochasticActTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, action_target, dones, rtg, _, attention_mask, _ = self.get_batch(self.batch_size)
        state_target, reward_target = torch.clone(states), torch.clone(rewards)

        state_preds, action_dist, reward_preds = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=rtg[:,0],
        )

        action_preds = action_dist.mode
        action_target = action_target.reshape(-1, action_target.shape[-1])[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target, action_target, reward_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()