import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PPO():
    def __init__(self,
                 actor_critic,
                 optimizer,
                 args):

        self.actor_critic = actor_critic

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch

        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

        self.max_grad_norm = args.max_grad_norm
        self.use_clipped_value_loss = args.use_clipped_value_loss

        self.optimizer = optimizer # optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts,globalstep):
        advantages = [None for i in range(rollouts.example_num)]
        for i in range(rollouts.example_num):
            advantages[i] = [a-b for a,b in zip(rollouts.returns[i],rollouts.value_preds_list[i][:-1])]

        flatten_advantages = []
        for item in advantages:
            flatten_advantages.extend(item)

        flatten_advantages = np.array(flatten_advantages)
        flatten_advantages_mean = flatten_advantages.mean()
        flatten_advantages_std = flatten_advantages.std()
        
        for i in range(rollouts.example_num):
            for j in range(len(advantages[i])):
                advantages[i][j] = (advantages[i][j]-flatten_advantages_mean)/(flatten_advantages_std+1e-5)


        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)
            if data_generator == None:
                return value_loss_epoch, action_loss_epoch,dist_entropy_epoch,globalstep

            for sample in data_generator:
                obs_input_ids_batch, obs_attention_mask_batch,example_index_batch,\
                action_batch,old_action_log_prob_batch, value_pred_batch, return_batch, advantage_batch = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_prob, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_input_ids_batch,
                    obs_attention_mask_batch,
                    example_index_batch,
                    action_batch,
                    rollouts)


                ratio = torch.exp(action_log_prob -
                                  old_action_log_prob_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * advantage_batch
                # action_loss = -torch.min(surr1, surr2).mean()
                action_loss = -surr2.mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_pred_batch + \
                        (values - value_pred_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -dist_entropy * self.entropy_coef).backward() #  
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                globalstep+=1

                value_loss_epoch += self.value_loss_coef * value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += -self.entropy_coef * dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch,dist_entropy_epoch,globalstep
