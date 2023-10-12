import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self,example_num,use_stop):
        self.actions_list = [[] for _ in range(example_num)]
        self.value_preds_list = [[] for _ in range(example_num)]
        self.actions_log_prob_list = [[] for _ in range(example_num)]
        self.steps_reward_list = [[] for _ in range(example_num)]
        self.final_reward_list = [[] for _ in range(example_num)]
        self.obs_input_ids_list = [[] for _ in range(example_num)]
        self.obs_attention_mask_list = [[] for _ in range(example_num)]
        self.returns = [[] for _ in range(example_num)]
        self.example_num = example_num
        self.max_step = 0
        self.use_stop = use_stop
    
    def initialize_example_actions(self,action_input_ids,action_attention_mask,drug1_sent_num,drug2_sent_num):
        if self.use_stop:
            sent_num = (drug1_sent_num+drug2_sent_num + 1).tolist()
        else:
            sent_num = (drug1_sent_num+drug2_sent_num).tolist()
        self.action_input_ids_list = []
        self.action_attention_mask_list = []
        self.sent_num_list = sent_num
        temp = 0
        sent_idx = [0]
        for num in sent_num:
            temp+=num
            sent_idx.append(temp)
        for i in range(len(sent_num)):
                self.action_input_ids_list.append(action_input_ids[sent_idx[i]:sent_idx[i+1]])
                self.action_attention_mask_list.append(action_attention_mask[sent_idx[i]:sent_idx[i+1]])

    def insert(self, 
               action_list,
               value_list,
               action_log_prob_list,
               step_reward_list,
               state_input_ids_list,
               state_attention_mask_list,
               bad_list,
               new_bad_list,
               stop_list):
        for i,(action,value,action_log_prob) in enumerate(zip(\
            action_list,value_list,action_log_prob_list
        )):
            if not bad_list[i]:
                self.value_preds_list[i].append(value.item())
                self.actions_list[i].append(action)
                self.actions_log_prob_list[i].append(action_log_prob.item())
                self.steps_reward_list[i].append(step_reward_list[i].item())
                self.obs_input_ids_list[i].append(state_input_ids_list[i])
                self.obs_attention_mask_list[i].append(state_attention_mask_list[i])
            elif new_bad_list[i]:
                self.value_preds_list[i].append(value.item())
            if stop_list[i]:
                assert False
                self.value_preds_list[i].append(value.item())
                self.actions_list[i].append(action)
                self.actions_log_prob_list[i].append(action_log_prob.item())
                self.steps_reward_list[i].append(step_reward_list[i].item())
                self.obs_input_ids_list[i].append(state_input_ids_list[i])
                self.obs_attention_mask_list[i].append(state_attention_mask_list[i])
        self.max_step+=1
    
    def insert_initialize_reward_list(self,initialize_reward_list):
        for i in range(len(initialize_reward_list)):
            self.steps_reward_list[i].append(initialize_reward_list[i])

    def compute_returns(self,
                        use_gae,
                        gamma,
                        gae_lambda):
        
        for i in range(self.example_num):
            action_num = len(self.actions_list[i])
            steps_reward_num = len(self.steps_reward_list[i])
            assert action_num+1==steps_reward_num
            for j in range(action_num):
                if self.use_stop:
                    if self.actions_list[i][j] == self.sent_num_list[i]-1:
                        if j!=action_num-1:
                            assert False
                        self.final_reward_list[i].append(self.steps_reward_list[i][j+1]-self.steps_reward_list[i][j])
                        assert abs(self.steps_reward_list[i][j+1]-self.steps_reward_list[i][j]) < 0.0001
                    else:
                        self.final_reward_list[i].append(self.steps_reward_list[i][j+1]-self.steps_reward_list[i][j])
                else:
                    self.final_reward_list[i].append(self.steps_reward_list[i][j+1]-self.steps_reward_list[i][j])
        
        if use_gae:
            for i in range(self.example_num):
                gae = 0
                for step in reversed(range(len(self.final_reward_list[i]))):
                    delta = self.final_reward_list[i][step] + gamma * self.value_preds_list[i][
                        step + 1] - self.value_preds_list[i][step]
                    gae = delta + gamma * gae_lambda  * gae
                    self.returns[i].insert(0,gae + self.value_preds_list[i][step])
        # print()
        # else:
        #     self.returns[-1] = next_value
        #     for step in reversed(range(self.rewards.size(0))):
        #         self.returns[step] = self.returns[step + 1] * \
        #             gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        
        flatten_obs_input_ids_list = []
        flatten_obs_attention_mask_list = []
        flatten_action_list = []
        flatten_value_pred_list = []
        flatten_return_list = []
        flatten_old_action_log_prob_list = []
        flatten_advantage_list = []
        batch_index2example_index_dict = dict()
        batch_index = 0
        for i in range(self.example_num):
            for j in range(len(advantages[i])):
                batch_index2example_index_dict[batch_index] = i
                batch_index += 1
                flatten_obs_input_ids_list.append(self.obs_input_ids_list[i][j])
                flatten_obs_attention_mask_list.append(self.obs_attention_mask_list[i][j])
                flatten_action_list.append(self.actions_list[i][j])
                flatten_old_action_log_prob_list.append(self.actions_log_prob_list[i][j])
                flatten_value_pred_list.append(self.value_preds_list[i][j])
                flatten_return_list.append(self.returns[i][j])
                flatten_advantage_list.append(advantages[i][j])
        
        flatten_obs_input_ids = torch.cat(flatten_obs_input_ids_list,dim=0)
        flatten_obs_attention_mask = torch.cat(flatten_obs_attention_mask_list,dim=0)
        flatten_action = torch.tensor(flatten_action_list).to(flatten_obs_input_ids.device)
        flatten_old_action_log_prob = torch.tensor(flatten_old_action_log_prob_list).to(flatten_obs_input_ids.device)
        flatten_value_pred = torch.tensor(flatten_value_pred_list).to(flatten_obs_input_ids.device)
        flatten_return = torch.tensor(flatten_return_list).to(flatten_obs_input_ids.device)
        flatten_advantage = torch.tensor(flatten_advantage_list).to(flatten_obs_input_ids.device)
       

        batch_size = len(flatten_advantage_list)
        mini_batch_size = batch_size // num_mini_batch

        if mini_batch_size == 0:
            return None

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_input_ids_batch = flatten_obs_input_ids[indices]
            obs_attention_mask_batch = flatten_obs_attention_mask[indices]
            action_batch = flatten_action[indices]
            old_action_log_prob_batch = flatten_old_action_log_prob[indices]
            value_pred_batch = flatten_value_pred[indices]
            return_batch = flatten_return[indices]
            advantage_batch = flatten_advantage[indices]
            # action_input_ids_list = [] 
            # action_attention_mask_list = []
            # sent_num_list = []
            example_index_list = []
            for batch_index in indices:
                example_index = batch_index2example_index_dict[batch_index]
                example_index_list.append(example_index)
                # action_input_ids_list.append(self.action_input_ids_list[example_index])
                # action_attention_mask_list.append(self.action_attention_mask_list[example_index])
                # sent_num_list.append(self.sent_num_list[example_index])
            example_index_batch = torch.tensor(example_index_list).to(obs_input_ids_batch.device)
            # action_input_ids_batch = torch.cat(action_input_ids_list,dim=0)
            # action_attention_mask_batch = torch.cat(action_attention_mask_list,dim=0)
            # sent_num_batch = torch.tensor(sent_num_list).to(obs_input_ids_batch.device)
            


            yield obs_input_ids_batch, obs_attention_mask_batch,example_index_batch,\
                action_batch,old_action_log_prob_batch, value_pred_batch, return_batch, advantage_batch
