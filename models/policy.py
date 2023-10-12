import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.roberta_actor import roberta_actor
from models.roberta_critic import roberta_critic
from transformers import RobertaModel,RobertaForSequenceClassification
from tools.ppo_utils import select_action,trans_objects2list_with_none,return_action_log_prob

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        # self.actor_model = roberta_actor(args) # input : all possible actions and state; output: probs
        # self.critic_model = roberta_critic(args) # input : state; output: value
        self.llm = RobertaModel.from_pretrained(args.pretrained_model_path)
        self.args = args
        self.config = self.llm.config
        
        self.action_tran = nn.Linear(self.config.hidden_size,self.config.hidden_size)
        self.state_tran = nn.Linear(self.config.hidden_size,self.config.hidden_size)
        self.predict_porbs = nn.Linear(self.config.hidden_size*2,1)
        self.values = nn.Linear(self.config.hidden_size, 1)
        orthogonal_init(self.action_tran)
        orthogonal_init(self.state_tran)
        orthogonal_init(self.predict_porbs)
        orthogonal_init(self.values)



    def return_probs_list(self,action_input_ids,action_attention_mask,state_input_ids,state_attention_mask,drug1_sent_num,drug2_sent_num):
        action_input_ids = action_input_ids.squeeze()
        action_attention_mask = action_attention_mask.squeeze()
        action_encodings = self.llm(action_input_ids,action_attention_mask).last_hidden_state[:,0,:]
        state_encodings = self.llm(state_input_ids,state_attention_mask).last_hidden_state[:,0,:]

        sent_num = drug1_sent_num+drug2_sent_num # stop action 
        if self.args.use_stop:
            sent_num = sent_num + 1
        state_encodings_list = []
        for i,num in enumerate(sent_num.tolist()):
            state_encodings_list.append(state_encodings[i].unsqueeze(0).repeat(num,1))
        repeated_state_encodings = torch.cat(state_encodings_list,dim=0)

        probs = self.predict_porbs(torch.cat((action_encodings,repeated_state_encodings),dim=1))

        return probs.squeeze()

    def forward(self):
        raise NotImplementedError
    
    def evaluate_actions(self,
                        obs_input_ids_batch,
                        obs_attention_mask_batch,
                        example_index_batch,
                        action_batch,
                        rollouts):
        example_index_batch = example_index_batch.tolist()
        example_index_list = sorted(set(example_index_batch))
        batch_index2current_example_index = dict()
        for batch_index,example_index in enumerate(example_index_batch):
            batch_index2current_example_index[batch_index] = example_index_list.index(example_index)

        action_input_ids_list = []
        action_attention_mask_list = []
        sent_num_list = []
        for i in example_index_list:
            action_input_ids_list.append(rollouts.action_input_ids_list[i])
            action_attention_mask_list.append(rollouts.action_attention_mask_list[i])
            sent_num_list.append(rollouts.sent_num_list[i])
        action_input_ids = torch.cat(action_input_ids_list,dim=0)
        action_attention_mask = torch.cat(action_attention_mask_list,dim=0)

        action_encodings = self.llm(action_input_ids,action_attention_mask).last_hidden_state[:,0,:]

        action_encodings = self.action_tran(action_encodings)

        temp = 0
        sent_idx = [0]
        for num in sent_num_list:
            temp+=num
            sent_idx.append(temp)

        action_encodings_list = []

        for i in range(len(sent_num_list)):
            action_encodings_list.append(action_encodings[sent_idx[i]:sent_idx[i+1]])

        
        repeat_action_encodings_list = []
        repeat_sent_num_list = []
        for i in range(len(example_index_batch)):
            repeat_sent_num_list.append(sent_num_list[batch_index2current_example_index[i]])
            repeat_action_encodings_list.append(action_encodings_list[batch_index2current_example_index[i]])

        repeat_action_encodings = torch.cat(repeat_action_encodings_list,dim=0)

        state_encodings = self.llm(obs_input_ids_batch,obs_attention_mask_batch).last_hidden_state[:,0,:]

        state_encodings = self.state_tran(state_encodings)

        values = self.values(state_encodings)

        state_encodings_list = []
        for i,num in enumerate(repeat_sent_num_list):
            state_encodings_list.append(state_encodings[i].unsqueeze(0).repeat(num,1))
        repeated_state_encodings = torch.cat(state_encodings_list,dim=0)

        probs = self.predict_porbs(torch.cat((repeat_action_encodings,repeated_state_encodings),dim=1)).squeeze()
        action_log_prob,entropies = return_action_log_prob(probs,repeat_sent_num_list,action_batch)

        return values,action_log_prob,entropies.mean(),None


    def act(self, 
            action_input_ids,
            action_attention_mask,
            state_input_ids,
            state_attention_mask,
            drug1_sent_num,
            drug2_sent_num,
            old_actions_list,
            bad_list,
            deterministic=False):
        if self.args.use_stop:
            sent_num = (drug1_sent_num+drug2_sent_num + 1).tolist()
        else:
            sent_num = (drug1_sent_num+drug2_sent_num).tolist()
            
        temp = 0
        sent_idx = [0]
        for num in sent_num:
            temp+=num
            sent_idx.append(temp)

        filter_action_input_ids_list = []
        filter_action_attention_mask_list = []
        filter_state_input_ids_list = []
        filter_state_attention_mask_list = []

        for i in range(len(sent_num)):
            if not bad_list[i]:
                filter_action_input_ids_list.append(action_input_ids[sent_idx[i]:sent_idx[i+1]])
                filter_action_attention_mask_list.append(action_attention_mask[sent_idx[i]:sent_idx[i+1]])
                filter_state_input_ids_list.append(state_input_ids[i])
                filter_state_attention_mask_list.append(state_attention_mask[i])
        
        filter_action_input_ids = torch.cat(filter_action_input_ids_list,dim=0)
        filter_action_attention_mask = torch.cat(filter_action_attention_mask_list,dim=0)
        filter_state_input_ids = torch.cat(filter_state_input_ids_list,dim=0)
        filter_state_attention_mask = torch.cat(filter_state_attention_mask_list,dim=0)


        action_encodings = self.llm(filter_action_input_ids,filter_action_attention_mask).last_hidden_state[:,0,:]

        action_encodings = self.action_tran(action_encodings)

        state_encodings = self.llm(filter_state_input_ids,filter_state_attention_mask).last_hidden_state[:,0,:]

        state_encodings = self.state_tran(state_encodings)

        values = self.values(state_encodings)

        value_list = trans_objects2list_with_none(values,bad_list)

        state_encodings_list = []
        filter_sent_num_list = []
        for i in range(len(bad_list)):
            if not bad_list[i]:
                filter_sent_num_list.append(sent_num[i])
        for i,num in enumerate(filter_sent_num_list):
            state_encodings_list.append(state_encodings[i].unsqueeze(0).repeat(num,1))
        repeated_state_encodings = torch.cat(state_encodings_list,dim=0)

        probs = self.predict_porbs(torch.cat((action_encodings,repeated_state_encodings),dim=1)).squeeze()


        temp = 0
        filter_sent_idx = [0]
        for num in filter_sent_num_list:
            temp+=num
            filter_sent_idx.append(temp)
        
        probs_list = []
        probs_idx = 0
        for i in range(len(bad_list)):
            if bad_list[i]:
                probs_list.append(None)
            else:
                probs_list.append(probs[filter_sent_idx[probs_idx]:filter_sent_idx[probs_idx+1]])
                probs_idx+=1
        # end
        action_list,action_log_prob_list,action_log_probs_list,stop_num,bad_list,new_bad_list = select_action(probs_list,drug1_sent_num,drug2_sent_num,old_actions_list,bad_list,deterministic=False,use_stop = self.args.use_stop)

        return value_list, action_list, action_log_prob_list,bad_list,new_bad_list

    def get_value(self, state_input_ids_list,state_attention_mask_list,bad_list):
        filter_state_input_ids_list = []
        filter_state_attention_mask_list = []
        for i in range(len(bad_list)):
            if not bad_list[i]:
                filter_state_input_ids_list.append(state_input_ids_list[i])
                filter_state_attention_mask_list.append(state_attention_mask_list[i])
        
        filter_state_input_ids = torch.cat(filter_state_input_ids_list,dim=0)
        filter_state_attention_mask = torch.cat(filter_state_attention_mask_list,dim=0)

        state_encodings = self.llm(filter_state_input_ids,filter_state_attention_mask).last_hidden_state[:,0,:]

        values = self.values(state_encodings)

        values_list = trans_objects2list_with_none(values,bad_list)

        return values_list

    # def evaluate_actions(self, inputs, rnn_hxs, masks, action):
    #     value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
    #     dist = self.dist(actor_features)

    #     action_log_probs = dist.log_probs(action)
    #     dist_entropy = dist.entropy().mean()

    #     return value, action_log_probs, dist_entropy, rnn_hxs

