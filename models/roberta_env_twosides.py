import torch
import pickle
import torch.nn.functional as F
from torch import nn
from transformers import RobertaForSequenceClassification,RobertaModel
from tools.ppo_utils import trans_objects2list_with_none


# torch.set_printoptions(sci_mode=False)
class roberta_env_twosides(nn.Module):

    def __init__(self, args):
        super(roberta_env_twosides, self).__init__()

        self.llm = RobertaForSequenceClassification.from_pretrained(args.pretrained_model_path,num_labels=200)
        self.args = args
        self.config = self.llm.config
        self.num_labels = args.num_labels
        self.correct_bonus = 2.0
        self.incorrect_bonus = 1.8
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        # self.relation_name_input_ids,self.relation_name_attention_mask = pickle.load(open('./data/twosides/relation_name_tokenized.pkl','rb'))
    
    def forward(
        self,input_ids,attention_mask,labels
    ):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = torch.sigmoid(outputs.logits)

        loss_fun = nn.BCELoss(reduction='none')
        step_rewards = 1-loss_fun(logits,labels.float())
        step_rewards = torch.mean(step_rewards,dim=1)

        return logits, step_rewards

        # pooled_logits = outputs.logits
        # label_probs = torch.softmax(pooled_logits,dim=1)
        
        # predicts = torch.argmax(label_probs, axis=-1)

        # correct = (predicts == labels).float()
        # correct_probs = label_probs[torch.arange(batch_size), labels]
        
        # not_label_probs = label_probs

        # not_label_probs[torch.arange(batch_size), labels] = -1

        # # [batch_size, num_classes]
        # max_not_label_probs, _ = torch.max(not_label_probs, -1)
        # # [batch_size, 1]

        # # Compute piecewise gap reward
        # gap = correct_probs - max_not_label_probs

        # step_rewards = gap * (self.correct_bonus * correct + self.incorrect_bonus * (1 - correct))

        # return predicts, step_rewards

    def step(self,old_actions_list,action_list,idxs,rel_ids,bad_list,dataset):
        new_bad_list = [False for _ in range(len(bad_list))]
        old_actions_list = [old_actions+[action] for old_actions, action in zip(old_actions_list,action_list)]
        state_input_ids_list = []
        state_attention_mask_list = []
        stop_list = [False for _ in range(len(old_actions_list))]
        for i,(old_actions,idx) in enumerate(zip(old_actions_list,idxs)):
            if not bad_list[i]: # 超长度了
                state_input_ids,state_attention_mask,bad,stop = dataset.get_state_by_actions(old_actions,idx)
                stop_list[i] = stop
                if bad.item() == 1:
                    bad_list[i] = True
                    new_bad_list[i] = True
                else:
                    state_input_ids_list.append(state_input_ids.unsqueeze(0))
                    state_attention_mask_list.append(state_attention_mask.unsqueeze(0))
            if bad_list[i]:
                rel_ids[i] = None
                state_input_ids_list.append(None)
                state_attention_mask_list.append(None)
        
        if all(bad_list) == True:
            return None,None,None,bad_list,new_bad_list,stop_list

        filter_state_input_ids_list = list(filter(lambda x: x != None,state_input_ids_list))
        filter_state_attention_mask_list = list(filter(lambda x: x != None,state_attention_mask_list))
        filter_labels = list(filter(lambda x: x != None,rel_ids))

        filter_state_input_ids = torch.cat(filter_state_input_ids_list,dim=0)
        filter_state_attention_mask = torch.cat(filter_state_attention_mask_list,dim=0)
        filter_labels = torch.tensor(filter_labels).to(filter_state_input_ids.device)

        _,step_rewards = self.forward(filter_state_input_ids,filter_state_attention_mask,filter_labels)

        step_reward_list = trans_objects2list_with_none(step_rewards, bad_list)

        for i in range(len(old_actions_list)):
            if stop_list[i]:
                assert False
                bad_list[i] = True
                new_bad_list[i] = True


        return state_input_ids_list,state_attention_mask_list,step_reward_list,bad_list,new_bad_list,stop_list



        