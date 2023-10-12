import torch
import torch.distributions as dist

def select_action(probs,drug1_sent_num,drug2_sent_num,old_actions_list,bad_list,deterministic,use_stop):
    new_bad_list = [False for _ in range(len(bad_list))]
    stop_num = 0
    example_num = len(old_actions_list)
    if use_stop:
        sent_num = (drug1_sent_num+drug2_sent_num +1).tolist()
    else:
        sent_num = (drug1_sent_num+drug2_sent_num).tolist()
    # stop_list = [False for _ in range(len(sent_num))]
    temp = 0
    sent_idx = [0]
    for num in sent_num:
        temp+=num
        sent_idx.append(temp)
    action_list = []
    action_log_prob_list = []
    action_log_probs_list = []
    for i in range(example_num):
        if bad_list[i]:
            action_list.append(None)
            action_log_prob_list.append(None)
            action_log_probs_list.append(None)
        else:
            if len(old_actions_list[i]) == sent_num[i]:
                action_list.append(None) # TODO -10000 
                action_log_prob_list.append(None)
                action_log_probs_list.append(None)
                bad_list[i] = True
                new_bad_list[i] = True
            else:
                if isinstance(probs,list):
                    _probs = probs[i]
                else:
                    _probs = probs[sent_idx[i]:sent_idx[i+1]]
                _probs = torch.softmax(_probs,dim=0)
                for old_action in old_actions_list[i]:
                    _probs[old_action] = 0
                if deterministic:
                    idx = torch.argmax(_probs,dim=0).item()
                else:
                    idx = torch.multinomial(_probs, 1, replacement=False).tolist()[0]
                if idx in old_actions_list[i]:
                    assert False
                if idx == _probs.size()[0]-1:
                    # stop_list[i] = True
                    stop_num+=1
                _probs = torch.log(_probs)
                action_list.append(idx)
                action_log_prob_list.append(_probs[idx])
                action_log_probs_list.append(_probs)
    if not use_stop:
        stop_num = 0
    return action_list,action_log_prob_list,action_log_probs_list,stop_num,bad_list,new_bad_list

def select_action_wo_stop(probs,drug1_sent_num,drug2_sent_num,old_actions_list,bad_list,deterministic):
    example_num = len(old_actions_list)
    sent_num = (drug1_sent_num+drug2_sent_num).tolist()
    temp = 0
    sent_idx = [0]
    for num in sent_num:
        temp+=num
        sent_idx.append(temp)
    action_list = []
    action_log_prob_list = []
    action_log_probs_list = []
    for i in range(example_num):
        if bad_list[i]:
            action_list.append(None)
            action_log_prob_list.append(None)
            action_log_probs_list.append(None)
        else:
            if len(old_actions_list[i]) == sent_num[i]: 
                action_list.append(-10000) 
                action_log_prob_list.append(None)
                action_log_probs_list.append(None)
            else:
                if isinstance(probs,list):
                    _probs = probs[i]
                else:
                    _probs = probs[sent_idx[i]:sent_idx[i+1]]
                _probs = torch.softmax(_probs,dim=0)
                for old_action in old_actions_list[i]:
                    _probs[old_action] = 0
                if deterministic:
                    idx = torch.argmax(_probs,dim=0).item()
                else:
                    idx = torch.multinomial(_probs, 1, replacement=False).tolist()[0]
                if idx in old_actions_list[i]:
                    assert False
                _probs = torch.log(_probs)
                action_list.append(idx)
                action_log_prob_list.append(_probs[idx])
                action_log_probs_list.append(_probs)
    return action_list,action_log_prob_list,action_log_probs_list

def return_action_log_prob(probs,sent_num,action_batch):
    action_log_prob = []
    temp = 0
    sent_idx = [0]
    entropies = []
    for num in sent_num:
        temp+=num
        sent_idx.append(temp)
    for i in range(len(sent_num)):
        _probs = probs[sent_idx[i]:sent_idx[i+1]]
        _probs = torch.softmax(_probs,dim=0)
        dist_i = dist.Categorical(_probs)
        entropy_i = dist_i.entropy()
        entropies.append(entropy_i)
        _probs = torch.log(_probs)
        action_log_prob.append(_probs[action_batch[i]])
    action_log_prob = torch.stack(action_log_prob,dim=0)
    entropies = torch.stack(entropies,dim=0)
    return action_log_prob,entropies
    

def trans_objects2list_with_none(objects,bad_list):
    object_list = []
    idx = 0
    for i in range(len(bad_list)):
        if bad_list[i]:
            object_list.append(None)
        else:
            object_list.append(objects[idx])
            idx+=1
    return object_list