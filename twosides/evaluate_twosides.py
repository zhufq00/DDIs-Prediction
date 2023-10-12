import torch
import random
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from sklearn import metrics
from tools.ppo_utils import select_action,select_action_wo_stop
from sklearn.metrics import roc_auc_score, average_precision_score



def compute_logits(args,test_dataloader,env_model,random_act,actor_critic):
    logits_list = []
    labels_list = []
    idxs_list = []

    with torch.no_grad():
        batch_buffer = []
        eval_actions_batch_size = args.eval_actions_batch_size
        batch_sent_num = 0
        stop_num = 0

        null_sent_instance_list = []
        print('len data={},len dataset={},local_rank = {}'.format(len(test_dataloader),len(test_dataloader.dataset),args.local_rank))
        for step, batch in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
            if batch == 0:
                if step != len(test_dataloader)-1:
                    continue
            batch = tuple(t.to(args.device) for t in batch)
            if batch[2].item()+batch[3].item() == 0:
                null_sent_instance_list.append(batch)
                if step != len(test_dataloader)-1:
                    continue
            else:
                batch_buffer.append(batch)
            batch_sent_num+=batch[0].size()[1]
            if batch_sent_num>eval_actions_batch_size:
                batch_sent_num = batch[0].size()[1]
                batch_list = batch_buffer[0:-1]
                batch_buffer = batch_buffer[-1:]
            elif step==(len(test_dataloader)-1):
                batch_list = batch_buffer
            else:
                continue
            example_num = len(batch_list)
            action_input_ids = torch.cat([batch[0] for batch in batch_list],dim=1)
            action_attention_mask = torch.cat([batch[1] for batch in batch_list],dim=1)
            drug1_sent_num = torch.cat([batch[2] for batch in batch_list],dim=0)
            drug2_sent_num = torch.cat([batch[3] for batch in batch_list],dim=0)
            idxs = torch.cat([batch[4] for batch in batch_list],dim=0).tolist()
            rel_ids = torch.cat([batch[5] for batch in batch_list],dim=0)
            idxs_list.append(idxs)

            old_actions_list = [[] for _ in range(example_num)]

            action_num = 0

            final_state_input_ids_list = [None for _ in range(example_num)]
            final_state_attention_mask_list = [None for _ in range(example_num)]
            bad_list = [False for _ in range(example_num)]

            while True:
                state_input_ids_list = []
                state_attention_mask_list = []
                for i,(old_actions,idx) in enumerate(zip(old_actions_list,idxs)):
                    if not bad_list[i]:

                        state_input_ids,state_attention_mask,bad,stop = test_dataloader.dataset.get_state_by_actions(old_actions,idx)
                        if bad.item() == 1:
                            bad_list[i] = True
                            final_state_input_ids_list[i] = latest_state_input_ids[i].unsqueeze(0)
                            final_state_attention_mask_list[i] = latest_state_attention_mask[i].unsqueeze(0)
                        elif stop.item() == 1:
                            bad_list[i] = True
                            final_state_input_ids_list[i] = state_input_ids.unsqueeze(0)
                            final_state_attention_mask_list[i] = state_attention_mask.unsqueeze(0)
                        else:
                            state_input_ids_list.append(state_input_ids.unsqueeze(0))
                            state_attention_mask_list.append(state_attention_mask.unsqueeze(0))
                    if bad_list[i]:
                        state_input_ids_list.append(final_state_input_ids_list[i])
                        state_attention_mask_list.append(final_state_attention_mask_list[i])   

                if all(bad_list)==True:
                    break
                latest_state_input_ids = torch.cat(state_input_ids_list,dim=0)
                latest_state_attention_mask = torch.cat(state_attention_mask_list,dim=0)
    
                if not random_act:
                    if args.multi_gpu:
                        probs = actor_critic.module.return_probs_list(action_input_ids,action_attention_mask,latest_state_input_ids,latest_state_attention_mask,drug1_sent_num,drug2_sent_num)
                    else:
                        probs = actor_critic.return_probs_list(action_input_ids,action_attention_mask,latest_state_input_ids,latest_state_attention_mask,drug1_sent_num,drug2_sent_num)
                else:
                    probs = torch.ones((action_input_ids.size()[1])).to(action_input_ids.device)
                action_list,_,_,single_batch_stop_num,bad_list,new_bad_list = select_action(probs,drug1_sent_num,drug2_sent_num,old_actions_list,bad_list,deterministic=True,use_stop=args.use_stop)
                stop_num += single_batch_stop_num
                for i,action in enumerate(action_list):
                    if new_bad_list[i]:
                        final_state_input_ids_list[i] = latest_state_input_ids[i].unsqueeze(0)
                        final_state_attention_mask_list[i] = latest_state_attention_mask[i].unsqueeze(0)
                old_actions_list = [old_actions+[action] for old_actions, action in zip(old_actions_list,action_list)]
                action_num += 1
                # print(action_num)

            final_state_input_ids = torch.cat(final_state_input_ids_list,dim=0)
            final_state_attention_mask = torch.cat(final_state_attention_mask_list,dim=0)

            if args.multi_gpu:
                logits,_ = env_model.module.forward(final_state_input_ids,final_state_attention_mask,rel_ids)
            else:
                logits,_ = env_model(final_state_input_ids,final_state_attention_mask,rel_ids)
            logits_list.append(logits)
            labels_list.append(rel_ids)
        
        if len(null_sent_instance_list)!=0:
            print('null_sent_instance_list = {},rank = {}'.format(len(null_sent_instance_list),args.local_rank))
            idxs = torch.cat([batch[4] for batch in null_sent_instance_list],dim=0).tolist()
            state_input_ids,state_attention_mask,rel_ids = test_dataloader.dataset.get_vanilla_prompt(idxs)
            if args.multi_gpu:
                logits,_ = env_model.module.forward(state_input_ids,state_attention_mask,rel_ids)
            else:
                logits,_ = env_model(state_input_ids,state_attention_mask,rel_ids)
            logits_list.append(logits)
            labels_list.append(rel_ids)
            idxs_list.append(idxs)
    
    return logits_list,labels_list,idxs_list
    



def evaluate_twosides(pos_test_dataloader,neg_test_dataloader,actor_critic,env_model,args,logger,random_act=False):
    actor_critic.eval()  

    pos_scores_list,pos_labels_list,pos_idxs_list = compute_logits(args,pos_test_dataloader,env_model,random_act,actor_critic)
    neg_scores_list,neg_labels_list,neg_idxs_list= compute_logits(args,neg_test_dataloader,env_model,random_act,actor_critic)

    rerange_pos_scores = {}
    rerange_neg_scores = {}
    rerange_pos_labels = {}
    rerange_neg_labels = {}
    

    for pos_scores, idxs,labels in zip(pos_scores_list,pos_idxs_list,pos_labels_list):
        for i, idx in enumerate(idxs):
            rerange_pos_scores[idx] = pos_scores[i]
            rerange_pos_labels[idx] = labels[i]

    for neg_scores, idxs,labels in zip(neg_scores_list,neg_idxs_list,neg_labels_list):
         for i, idx in enumerate(idxs):
            rerange_neg_scores[idx] = neg_scores[i]
            rerange_neg_labels[idx] = labels[i]

    pos_scores_list = []
    for i in range(len(rerange_pos_scores)):
        pos_scores_list.append(rerange_pos_scores[i])

    neg_scores_list = []
    for i in range(len(rerange_neg_scores)):
        neg_scores_list.append(rerange_neg_scores[i])

    pos_labels_list = []
    for i in range(len(rerange_pos_labels)):
        pos_labels_list.append(rerange_pos_labels[i])

    neg_labels_list = []
    for i in range(len(rerange_neg_labels)):
        neg_labels_list.append(rerange_neg_labels[i])

    pos_scores = torch.cat([item.unsqueeze(0) for item in pos_scores_list],dim=0)
    neg_scores = torch.cat([item.unsqueeze(0) for item in neg_scores_list],dim=0)
    pos_labels = torch.cat([item.unsqueeze(0) for item in pos_labels_list],dim=0)
    neg_labels = torch.cat([item.unsqueeze(0) for item in neg_labels_list],dim=0)

    assert torch.equal(pos_labels,neg_labels)

    pos_scores = np.array(pos_scores.cpu())
    neg_scores = np.array(neg_scores.cpu())
    pos_labels = np.array(pos_labels.cpu())
    neg_labels = np.array(neg_labels.cpu())

    

    labels = pos_labels

    pred_class = {}
    for r in range(200):
        index = labels[:,r] > 0 
        pred_class[r] = {'score': list(pos_scores[index,r]) + list(neg_scores[index,r]), 
                'preds': list((pos_scores[index,r] > 0.5).astype('int')) + list((neg_scores[index,r]>0.5).astype('int')),
                'label': [1] * np.sum(index) + [0] * np.sum(index)}

    roc_auc = []
    prc_auc = []
    ap = []
    for r in range(200):
        label = pred_class[r]['label']
        score = pred_class[r]['score']
        sort_label = np.array(sorted(zip(score, label), reverse=True))
        roc_auc.append(roc_auc_score(label, score))
        prc_auc.append(average_precision_score(label, score))
        k = int(len(label)//2)
        apk = np.sum(sort_label[:k,1])
        ap.append(apk/k)

    pred_class_pos = {}
    for r in range(200):
        pred_class_pos[r] = {'score': list(pos_scores[:,r]), 
                # 'preds': list((pos_scores[r] > 0.5).astype('int')),
                'label': labels[:,r]}

    roc_auc_pos = []
    prc_auc_pos = []
    ap_pos = []
    for r in range(200):
        label = pred_class_pos[r]['label']
        score = pred_class_pos[r]['score']
        sort_label = np.array(sorted(zip(score, label), reverse=True))
        roc_auc_pos.append(roc_auc_score(label, score))
        prc_auc_pos.append(average_precision_score(label, score))

        def calculate_ap_at_k(y_true, y_scores, k):

            sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
            

            sorted_indices = sorted_indices[:k]
            

            num_relevant = 0
            precision_sum = 0.0
            
            for i, index in enumerate(sorted_indices):
                if y_true[index] == 1:
                    num_relevant += 1
                    precision = num_relevant / (i + 1)  
                    precision_sum += precision
            

            ap = precision_sum / num_relevant if num_relevant > 0 else 0.0
            
            return ap
        k = int(len(label)//2)
        ap_pos.append(calculate_ap_at_k(label,score,k))
    

    return np.mean(roc_auc), np.mean(prc_auc), np.mean(ap),np.mean(roc_auc_pos), np.mean(prc_auc_pos), np.mean(ap_pos)



    

    pos_scores_list.append(pos_logits)
    neg_scores_list.append(neg_logits)
    labels_list.append(rel_ids)
            # correct = torch.eq(predicts.flatten(), rel_ids.flatten()).float()         
            # acc = correct.sum().item() / len(correct)

            # y_pred.extend(predicts.flatten().tolist())
            # y_true.extend(rel_ids.flatten().tolist())
            # avg_acc.append(acc)



    if args.multi_gpu:
        gather_objects = [None for _ in range(torch.distributed.get_world_size())]
        gather_objects[args.local_rank] = [y_pred,y_true,stop_num]

    acc = metrics.accuracy_score(y_true,y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average=None).mean() 
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    print('len = {},len dataset={},acc = {} f1 = {}, kappa = {} local_rank = {}'.format(len(y_true),len(test_dataloader.dataset)-test_dataloader.dataset.pad_num,acc,f1,kappa,torch.distributed.get_rank()))
    
    if args.multi_gpu:
        output = [None for _ in gather_objects]
        dist.gather_object(
            gather_objects[dist.get_rank()],
            output if dist.get_rank() == 0 else None,
            dst=0
        )
        dist.barrier()
    if args.local_rank in [-1,0]:
        # Assumes world_size of 3.
        if args.multi_gpu:
            y_true = []
            y_pred = []
            stop_num = 0
            for data in output:
                y_pred.extend(data[0])
                y_true.extend(data[1])
                stop_num += data[2]
            acc = metrics.accuracy_score(y_true,y_pred)
            f1 = metrics.f1_score(y_true, y_pred, average=None).mean() 
            kappa = metrics.cohen_kappa_score(y_true, y_pred)

        logger.info("acc = {:>5,}".format(acc,'.4f'))
        logger.info("f1 = {}".format(f1,'.4f'))
        logger.info("kappa = {}".format(kappa,'.4f'))
        logger.info("len = {}".format(len(y_true)))
        logger.info("len dataset= {}".format(test_dataloader.dataset.true_data_len))
        logger.info("stop_num = {}".format(stop_num))
    return f1


def evaluate_vanilla(test_dataloader,actor_critic,env_model,args,logger):
    avg_acc = []
    actor_critic.eval()  

    y_pred = []
    y_true = []

    with torch.no_grad():
        batch_buffer = []
        eval_actions_batch_size = args.eval_actions_batch_size
        batch_sent_num = 0
        for step, batch in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
            if batch == 0:
                continue
            batch = tuple(t.to(args.device) for t in batch)
            batch_buffer.append(batch)
            batch_sent_num+=batch[0].size()[1]
            if batch_sent_num>eval_actions_batch_size:
                batch_sent_num = batch[0].size()[1]
                batch_list = batch_buffer[0:-1]
                batch_buffer = batch_buffer[-1:]
            elif step==(len(test_dataloader)-1):
                batch_list = batch_buffer
            else:
                continue
            example_num = len(batch_list)

            idxs = torch.cat([batch[4] for batch in batch_list],dim=0).tolist()

            state_input_ids,state_attention_mask,rel_ids = test_dataloader.dataset.get_vanilla_prompt(idxs)
            if args.multi_gpu:
                predicts,_ = env_model.module.forward(state_input_ids,state_attention_mask,rel_ids)
            else:
                predicts,_ = env_model(state_input_ids,state_input_ids,rel_ids)
            y_pred.extend(predicts.flatten().tolist())
            y_true.extend(rel_ids.flatten().tolist())

    if args.multi_gpu:
        gather_objects = [None for _ in range(torch.distributed.get_world_size())]
        gather_objects[args.local_rank] = [y_pred,y_true]

    acc = metrics.accuracy_score(y_true,y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average=None).mean() 
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    print('len = {},len dataset={},acc = {} f1 = {}, kappa = {} local_rank = {}'.format(len(y_true),test_dataloader.dataset.true_data_len,acc,f1,kappa,torch.distributed.get_rank()))
    
    if args.multi_gpu:
        output = [None for _ in gather_objects]
        dist.gather_object(
            gather_objects[dist.get_rank()],
            output if dist.get_rank() == 0 else None,
            dst=0
        )
        dist.barrier()
    if args.local_rank in [-1,0]:
        # Assumes world_size of 3.
        if args.multi_gpu:
            y_true = []
            y_pred = []
            for data in output:
                y_pred.extend(data[0])
                print(len(data[0]))
                y_true.extend(data[1])
            acc = metrics.accuracy_score(y_true,y_pred)
            f1 = metrics.f1_score(y_true, y_pred, average=None).mean() 
            kappa = metrics.cohen_kappa_score(y_true, y_pred)

        logger.info("acc = {:>5,}".format(acc,'.4f'))
        logger.info("f1 = {}".format(f1,'.4f'))
        logger.info("kappa = {}".format(kappa,'.4f'))
        logger.info("len = {}".format(len(y_true)))
        logger.info("len dataset= {}".format(test_dataloader.dataset.true_data_len))
    return f1