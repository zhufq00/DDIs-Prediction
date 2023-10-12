import torch
import random
import numpy as np
import pickle
import torch.distributed as dist
from tqdm import tqdm
from sklearn import metrics
from tools.ppo_utils import select_action,select_action_wo_stop


def prepare_data(idxs,actor_critic,train_dataloader,args,random_act=False):
    null_sent_instance_list = []

    return_input_ids_list = []
    return_attention_mask_list = []
    return_rel_ids_list = []

    batch_buffer = []
    batch_sent_num = 0
    stop_num = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
            if batch == 0:
                if step != len(train_dataloader)-1:
                    continue
            batch = tuple(t.to(args.device) for t in batch)
            if batch[2].item()+batch[3].item() == 0:
                null_sent_instance_list.append(batch)
                if step != len(train_dataloader)-1:
                    continue
            else:
                batch_buffer.append(batch)
            batch_sent_num+=batch[0].size()[1]
            if batch_sent_num>args.eval_actions_batch_size:
                batch_sent_num = batch[0].size()[1]
                batch_list = batch_buffer[0:-1]
                batch_buffer = batch_buffer[-1:]
            elif step==(len(train_dataloader)-1):
                batch_list = batch_buffer
            else:
                continue
            action_input_ids = torch.cat([batch[0] for batch in batch_list],dim=1)
            action_attention_mask = torch.cat([batch[1] for batch in batch_list],dim=1)
            drug1_sent_num = torch.cat([batch[2] for batch in batch_list],dim=0)
            drug2_sent_num = torch.cat([batch[3] for batch in batch_list],dim=0)
            idxs = torch.cat([batch[4] for batch in batch_list],dim=0).tolist()
            rel_ids = torch.cat([batch[5] for batch in batch_list],dim=0)


            sent_num = (drug1_sent_num+drug2_sent_num).tolist()

                # if args.random_act:
                #     probs = torch.ones(input_ids.size()[0:2]).to(input_ids.device)
            example_num = len(batch_list)
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
                        state_input_ids,state_attention_mask,bad,stop = train_dataloader.dataset.get_state_by_actions(old_actions,idx)
                        if bad.item() == 1:
                            bad_list[i] = True
                            old_actions_list[i] = old_actions_list[i][0:-1] # modified
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

            # final_state_input_ids = torch.cat(final_state_input_ids_list,dim=0)
            # final_state_attention_mask = torch.cat(final_state_attention_mask_list,dim=0)
            return_input_ids_list.extend(final_state_input_ids_list)
            return_attention_mask_list.extend(final_state_attention_mask_list)
            return_rel_ids_list.extend(rel_ids.tolist())
            
        if len(null_sent_instance_list)!=0:
            print('null_sent_instance_list = {},rank = {}'.format(len(null_sent_instance_list),args.local_rank))
            idxs = torch.cat([batch[4].unsqueeze(0) for batch in null_sent_instance_list],dim=0).tolist()
            state_input_ids,state_attention_mask,rel_ids = train_dataloader.dataset.get_vanilla_prompt(idxs)
            return_input_ids_list.extend(torch.split(state_input_ids,1,dim=0))
            return_attention_mask_list.extend(torch.split(state_attention_mask,1,dim=0))
            return_rel_ids_list.extend(rel_ids.tolist())
    
    combined_list = list(zip(return_input_ids_list, return_attention_mask_list, return_rel_ids_list))
    random.shuffle(combined_list)
    return_input_ids_list, return_attention_mask_list, return_rel_ids_list = zip(*combined_list)

    pickle.dump([return_input_ids_list,return_attention_mask_list,return_rel_ids_list],open('./data/drugbank/train_rl1_data.pkl','wb'))

    return # return_input_ids_list,return_attention_mask_list,return_rel_ids_list
        


def evaluate(test_dataloader,actor_critic,env_model,args,logger,random_act=False):
    avg_acc = []
    actor_critic.eval()  

    y_pred = []
    y_true = []

    data_list = []

    stop_num = 0

    with torch.no_grad():
        batch_buffer = []
        eval_actions_batch_size = args.eval_actions_batch_size
        batch_sent_num = 0

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
            # if args.use_stop:
            #     action_input_ids = torch.cat([batch[0][:,0:-1,:] for batch in batch_list],dim=1)
            #     action_attention_mask = torch.cat([batch[1][:,0:-1,:] for batch in batch_list],dim=1)
            # else:
            action_input_ids = torch.cat([batch[0] for batch in batch_list],dim=1)
            action_attention_mask = torch.cat([batch[1] for batch in batch_list],dim=1)
            drug1_sent_num = torch.cat([batch[2] for batch in batch_list],dim=0)
            drug2_sent_num = torch.cat([batch[3] for batch in batch_list],dim=0)
            idxs = torch.cat([batch[4] for batch in batch_list],dim=0).tolist()
            rel_ids = torch.cat([batch[5] for batch in batch_list],dim=0)

            if args.use_stop:
                sent_num = (drug1_sent_num+drug2_sent_num+1).tolist()
            else:
                sent_num = (drug1_sent_num+drug2_sent_num).tolist()

            # if args.random_act:
            #     probs = torch.ones(input_ids.size()[0:2]).to(input_ids.device)

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
                            old_actions_list[i] = old_actions_list[i][0:-1] # modified
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
                    probs = torch.rand((action_input_ids.size()[1])).to(action_input_ids.device)
                action_list,_,_,single_batch_stop_num,bad_list,new_bad_list = select_action(probs,drug1_sent_num,drug2_sent_num,old_actions_list,bad_list,deterministic=False,use_stop=args.use_stop)
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
                predicts,_ = env_model.module.compute_rewards(final_state_input_ids,final_state_attention_mask,rel_ids)
            else:
                predicts,_ = env_model.compute_rewards(final_state_input_ids,final_state_attention_mask,rel_ids)
            data_list.append([idxs,old_actions_list])

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

            with open('./data/drugbank/id2rel.txt','r') as file:
                id2rel = [item.strip() for item in file.readlines()]
            for i,(input_ids,p_id,id) in enumerate(zip(final_state_input_ids,predicts.tolist(),rel_ids.tolist())):
                with open('./cases/{}.txt'.format(i),'w') as file:
                    text = tokenizer.decode(input_ids.tolist(),skip_special_tokens=True)
                    file.write(text)
                    file.write("\n\nPredict interaction: {}".format(id2rel[p_id]))
                    file.write("\n\nTrue interaction: {}".format(id2rel[id]))

            
            
            correct = torch.eq(predicts.flatten(), rel_ids.flatten()).float()         
            acc = correct.sum().item() / len(correct)

            y_pred.extend(predicts.flatten().tolist())
            y_true.extend(rel_ids.flatten().tolist())
            avg_acc.append(acc)

        if len(null_sent_instance_list)!=0:
            print('null_sent_instance_list = {},rank = {}'.format(len(null_sent_instance_list),args.local_rank))
            idxs = torch.cat([batch[4] for batch in null_sent_instance_list],dim=0).tolist()
            state_input_ids,state_attention_mask,rel_ids = test_dataloader.dataset.get_vanilla_prompt(idxs)
            if args.multi_gpu:
                predicts,_ = env_model.module.compute_rewards(state_input_ids,state_attention_mask,rel_ids)
            else:
                predicts,_ = env_model.compute_rewards(state_input_ids,state_input_ids,rel_ids)
            y_pred.extend(predicts.flatten().tolist())
            y_true.extend(rel_ids.flatten().tolist())
            data_list.append([idxs,[[] for _ in range(len(idxs))]])

    if args.multi_gpu:
        gather_objects = [None for _ in range(torch.distributed.get_world_size())]
        gather_objects[args.local_rank] = [y_pred,y_true,stop_num,data_list]

    acc = metrics.accuracy_score(y_true,y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average=None).mean() 
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    print('len = {},len dataset={},acc = {} f1 = {}, kappa = {} local_rank = {}'.format(len(y_true),len(test_dataloader.dataset)-test_dataloader.dataset.pad_num,acc,f1,kappa,args.local_rank))
    
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
            data_list = []
            stop_num = 0
            for data in output:
                y_pred.extend(data[0])
                y_true.extend(data[1])
                data_list.extend(data[3])
                stop_num += data[2]
            acc = metrics.accuracy_score(y_true,y_pred)
            f1 = metrics.f1_score(y_true, y_pred, average=None).mean() 
            kappa = metrics.cohen_kappa_score(y_true, y_pred)
        import pickle
        pickle.dump(data_list,open('./data/ddi/dev_rl0_data.pkl','wb'))

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
                predicts,_ = env_model.module.compute_rewards(state_input_ids,state_attention_mask,rel_ids)
            else:
                predicts,_ = env_model.compute_rewards(state_input_ids,state_input_ids,rel_ids)
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