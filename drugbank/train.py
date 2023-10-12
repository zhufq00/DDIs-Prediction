import torch
import time
import math
import os
import pickle
import copy
import numpy as np
from collections import deque
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tools.common import optimizer_to,format_time
from tools.drugbank_dataset_ppo import drugbank_dataset_ppo
from tools.twosides_dataset_ppo import twosides_dataset_ppo
from drugbank.evaluate import evaluate,evaluate_vanilla
from twosides.evaluate_twosides import evaluate_twosides
from tools.storage import RolloutStorage
from tools.ppo import PPO
from drugbank.evaluate import prepare_data
from tqdm import tqdm
DATASET_MODEL_CLASS = {
    'drugbank': drugbank_dataset_ppo,
    'twosides': twosides_dataset_ppo
}


def train(args,train_dataloader,actor_critic,env_model,ac_optimizer,writer,logger,globalstep,eval_time,start_date):
    t0 = time.time()
    
    if args.eval_before_train:
        actor_critic.eval()
        # dev_data = DATASET_MODEL_CLASS[args.dataset](args,'dev')
        # dev_sampler = SequentialSampler(dev_data) if args.local_rank == -1 else DistributedSampler(dev_data)
        # dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1,num_workers=8)
        # dev_f1 = evaluate(dev_dataloader,actor_critic,env_model,args,logger,random_act = True)
        dev_data = DATASET_MODEL_CLASS[args.dataset](args,'dev')
        dev_sampler = SequentialSampler(dev_data) if args.local_rank == -1 else DistributedSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1,num_workers=8)
        dev_f1 = evaluate(dev_dataloader,actor_critic,env_model,args,logger,random_act = False)

    
    # if args.eval_before_train:
    #     actor_critic.eval()
    #     dev_data = drugbank_dataset_ppo(args,'dev')
    #     dev_sampler = SequentialSampler(dev_data) if args.local_rank == -1 else DistributedSampler(dev_data)
    #     dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1,num_workers=0)
    #     dev_f1 = evaluate(dev_dataloader,actor_critic,env_model,args,logger)
    if args.eval_vanilla_before_train:
        actor_critic.eval()
        dev_data = DATASET_MODEL_CLASS[args.dataset](args,'dev')
        dev_sampler = SequentialSampler(dev_data) if args.local_rank == -1 else DistributedSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1,num_workers=0)
        dev_f1 = evaluate_vanilla(dev_dataloader,actor_critic,env_model,args,logger)
        
    batch_buffer = []
    batch_sent_num = 0

    if args.multi_gpu:
        actor_critic_module = actor_critic.module
        env_model_module = env_model.module
    else:
        actor_critic_module = actor_critic
        env_model_module = env_model

    agent = PPO(actor_critic_module,ac_optimizer,args)

    episode_rewards = deque(maxlen=100)
    actor_critic.train()

    # env_input_ids_buffer = []
    # env_attention_mask_buffer = []
    # rel_ids_buffer = []
    idxs_buffer = []

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        # if batch[4].tolist()[0] in ignore_index_list:
        #     continue
        if batch[0].size()[1]-1 == 0:
            continue
        else:
            batch_buffer.append(batch)
        batch_sent_num+=batch[0].size()[1]
        if batch_sent_num>args.train_actions_batch_size:
            batch_sent_num = batch[0].size()[1]
            batch_list = batch_buffer[0:-1]
            batch_buffer = batch_buffer[-1:]
        elif step==(len(train_dataloader)-1):
            batch_list = batch_buffer
        else:
            continue

        if batch_list == []:
            continue

        action_input_ids = torch.cat([batch[0].squeeze() for batch in batch_list],dim=0)
        action_attention_mask = torch.cat([batch[1].squeeze() for batch in batch_list],dim=0)
        drug1_sent_num = torch.cat([batch[2] for batch in batch_list],dim=0)
        drug2_sent_num = torch.cat([batch[3] for batch in batch_list],dim=0)
        idxs = torch.cat([batch[4] for batch in batch_list],dim=0).tolist()
        rel_ids = torch.cat([batch[5] for batch in batch_list],dim=0)
        example_num = len(batch_list)
        rollouts = RolloutStorage(example_num,args.use_stop)
        

        initialize_state_input_ids,initialize_state_attention_mask,_ = train_dataloader.dataset.get_vanilla_prompt(idxs)
        
        state_input_ids_list = [initialize_state_input_ids[i].unsqueeze(0) for i in range(example_num)]
        state_attention_mask_list = [initialize_state_attention_mask[i].unsqueeze(0) for i in range(example_num)]
        
        old_actions_list = [[] for _ in range(example_num)]
        bad_list = [False for _ in range(example_num)]
        numstep = 0
        with torch.no_grad():
            _, initialize_rewards = env_model_module.compute_rewards(initialize_state_input_ids,initialize_state_attention_mask,rel_ids)
        initialize_reward_list = initialize_rewards.tolist()
        rollouts.insert_initialize_reward_list(initialize_reward_list)
        rollouts.initialize_example_actions(action_input_ids,action_attention_mask,drug1_sent_num,drug2_sent_num)
        rel_ids = rel_ids.tolist()
        rel_ids_copy = copy.deepcopy(rel_ids)
        
        while True:
            with torch.no_grad():
                value_list, action_list, action_log_prob_list,bad_list,new_bad_list1 = actor_critic_module.act(
                    action_input_ids,
                    action_attention_mask,
                    state_input_ids_list,
                    state_attention_mask_list,
                    drug1_sent_num,
                    drug2_sent_num,
                    old_actions_list,
                    bad_list)    
                
                state_input_ids_list,state_attention_mask_list,step_reward_list,bad_list,new_bad_list2,stop_list = env_model_module.step(
                    old_actions_list,
                    action_list,
                    idxs,
                    rel_ids,
                    bad_list,
                    train_dataloader.dataset) 

                new_bad_list = [item1 or item2 for item1,item2 in zip(new_bad_list1,new_bad_list2)]

                # print("action_list",action_list)
                # print("value_list",[None if item == None else round(item.item(),2) for item in value_list])
                # print("step_reward_list",None if step_reward_list == None else [None if item == None else round(item.item(),2) for item in step_reward_list])
                
                rollouts.insert(action_list,
                                value_list,
                                action_log_prob_list,
                                step_reward_list,
                                state_input_ids_list,
                                state_attention_mask_list,
                                bad_list,
                                new_bad_list,
                                stop_list)
                if step_reward_list!=None:
                    for item in step_reward_list:
                        if item!=None:
                            episode_rewards.append(item.item())

                if all(bad_list) == True:
                    break

                # print(numstep)
                numstep+=1

                old_actions_list = [old_actions+[action] for old_actions, action in zip(old_actions_list,action_list)]



        rollouts.compute_returns(args.use_gae, args.gamma, args.gae_lambda)

        value_loss, action_loss, entropy_loss,globalstep = agent.update(rollouts,globalstep)  

        if (globalstep+1) > (args.eval_step*(eval_time+1)):
            torch.cuda.empty_cache()
            actor_critic.eval()
            # dev_data = DATASET_MODEL_CLASS[args.dataset](args,'dev')
            # dev_sampler = SequentialSampler(dev_data) if args.local_rank == -1 else DistributedSampler(dev_data)
            # dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1,num_workers=8)
            # dev_f1 = evaluate(dev_dataloader,actor_critic,env_model,args,logger,random_act = True)

            dev_data = DATASET_MODEL_CLASS[args.dataset](args,'test')
            dev_sampler = SequentialSampler(dev_data) if args.local_rank == -1 else DistributedSampler(dev_data)
            dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1,num_workers=8)
            dev_f1 = evaluate(dev_dataloader,actor_critic,env_model,args,logger,random_act = False)
            # if args.local_rank in [-1,0]:
            #     logger.info('before updated env with rl performance')
            actor_critic.train()
            torch.cuda.empty_cache()

        # if True:
        #     # input_ids_list = [item[-1] if item != [] else  initialize_state_input_ids[i].unsqueeze(0) for i,item in enumerate(rollouts.obs_input_ids_list)]
        #     # attention_mask_list = [item[-1] if item != [] else  initialize_state_attention_mask[i].unsqueeze(0) for i,item in enumerate(rollouts.obs_attention_mask_list)]

        #     # env_input_ids_buffer.extend(input_ids_list)
        #     # env_attention_mask_buffer.extend(attention_mask_list)
        #     # rel_ids_buffer.extend(rel_ids_copy)
        #     idxs_buffer.extend(idxs)

        #     if (globalstep+1) > (args.eval_step*(eval_time+1)):

        #         env_input_ids_buffer,env_attention_mask_buffer, rel_ids_buffer = prepare_data(idxs_buffer,actor_critic,train_dataloader,args,random_act=True)
        #         # pickle.dump([env_input_ids_buffer,env_attention_mask_buffer, rel_ids_buffer],open('./data.pkl','wb'))

        #         exp_len = len(env_input_ids_buffer)

        #         env_batch_size = 16

        #         step_i_list = list(range(0,exp_len,env_batch_size))[0:-1]
        #         for step_i in tqdm(step_i_list,total = len(step_i_list)):
        #             input_ids_list = env_input_ids_buffer[step_i:step_i+env_batch_size]
        #             attention_mask_list = env_attention_mask_buffer[step_i:step_i+env_batch_size]
        #             rel_ids = rel_ids_buffer[step_i:step_i+env_batch_size]
        #         # while len(env_input_ids_buffer)>=env_batch_size:
        #             # input_ids_list = env_input_ids_buffer[0:env_batch_size]
        #             # attention_mask_list = env_attention_mask_buffer[0:env_batch_size]
        #             # rel_ids = rel_ids_buffer[0:env_batch_size]
        #             # env_input_ids_buffer = env_input_ids_buffer[env_batch_size:]
        #             # env_attention_mask_buffer = env_attention_mask_buffer[env_batch_size:]
        #             # rel_ids_buffer = rel_ids_buffer[env_batch_size:] 

        #             input_ids = torch.cat(input_ids_list,dim=0)
        #             attention_mask = torch.cat(attention_mask_list,dim=0)
        #             rel_ids = torch.tensor(rel_ids).to(input_ids.device)

        #             env_loss = env_model_module(input_ids,attention_mask,rel_ids)
        #             env_loss_item = env_loss.item()
        #             env_loss.backward()
        #             if (step_i/env_batch_size) % 10 == 0 and args.local_rank in [-1,0]:
        #                 logger.info('env train batch = {}:{} loss = {}'.format(int(step_i/env_batch_size),len(step_i_list),env_loss_item))
        #             torch.nn.utils.clip_grad_norm_(env_model_module.parameters(), 1.0*args.gradient_accumulation_steps)
        #             env_optimizer.step()
        #             env_optimizer.zero_grad() 

        # if (globalstep+1) > (args.eval_step*(eval_time+1)):
        #     eval_time+=1
        #     torch.cuda.empty_cache()
        #     actor_critic.eval()
        #     # dev_data = DATASET_MODEL_CLASS[args.dataset](args,'dev')
        #     # dev_sampler = SequentialSampler(dev_data) if args.local_rank == -1 else DistributedSampler(dev_data)
        #     # dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1,num_workers=8)
        #     # dev_f1 = evaluate(dev_dataloader,actor_critic,env_model,args,logger,random_act = True)

        #     dev_data = DATASET_MODEL_CLASS[args.dataset](args,'test')
        #     dev_sampler = SequentialSampler(dev_data) if args.local_rank == -1 else DistributedSampler(dev_data)
        #     dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1,num_workers=8)
        #     dev_f1 = evaluate(dev_dataloader,actor_critic,env_model,args,logger,random_act = True)
        #     if args.local_rank in [-1,0]:
        #         logger.info('after updated env random performance')

        #     dev_data = DATASET_MODEL_CLASS[args.dataset](args,'test')
        #     dev_sampler = SequentialSampler(dev_data) if args.local_rank == -1 else DistributedSampler(dev_data)
        #     dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1,num_workers=8)
        #     dev_f1 = evaluate(dev_dataloader,actor_critic,env_model,args,logger,random_act = False)
        #     if args.local_rank in [-1,0]:
        #         logger.info('rl with after updated env random performance')

        #     actor_critic.train()
        #     torch.cuda.empty_cache()

        if (globalstep+1) > (args.eval_step*(eval_time+1)) and args.local_rank in [-1,0]:
            checkpoints_dir = './checkpoints/{}/{}'.format(start_date,args.annotation)
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)
            checkpoint_path = os.path.join(checkpoints_dir,'checkpoint_epoch{}.pt'.format(globalstep))
            optimizer_to(ac_optimizer,torch.device('cpu'))
            # optimizer_to(env_optimizer,torch.device('cpu'))
            logger.info('Save checkpoint to {}'.format(checkpoint_path))
            torch.save({'actor_critic':actor_critic.state_dict(),
                        'ac_optimizer':ac_optimizer.state_dict(),
                        'env_model': env_model.state_dict()
                        }, 
                        checkpoint_path)
            optimizer_to(ac_optimizer,args.device)
            # 'env_optimizer': env_optimizer.state_dict()
            # optimizer_to(env_optimizer,args.device)
            eval_time+=1
            
        if args.local_rank in [-1,0]: # 
                writer.add_scalar('mean', np.mean(episode_rewards), globalstep)
                writer.add_scalar('median', np.median(episode_rewards), globalstep)
                writer.add_scalar('min', np.min(episode_rewards), globalstep)
                writer.add_scalar('max', np.max(episode_rewards), globalstep)
                writer.add_scalar('value_loss', value_loss, globalstep)
                writer.add_scalar('action_loss', action_loss, globalstep)
                writer.add_scalar('entropy_loss', entropy_loss, globalstep)
                # writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], globalstep)
        if (globalstep+1) % args.log_step == 0 and args.local_rank in [-1,0] and len(episode_rewards) >= 100:
            elapsed = format_time(time.time() - t0)
            logger.info('globalstep {:>5} Batch {:>5,} of {:>5,}. mean/median reward {}/{}, min/max reward {}/{} Value Loss {:} Action Loss {:} Entropy Loss {:} Elapsed:{:}.'
            .format(globalstep,
                    step+1, 
                    len(train_dataloader),
                    format(np.mean(episode_rewards), '.4f'),
                    format(np.median(episode_rewards), '.4f'),
                    format(np.min(episode_rewards), '.4f'),
                    format(np.max(episode_rewards), '.4f'),
                    format(value_loss, '.4f'),
                    format(action_loss, '.4f'),
                    format(entropy_loss,'.4f'),
                    elapsed))
        
    # avg_loss = np.array(avg_loss).mean()
    return globalstep,eval_time # avg_loss,globalstep