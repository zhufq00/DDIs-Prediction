import gc
import sys
import os
current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(current_path)
import pickle
import torch
import random
import time
import logging
import math
import json
import datetime
import numpy as np

import argparse
import torch.distributed as dist
from transformers import AutoTokenizer
from datetime import date
from transformers import RobertaTokenizer,get_linear_schedule_with_warmup,RobertaForMultipleChoice,AdamW,get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tools.drugbank_dataset_rl import drugbank_dataset_rl
from tools.twosides_dataset_rl import twosides_dataset_rl
from torch.optim import Adam
from tools.common import seed_everything,Args,format_time
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import get_rank
from torch.utils.tensorboard import SummaryWriter  
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from sklearn import metrics
from models.roberta_cls import roberta_cls
from tools.drugbank_dataset_ppo import drugbank_dataset_ppo
# from evaluate import evaluate
from sklearn.metrics import roc_auc_score, average_precision_score
# from apex import amp

MODEL_CLASSES = {
    'roberta_cls': roberta_cls,
}

DATASET_MODEL_CLASS = {
    'drugbank': drugbank_dataset_rl,
    'twosides': twosides_dataset_rl
}

def train(args,train_dataloader,model,optimizer,writer,logger=None,global_step=0):
    
    avg_loss, avg_acc = [],[]

    model.zero_grad()
    t0 = time.time()
    for step, batch in enumerate(train_dataloader):
        model.train()
        batch = tuple(t.to(args.device) for t in batch)
        output = model(*(tuple(batch)[0:3]))
        loss = output[0]
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        loss = loss.item()
        avg_loss.append(loss)
        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0*args.gradient_accumulation_steps)
            optimizer.step()
            model.zero_grad()
            global_step += 1
            # break
        
        if global_step % 10==0 and args.local_rank in [-1,0]:
                writer.add_scalar('loss', loss, global_step)
                # writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_step)
        if (step+1) % args.log_step == 0 and args.local_rank in [-1,0]:
            elapsed = format_time(time.time() - t0)
            logger.info('Batch {:>5,} of {:>5,}.Loss: {:} Elapsed:{:}.'
            .format(step+1, len(train_dataloader),format(loss, '.4f'),elapsed))
        
        
    avg_loss = np.array(avg_loss).mean()
    return avg_loss,global_step


def evaluate(test_dataloader,model,args,logger):
    avg_acc = []
    model.eval()   

    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader,total=len(test_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            batch = [t.long() for t in batch]
            pos_input_ids,pos_attention_mask,labels = batch

            scores = model(pos_input_ids,pos_attention_mask,labels)[1]
            
            correct = torch.eq(torch.max(scores, dim=1)[1], labels.flatten()).float()         
            acc = correct.sum().item() / len(correct)

            y_pred.extend(torch.max(scores, dim=1)[1].tolist())
            y_true.extend(labels.flatten().tolist())
            avg_acc.append(acc)
            # break
        
    # gather_objects = [None for _ in range(torch.distributed.get_world_size())]
    # gather_objects[args.local_rank] = [y_pred,y_true]
    # # Note: Process group initialization omitted on each rank.

    # acc = metrics.accuracy_score(y_true,y_pred)
    # f1 = metrics.f1_score(y_true, y_pred, average=None).mean() 
    # kappa = metrics.cohen_kappa_score(y_true, y_pred)
    # print('len = {} acc = {} f1 = {}, kappa = {} local_rank = {}'.format(len(y_true),acc,f1,kappa,torch.distributed.get_rank()))
    # output = [None for _ in gather_objects]
    # dist.gather_object(
    #     gather_objects[dist.get_rank()],
    #     output if dist.get_rank() == 0 else None,
    #     dst=0
    # )
    # dist.barrier()
    if args.local_rank in [-1,0]:
        # Assumes world_size of 3.
        # y_true = []
        # y_pred = []
        # for data in output:
        #     y_pred.extend(data[0])
        #     print(len(data[0]))
        #     y_true.extend(data[1])
        acc = metrics.accuracy_score(y_true,y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average=None).mean() 
        kappa = metrics.cohen_kappa_score(y_true, y_pred)
        logger.info("acc = {:>5,}".format(acc,'.4f'))
        logger.info("f1 = {}".format(f1))
        logger.info("kappa = {}".format(kappa))
        logger.info("len = {}".format(len(y_true)))
        logger.info("len dataset= {}".format(len(test_dataloader.dataset)))
    return f1

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file',help='config file path')
    config_file = 'configs/main_drugbank.yaml'# parser.parse_args().config_file
    args = Args(config_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    start_date = date.today().strftime('%m-%d')
    if args.eval:
        log_path = './log/{}/{}-eval.log'.format(start_date,args.annotation)
    else:
        log_path = './log/{}/{}.log'.format(start_date,args.annotation)
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    torch.cuda.empty_cache()
    seed_everything(args.seed)
    if args.multi_gpu and args.use_gpu:
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else :
        args.local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_gpu == False:
        device = torch.device('cpu')
    args.device = device
    logger = None
    if args.local_rank in [-1,0]:
        logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s | %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=log_path,
            filemode=args.filemode)
        logger = logging.getLogger()
        logger.info("Process rank: {}, device: {}, distributed training: {}".format(
                    args.local_rank,device, bool(args.local_rank != -1)))
        logger.info("Training/evaluation parameters %s", args.to_str())
    
    if args.local_rank in [-1,0]:
        tensorboard_path = './tensorboard/{}/{}'.format(start_date,args.annotation)
        if not os.path.exists(os.path.dirname(tensorboard_path)):
            os.makedirs(os.path.dirname(tensorboard_path))
        writer = SummaryWriter(tensorboard_path)
    else:
        writer = None

    if args.eval:
        model = MODEL_CLASSES[args.model_type](args)
        # checkpoint = torch.load(args.checkpoint,map_location='cpu')

        # state_dict = checkpoint['model']
        # new_state_dict = {}
        # for key, value in state_dict.items():
        #     if key.startswith('module.'):
        #         new_state_dict[key[7:]] = value
        #     else:
        #         new_state_dict[key] = value

        # model.load_state_dict(new_state_dict,strict=True)
        model.to(args.device)
        dev_data = DATASET_MODEL_CLASS[args.dataset](args,'test')
        dev_sampler = RandomSampler(dev_data) if args.local_rank == -1 else DistributedSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=4)
        test_acc = evaluate(dev_dataloader,model,args,logger)
        # logger.info("dev_acc{}".format(test_acc))
        return
    start_epoch = 0

    model = MODEL_CLASSES[args.model_type](args)
    if args.resume:
        checkpoint = torch.load(args.checkpoint,map_location='cpu')
        state_dict = checkpoint['model']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict,strict=True)
        model.to(args.device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": args.weight_decay,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay": 0.0,},
    ]

    start_epoch = 0
    optimizer = Adam(optimizer_grouped_parameters,eps = args.epsilon,betas=(0.9,0.98),lr=args.lr)
    
    # if args.resume and args.local_rank in [-1,0]:
    #     state_dict = checkpoint['optimizer']
    #     new_state_dict = {}
    #     for key, value in state_dict.items():
    #         if key.startswith('module.'):
    #             new_state_dict[key[7:]] = value
    #         else:
    #             new_state_dict[key] = value

        # optimizer.load_state_dict(new_state_dict)
    model.to(args.device)

    global_step = 0
    best_performance = 0
    best_checkpoint_path = None
    # train_raw_data = pickle.load(open(os.path.join(args.data_dir,'train'),'rb'))
    patience = args.patience
    fail_time = 0
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
                                                         #  find_unused_parameters=True)

    for epoch in range(int(args.num_train_epochs)):
        if fail_time>=patience:
            break
        if epoch < start_epoch:
            continue
        if args.local_rank in [-1,0]:
            logger.info('local_rank={},epoch={}'.format(args.local_rank, epoch))
        train_dataset = DATASET_MODEL_CLASS[args.dataset](args,'train')
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size,num_workers=4)
        torch.cuda.empty_cache()
        train_loss,global_step = train(args, train_dataloader,model,optimizer,writer,logger,global_step)
        if args.local_rank in [-1,0]:
            logger.info('epoch={},loss={}'.format(epoch, train_loss))
        torch.cuda.empty_cache()
        gc.collect()
        
        if args.local_rank in [-1,0]:
            dev_data = DATASET_MODEL_CLASS[args.dataset](args,'dev')
            dev_sampler = RandomSampler(dev_data) # if args.local_rank == -1 else DistributedSampler(dev_data)
            dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=4)
            dev_acc = evaluate(dev_dataloader,model,args,logger)
            writer.add_scalar('dev_acc', dev_acc, epoch)
            logger.info("epoch={},dev_acc={}".format(epoch,dev_acc))
            if dev_acc > best_performance: #epoch % 10 == 0: # :
                checkpoints_dir = './checkpoints/{}/{}'.format(start_date,args.annotation)
                if not os.path.exists(checkpoints_dir):
                    os.makedirs(checkpoints_dir)
                checkpoint_path = os.path.join(checkpoints_dir,'checkpoint_epoch{}.pt'.format(epoch))
                best_checkpoint_path = checkpoint_path
                best_performance = dev_acc
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch},checkpoint_path)
                logger.info('Save best checkpoint to {}'.format(checkpoint_path))
                fail_time = 0
            else:
                fail_time+=1
    if args.test:
        if args.local_rank in [-1,0]:
            logger.info("test best_checkpoint_path={}".format(best_checkpoint_path))
            checkpoint = torch.load(best_checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            dev_data = DATASET_MODEL_CLASS[args.dataset](args,'test')
            dev_sampler = RandomSampler(dev_data) # if args.local_rank == -1 else DistributedSampler(dev_data)
            dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=4)
            test_acc = evaluate(dev_dataloader,model,args,logger)
            logger.info("best epoch={},test_f1={}".format(checkpoint['epoch'], test_acc))

if __name__=='__main__':
    main()

