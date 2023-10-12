import gc
import torch
import logging
import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(current_path)
from datetime import date
from transformers import get_scheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from tools.drugbank_dataset_ppo import drugbank_dataset_ppo
from tools.twosides_dataset_ppo import twosides_dataset_ppo
from torch.optim import Adam
from tools.common import seed_everything,Args,format_time
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import get_rank
from models.roberta_env_twosides import roberta_env_twosides
from models.policy import Policy
from twosides.train_twosides import train_twosides

DATASET_MODEL_CLASS = {
    'drugbank': drugbank_dataset_ppo,
    'twosides': twosides_dataset_ppo
}

from torch.utils.tensorboard import SummaryWriter

def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file',help='config file path')
    config_file = './configs/ppo_twosides.yaml'# parser.parse_args().config_file

    ######################## Setting Start ########################
    args = Args(config_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    start_date = date.today().strftime('%m-%d')
    if args.eval:
        log_path = './log/{}/{}-eval.log'.format(start_date,args.annotation)
    else:
        log_path = './log/{}/{}.log'.format(start_date,args.annotation)
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
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

    ######################## Initialize ########################
    actor_critic = Policy(args)
    env_model = roberta_env_twosides(args)

    initialize_checkpoint = torch.load(args.initialize_checkpoint,map_location='cpu')
    if args.policy_resume:
        policy_checkpoint = torch.load(args.policy_checkpoint,map_location='cpu')
        actor_critic.load_state_dict({k.replace('module.',''):v for k,v in policy_checkpoint['actor_critic'].items()},strict=True)
    else:
        actor_critic.load_state_dict(initialize_checkpoint['model'],strict=False)
    env_model.load_state_dict(initialize_checkpoint['model'],strict=True)

    actor_critic.to(args.device)
    env_model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in actor_critic.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": args.weight_decay,},
        {"params": [p for n, p in actor_critic.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay": 0.0,}
        ]
    optimizer = Adam(optimizer_grouped_parameters,eps = args.epsilon,betas=(0.9,0.98),lr=args.lr)


    start_epoch = 0
    global_step = 0
    best_performance = 0
    best_checkpoint_path = None
    patience = args.patience
    fail_time = 0
    eval_time = 0
    for epoch in range(int(args.num_train_epochs)):
        if fail_time>=patience:
            break
        if epoch < start_epoch:
            continue
        if args.local_rank in [-1,0]:
            logger.info('local_rank={},epoch={}'.format(args.local_rank, epoch))
        train_dataset = DATASET_MODEL_CLASS[args.dataset](args,'train',pos=True)
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset,drop_last=False)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size,num_workers=8)
        torch.cuda.empty_cache()
        global_step,eval_time = train_twosides(args,train_dataloader,actor_critic,env_model,optimizer,writer,logger,global_step,eval_time,start_date)
        torch.cuda.empty_cache()
        gc.collect()

if __name__=='__main__':
    main()

