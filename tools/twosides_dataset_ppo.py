import random
import numpy as np
import copy
import torch
import json
import pickle
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm


class twosides_dataset_ppo(Dataset):
    def __init__(self,args,state,pos):
        with open(args.info_dict_path, 'r') as file:
            self.info_dict = json.load(file)
        with open('./data/twosides/id2drug.json', 'r') as file:
            id2drug = json.load(file)
        self.cid2id = {}
        for key,value in id2drug.items():
            cid = str(int(value['cid'][3:]))
            self.cid2id[cid] = key
        self.id2cid = {}
        for key,value in self.cid2id.items():
            self.id2cid[value] = key
        if args.use_zero_shot:
            # with open('./data/twosides/{}_twosides_zero_shot.pkl'.format(state), 'rb') as file:
            #     self.data = pickle.load(file)
            with open('./data/twosides/{}_ddi.txt'.format(state), 'r') as file:
                self.data = []
                data = file.readlines()
                data = [data[i:i+2] for i in range(0,len(data),2)]
                for item in data:
                    drug1_id,drug2_id,rel_ids,_ = item[0].strip().split('\t')
                    drug3_id,drug4_id,_,_ = item[1].strip().split('\t')
                    rel_ids = [int(i) for i in rel_ids.split(',')]
                    self.data.append([drug1_id,drug2_id,drug3_id,drug4_id,rel_ids])
        else:
            with open('./data/twosides/{}.txt'.format(state), 'r') as file:
                self.data = []
                data = file.readlines()
                data = [data[i:i+2] for i in range(0,len(data),2)]
                for item in data:
                    drug1_id,drug2_id,rel_ids,_ = item[0].strip().split('\t')
                    drug3_id,drug4_id,_,_ = item[1].strip().split('\t')
                    rel_ids = [int(i) for i in rel_ids.split(',')]
                    self.data.append([drug1_id,drug2_id,drug3_id,drug4_id,rel_ids])
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
        self.state = state
        # gpu_num = self.args.gpu_num # torch.cuda.device_count()
        # pad_num = gpu_num - len(self.data)%gpu_num
        # if self.state != 'train' and  args.multi_gpu and pad_num!=0:
        #     for i in range(pad_num):
        #         self.data.append(None)
        # self.pad_num = pad_num
        # self.true_data_len = len(self.data)-self.pad_num
        self.pos = pos


    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        if self.data[index] == None:
            return 0

        drug1_id,drug2_id,drug3_id,drug4_id,rel_ids = self.data[index]
        if not self.pos:
            drug1_id = drug3_id
            drug2_id = drug4_id
        drug1_cid,drug2_cid = self.id2cid[drug1_id],self.id2cid[drug2_id]
        drug1_name,drug1_name_postfix_input_ids,drug1_summary,drug1_sent_list,drug1_sent_tokenized_list,drug1_prefix_sent_tokenized_list = self.info_dict[drug1_cid].values()
        drug2_name,drug2_name_postfix_input_ids,drug2_summary,drug2_sent_list,drug2_sent_tokenized_list,drug2_prefix_sent_tokenized_list = self.info_dict[drug2_cid].values()

        drug1_sent_num = len(drug1_prefix_sent_tokenized_list)
        drug2_sent_num = len(drug2_prefix_sent_tokenized_list)
        # sent_num = drug1_sent_num + drug2_sent_num

        pos_input_ids = [item[0] for item in drug1_prefix_sent_tokenized_list]+[item[0] for item in drug2_prefix_sent_tokenized_list]
        pos_attention_mask = [item[1] for item in drug1_prefix_sent_tokenized_list]+[item[1] for item in drug2_prefix_sent_tokenized_list]

        # if self.args.use_stop:
        #     input_ids += [self.stop_sent_input_id]
        #     attention_mask += [self.stop_sent_attention_mask]

        example = [pos_input_ids,pos_attention_mask,drug1_sent_num,drug2_sent_num,index,rel_ids]
        example = [torch.tensor(t,dtype=torch.long) for t in example]
        return example # ,drug1_sent_num+drug2_sent_num
    
    def get_vanilla_prompt(self,indexs):
        input_ids_list = []
        attention_mask_list = []
        rel_id_list = []

        for index in indexs:
            drug1_id,drug2_id,drug3_id,drug4_id,rel_ids = self.data[index]
            if not self.pos:
                drug1_id = drug3_id
                drug2_id = drug4_id
            drug1_cid,drug2_cid = self.id2cid[drug1_id],self.id2cid[drug2_id]
            drug1_name,drug1_name_postfix_input_ids,drug1_summary,drug1_sent_list,drug1_sent_tokenized_list,drug1_prefix_sent_tokenized_list = self.info_dict[drug1_cid].values()
            drug2_name,drug2_name_postfix_input_ids,drug2_summary,drug2_sent_list,drug2_sent_tokenized_list,drug2_prefix_sent_tokenized_list = self.info_dict[drug2_cid].values()
        
            prompt = "In the above context, we can predict that the drug-drug interactions between {} and {} is that: ".format(drug1_name,drug2_name)
            prompt_tokenized = self.tokenizer(prompt,
                        add_special_tokens=True,
                        return_token_type_ids=False,
                        padding="max_length",
                        truncation=True,
                        max_length=self.args.max_length)
            input_ids_list.append(prompt_tokenized['input_ids'])
            attention_mask_list.append(prompt_tokenized['attention_mask'])
            rel_id_list.append(rel_ids)

        example = [input_ids_list,attention_mask_list,rel_id_list]
        example = [torch.tensor(t,dtype=torch.long).to(self.args.device) for t in example]
        return example

    
    def get_state_by_actions(self,actions,index):
        drug1_id,drug2_id,drug3_id,drug4_id,rel_id = self.data[index]
        if not self.pos:
            drug1_id = drug3_id
            drug2_id = drug4_id
        drug1_cid,drug2_cid = self.id2cid[drug1_id],self.id2cid[drug2_id]
        drug1_name,drug1_name_postfix_input_ids,drug1_summary,drug1_sent_list,drug1_sent_tokenized_list,drug1_prefix_sent_tokenized_list = self.info_dict[drug1_cid].values()
        drug2_name,drug2_name_postfix_input_ids,drug2_summary,drug2_sent_list,drug2_sent_tokenized_list,drug2_prefix_sent_tokenized_list = self.info_dict[drug2_cid].values()
        
        prompt = "In the above context, we can predict that the drug-drug interactions between {} and {} is that: ".format(drug1_name,drug2_name)
        
        drug1_sent_num = len(drug1_sent_list)
        drug2_sent_num = len(drug2_sent_list)
        sent_num = drug1_sent_num+drug2_sent_num

        sorted(actions)
        select_sent_idxs = actions
        
        drug1_sent_select_idx = []
        drug2_sent_select_idx = []
        stop = False
        for i,idx in enumerate(list(select_sent_idxs)):
            if idx<drug1_sent_num:
                drug1_sent_select_idx.append(idx)
            elif idx < sent_num:
                drug2_sent_select_idx.append(idx)
            else:
                stop = True
                assert i==len(select_sent_idxs)-1

        sent_tokenized_list = drug1_sent_tokenized_list+drug2_sent_tokenized_list
        drug1_sent_input_ids = []
        for item in drug1_sent_select_idx:
            drug1_sent_input_ids.extend(sent_tokenized_list[item])
        drug2_sent_input_ids = []
        for item in drug2_sent_select_idx:
            drug2_sent_input_ids.extend(sent_tokenized_list[item])
        prompt_input_ids = self.tokenizer(prompt,add_special_tokens=False)['input_ids']

        if len(drug1_sent_select_idx)==0:
            drug1_name_postfix_input_ids = []
        if len(drug2_sent_select_idx)==0:
            drug2_name_postfix_input_ids = []
        input_ids = drug1_name_postfix_input_ids+drug1_sent_input_ids+drug2_name_postfix_input_ids+drug2_sent_input_ids+prompt_input_ids
        context = self.tokenizer.decode(input_ids)

        # true_context_length = len(self.tokenizer(context,add_special_tokens=True)['input_ids'])

        context_tokenized = self.tokenizer(context,
                    add_special_tokens=True,
                    return_token_type_ids=False,
                    padding="max_length",
                    truncation=True,
                    max_length=self.args.max_length)
        
        bad = False
        if sum(context_tokenized['attention_mask']) == self.args.max_length:
            bad = True

        if select_sent_idxs == []:
            bad = False
            
        example = [context_tokenized['input_ids'],context_tokenized['attention_mask'],bad,stop]
        example = [torch.tensor(t,dtype=torch.long).to(self.args.device) for t in example]
        return example

    def get_too_short_instance_index(self,index):
        drug1_id,drug2_id,_,_,rel_id = self.data[index]
        drug1_cid,drug2_cid = self.id2cid[drug1_id],self.id2cid[drug2_id]
        drug1_name,drug1_name_postfix_input_ids,drug1_summary,drug1_sent_list,drug1_sent_tokenized_list,drug1_prefix_sent_tokenized_list = self.info_dict[drug1_cid].values()
        drug2_name,drug2_name_postfix_input_ids,drug2_summary,drug2_sent_list,drug2_sent_tokenized_list,drug2_prefix_sent_tokenized_list = self.info_dict[drug2_cid].values()
        
        prompt = "In the above context, we can predict that the drug-drug interactions between {} and {} is that: ".format(drug1_name,drug2_name)
        
        drug1_sent_num = len(drug1_sent_list)
        drug2_sent_num = len(drug2_sent_list)
 
        # sent_tokenized_list = drug1_sent_tokenized_list+drug2_sent_tokenized_list
        drug1_sent_input_ids = []
        for item in drug1_sent_tokenized_list:
            drug1_sent_input_ids.extend(item)
        drug2_sent_input_ids = []
        for item in drug2_sent_tokenized_list:
            drug2_sent_input_ids.extend(item)
        prompt_input_ids = self.tokenizer(prompt,add_special_tokens=False)['input_ids']

        input_ids = drug1_name_postfix_input_ids+drug1_sent_input_ids+drug2_name_postfix_input_ids+drug2_sent_input_ids+prompt_input_ids

        if len(input_ids)>self.args.max_length*1.5:
            return True
        context = self.tokenizer.decode(input_ids)

        # true_context_length = len(self.tokenizer(context,add_special_tokens=True)['input_ids'])

        context_tokenized = self.tokenizer(context,
                    add_special_tokens=True,
                    return_token_type_ids=False,
                    padding="max_length",
                    truncation=True,
                    max_length=self.args.max_length+10)
        
        overflow = False
        if sum(context_tokenized['attention_mask']) >= self.args.max_length+10:
            overflow = True
        return overflow


    