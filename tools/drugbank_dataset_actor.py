from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
import numpy as np
import copy
import torch
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import pickle


class drugbank_dataset_actor(Dataset):
    def __init__(self,args,state):
        with open(args.ddi_dict_path, 'r') as file:
            self.ddi_dict = json.load(file)
        with open('./data/node2id.json', 'r') as file:
            self.accession2id = json.load(file)
        if args.use_zero_shot:
            with open('./data/{}_ddi.txt'.format(state), 'r') as file:
                self.data = [[int(num) for num in item.strip().split()] for item in file.readlines()]
        else:
            with open('./data/{}.txt'.format(state), 'r') as file:
                self.data = [[int(num) for num in item.strip().split()] for item in file.readlines()]
        self.id2accession = {}
        for key,value in self.accession2id.items():
            self.id2accession[value] = key
        with open('./data/id2rel2.txt','r') as file:
            self.id2rel = [item.strip() for item in file.readlines()]
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
        if 'roberta' not in args.pretrained_model_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.state = state
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # index = 46416
        drug1_id,drug2_id,rel_id = self.data[index]
        drug1_accession,drug2_accession = self.id2accession[drug1_id],self.id2accession[drug2_id]
        drug1_name,drug1_name_length,drug1_summary,drug1_sent_list,drug1_sent_tokenized_list,drug1_prefix_sent_tokenized_list = self.ddi_dict[drug1_accession].values()
        drug2_name,drug2_name_length,drug2_summary,drug2_sent_list,drug2_sent_tokenized_list,drug2_prefix_sent_tokenized_list = self.ddi_dict[drug2_accession].values()
        
        drug1_sent_num = len(drug1_prefix_sent_tokenized_list)
        drug2_sent_num = len(drug2_prefix_sent_tokenized_list)


        input_ids = [item[0] for item in drug1_prefix_sent_tokenized_list]+[item[0] for item in drug2_prefix_sent_tokenized_list]
        attention_mask = [item[1] for item in drug1_prefix_sent_tokenized_list]+[item[1] for item in drug2_prefix_sent_tokenized_list]
        example = [input_ids,attention_mask,drug1_sent_num,drug2_sent_num,index,rel_id]
        example = [torch.tensor(t,dtype=torch.long) for t in example]
        return example 
    