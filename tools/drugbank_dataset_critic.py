from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
import numpy as np
import copy
import torch
import json
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm


class drugbank_dataset_critic(Dataset):
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
        # self.softmaxed_sim_martix = torch.load('./data/softmaxed_sim_martix.pt')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index,actions):
        drug1_id,drug2_id,rel_id = self.data[index]
        drug1_accession,drug2_accession = self.id2accession[drug1_id],self.id2accession[drug2_id]
        drug1_name,drug1_name_length,drug1_summary,drug1_sent_list,drug1_sent_tokenized_list,drug1_prefix_sent_tokenized_list = self.ddi_dict[drug1_accession].values()
        drug2_name,drug2_name_length,drug2_summary,drug2_sent_list,drug2_sent_tokenized_list,drug2_prefix_sent_tokenized_list = self.ddi_dict[drug2_accession].values()
        
        prompt = "In the above context, we can predict that the drug-drug interactions between {} and {} is that: ".format(drug1_name,drug2_name)
        
        sent_list = drug1_sent_list+drug2_sent_list
        drug1_sent_num = len(drug1_sent_list)
        drug2_sent_num = len(drug2_sent_list)

        context_list = [prompt]
        for i in range(self.args.num_steps):
            idxs = actions[0:i+1]
            idxs = sorted(idxs)
            select_sent_idxs = idxs
            
            drug1_sent_select_idx = []
            drug2_sent_select_idx = []
            for i in list(select_sent_idxs):
                if i<drug1_sent_num:
                    drug1_sent_select_idx.append(i)
                else:
                    drug2_sent_select_idx.append(i)
            sent_tokenized_list = drug1_sent_tokenized_list+drug2_sent_tokenized_list
            drug1_sent_input_ids = []
            for item in drug1_sent_select_idx:
                drug1_sent_input_ids.extend(sent_tokenized_list[item][0])
            drug2_sent_input_ids = []
            for item in drug2_sent_select_idx:
                drug2_sent_input_ids.extend(sent_tokenized_list[item][0])
            drug1_name_postfix_input_ids = self.tokenizer(drug1_name+": ",add_special_tokens=False)['input_ids']
            drug2_name_postfix_input_ids = self.tokenizer(drug2_name+": ",add_special_tokens=False)['input_ids']
            prompt_input_ids = self.tokenizer(prompt,add_special_tokens=False)['input_ids']
            input_ids = drug1_name_postfix_input_ids+drug1_sent_input_ids+drug2_name_postfix_input_ids+drug2_sent_input_ids+prompt_input_ids
            context = self.tokenizer.decode(input_ids)

            # context = ''
            
            # first_drug1_sent = True
            # first_drug2_sent = True
            # for idx in idxs:
            #     if idx<drug1_sent_num:
            #         if first_drug1_sent:
            #             context = context+ drug1_name+': '+sent_list[idx]
            #             first_drug1_sent=False
            #         else:
            #             context = context+' '+sent_list[idx]
            #     else:
            #         if first_drug2_sent:
            #             context = context+ drug2_name+': '+sent_list[idx]
            #             first_drug2_sent=False
            #         else:
            #             context = context+' '+sent_list[idx]
            # context = context+prompt
            context_list.append(context)
        input_ids_list = []
        attention_mask_list = []
        rel_id_list = []
        for context in context_list:
            context_tokenized = self.tokenizer(context,
                        add_special_tokens=True,
                        return_token_type_ids=False,
                        padding="max_length",
                        truncation=True,
                        max_length=self.args.max_length)
            input_ids_list.append(context_tokenized['input_ids'])
            attention_mask_list.append(context_tokenized['attention_mask'])
            rel_id_list.append(rel_id)
        example = [input_ids_list,attention_mask_list,rel_id_list]
        example = [torch.tensor(t,dtype=torch.long) for t in example]
        return example

    