from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
import numpy as np
import copy
import torch
import json
import pickle
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tools.common import format_time
from tqdm import tqdm


class drugbank_dataset_rl(Dataset):
    def __init__(self,args,state):
        with open(args.ddi_dict_path, 'r') as file:
            self.ddi_dict = json.load(file)
        with open('./data/drugbank/node2id.json', 'r') as file:
            self.accession2id = json.load(file)
        if False:
            self.data = pickle.load(open('./data/ddi/test_ddi_zero_shot_100.pkl', 'rb'))
        elif args.use_zero_shot:
            with open('./data/drugbank/{}_ddi_zero_shot.pkl'.format(state), 'rb') as file:
                self.data = pickle.load(file)
                # random.shuffle(self.data)
                # self.data = self.data[0:100]
                # pickle.dump(self.data,open('./data/ddi/test_ddi_zero_shot_100.pkl', 'wb'))
        elif args.use_few_shot:
            if state == 'train':
                with open('./data/drugbank/{}_ddi_{}_shot.pkl'.format(state,args.shot_num), 'rb') as file:
                    self.data = pickle.load(file)
            else:
                with open('./data/drugbank/{}_ddi_{}_shot.pkl'.format(state,5), 'rb') as file:
                    self.data = pickle.load(file)
        else:
            with open('./data/drugbank/{}.txt'.format(state), 'r') as file:
                self.data = [[int(num) for num in item.strip().split()] for item in file.readlines()]
        self.id2accession = {}
        for key,value in self.accession2id.items():
            self.id2accession[value] = key
        with open('./data/drugbank/id2rel2.txt','r') as file:
            self.id2rel = [item.strip() for item in file.readlines()]
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
        if 'roberta' not in args.pretrained_model_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.state = state
        self.bg_max_length = int(200*(self.args.max_length/512))
        # self.softmaxed_sim_martix = torch.load('./data/softmaxed_sim_martix.pt')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        drug1_id,drug2_id,rel_id = self.data[index]
        drug1_accession,drug2_accession = self.id2accession[drug1_id],self.id2accession[drug2_id]
        drug1_name,drug1_name_postfix_input_ids,drug1_summary,drug1_sent_list,drug1_sent_tokenized_list,drug1_prefix_sent_tokenized_list = self.ddi_dict[drug1_accession].values()
        drug2_name,drug2_name_postfix_input_ids,drug2_summary,drug2_sent_list,drug2_sent_tokenized_list,drug2_prefix_sent_tokenized_list = self.ddi_dict[drug2_accession].values()
      
        if self.args.use_id:
            prompt = "In the above context, we can predict that the drug-drug interactions between {} and {} is that: ".format(drug1_accession,drug2_accession)
        else:
            prompt = "In the above context, we can predict that the drug-drug interactions between {} and {} is that: ".format(drug1_name,drug2_name)
        # prompt_tokenized = self.tokenizer(prompt)
        # prompt_input_ids = prompt_tokenized['input_ids']
        # prompt_attention_mask = prompt_tokenized['attention_mask']
        # prompt_length = len(prompt_attention_mask)

        if self.args.drug_name_only:
            prompt_tokenized = self.tokenizer(prompt,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                max_length=self.args.drug_only_max_length)
            example = [prompt_tokenized['input_ids'],prompt_tokenized['attention_mask'],rel_id]
            example = [torch.tensor(t,dtype=torch.long) for t in example]
            return example
        
        if self.args.not_random:
            drug1_summary_ = self.tokenizer(drug1_summary,
                                            add_special_tokens=False,
                                            return_token_type_ids=False,
                                            truncation=True,
                                            max_length=self.bg_max_length)['input_ids']
            drug1_summary_ = self.tokenizer.decode(drug1_summary_)
            if drug1_summary_[-1]!='.':
                drug1_summary_ = drug1_summary_[:-1] + '.'

            drug2_summary_ = self.tokenizer(drug2_summary,
                                            add_special_tokens=False,
                                            return_token_type_ids=False,
                                            truncation=True,
                                            max_length=self.bg_max_length)['input_ids']
            drug2_summary_ = self.tokenizer.decode(drug2_summary_)
            if drug2_summary_[-1]!='.':
                drug2_summary_ = drug2_summary_[:-1] + '.'

            prompt = drug1_name + ": " + drug1_summary_ + " " +drug2_name+": "+drug2_summary_ + " " + prompt 
            prompt_tokenized = self.tokenizer(prompt,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                max_length=self.args.max_length)
            example = [prompt_tokenized['input_ids'],prompt_tokenized['attention_mask'],rel_id]
            example = [torch.tensor(t,dtype=torch.long) for t in example]
            return example
            

        drug1_sent_num = len(drug1_sent_list)
        drug2_sent_num = len(drug2_sent_list)
        sent_tokenized_list = drug1_sent_tokenized_list+drug2_sent_tokenized_list
        sent_num = drug1_sent_num + drug2_sent_num
        selected_idxs = []
        latest_context_tokenized = self.tokenizer(prompt,
                            add_special_tokens=True,
                            return_token_type_ids=False,
                            padding="max_length",
                            truncation=True,
                            max_length=self.args.max_length)
        while True:
            if len(selected_idxs) == sent_num:
                break
            idx = random.randint(0,sent_num-1)
            if idx in selected_idxs:
                continue
            selected_idxs.append(idx)

            sorted(selected_idxs)
            
            drug1_sent_select_idx = []
            drug2_sent_select_idx = []
            for i in list(selected_idxs):
                if i<drug1_sent_num:
                    drug1_sent_select_idx.append(i)
                elif i < sent_num:
                    drug2_sent_select_idx.append(i)
                else:
                    assert False

                drug1_sent_input_ids = []
                for item in drug1_sent_select_idx:
                    drug1_sent_input_ids.extend(sent_tokenized_list[item])
                drug2_sent_input_ids = []
                for item in drug2_sent_select_idx:
                    drug2_sent_input_ids.extend(sent_tokenized_list[item])

                prompt_input_ids = self.tokenizer(prompt,add_special_tokens=False)['input_ids']
                if len(drug1_sent_select_idx)==0:
                    current_drug1_name_postfix_input_ids = []
                else:
                    current_drug1_name_postfix_input_ids = drug1_name_postfix_input_ids
                if len(drug2_sent_select_idx)==0:
                    current_drug2_name_postfix_input_ids = []
                else:
                    current_drug2_name_postfix_input_ids = drug2_name_postfix_input_ids
                input_ids = current_drug1_name_postfix_input_ids+drug1_sent_input_ids+current_drug2_name_postfix_input_ids+drug2_sent_input_ids+prompt_input_ids
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
                if bad:
                    break
                else:
                    latest_context_tokenized = context_tokenized
            
        example = [latest_context_tokenized['input_ids'],latest_context_tokenized['attention_mask'],rel_id]
        example = [torch.tensor(t,dtype=torch.long) for t in example]
        # elapsed = format_time(time.time() - t0)
        return example
