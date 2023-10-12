from torch import nn
from transformers.models.bart.modeling_bart import BartForConditionalGeneration,BartModel
from torch.nn import CrossEntropyLoss
from transformers import RobertaForMultipleChoice, RobertaForSequenceClassification,RobertaModel
import torch.nn.functional as F

import torch
import pickle

class twosides_cls(nn.Module):

    def __init__(self, args):
        super(twosides_cls, self).__init__()
        self.args = args
        self.llm = RobertaForSequenceClassification.from_pretrained(args.pretrained_model_path,num_labels = 200)
        # if args.vocab_size is not None:
        #    self.llm.resize_token_embeddings(args.vocab_size)
        self.config = self.llm.config
        self.bce_loss = nn.BCELoss()
        # self.llm.config.pad_token_id = self.llm.config.eos_token_id
        # self.dropout = nn.Dropout(self.config.classifier_dropout)
        # self.dropout = nn.Dropout(0.0)
        # self.dense = nn.Linear(768, 200)
        # self.out_proj = nn.Linear(self.config.n_embd, 1)

        # self.relation_name_input_ids,self.relation_name_attention_mask = pickle.load(open('./data/twosides/relation_name_tokenized.pkl','rb'))
        # self.relation_name_pooler_output = pickle.load(open('./data/twosides/relation_name_pooler_output.pkl','rb'))
    
    def forward(
        self,pos_input_ids,pos_attention_mask,labels,neg_input_ids,neg_attention_mask
    ):
        batch_size,max_length = pos_input_ids.size()

        input_ids = torch.cat((pos_input_ids,neg_input_ids),dim=0)
        attention_mask = torch.cat((pos_attention_mask,neg_attention_mask),dim=0)
        
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = torch.sigmoid(outputs.logits)
        pos_logits,neg_logits = logits[0:batch_size],logits[batch_size:]

        losses = []
        for pos,neg,label in zip(pos_logits,neg_logits,labels):
            pos_scores = pos[label>0]
            neg_scores = neg[label>0]
            scores = torch.cat([pos_scores, neg_scores], dim=0)
            label = torch.cat([torch.ones(len(pos_scores)), torch.zeros(len(neg_scores))], dim=0).to(pos_scores.device)
            loss = self.bce_loss(scores, label) 
            losses.append(loss)
        
        loss = torch.mean(torch.stack(losses,dim=0))

        # pos_scores = pos_logits[labels>0]
        # neg_scores = neg_logits[labels>0]
        # scores = torch.cat([pos_scores, neg_scores], dim=0)
        # labels = torch.cat([torch.ones(len(pos_scores)), torch.zeros(len(neg_scores))], dim=0).to(pos_scores.device)
        # loss = self.bce_loss(scores, labels) 
        return loss,pos_logits,neg_logits