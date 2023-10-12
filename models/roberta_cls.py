from torch import nn
from transformers.models.bart.modeling_bart import BartForConditionalGeneration,BartModel
from torch.nn import CrossEntropyLoss
from transformers import RobertaForMultipleChoice, RobertaForSequenceClassification,RobertaModel
import torch.nn.functional as F
import torch
import pickle

class roberta_cls(nn.Module):
    def __init__(self, args):
        super(roberta_cls, self).__init__()
        self.args = args
        self.llm = RobertaForSequenceClassification.from_pretrained(
            args.pretrained_model_path,
            num_labels=86)
        # if args.vocab_size is not None:
        #    self.llm.resize_token_embeddings(args.vocab_size)
        self.config = self.llm.config
        # self.relation_name_input_ids,self.relation_name_attention_mask = pickle.load(open('./data/ddi/relation_name_tokenized.pkl','rb'))
        # self.llm.config.pad_token_id = self.llm.config.eos_token_id
        # self.dropout = nn.Dropout(self.config.classifier_dropout)
        # self.dropout = nn.Dropout(0.0)
        # self.dense = nn.Linear(self.config.d_model, self.config.d_model)
        # self.out_proj = nn.Linear(self.config.n_embd, 1)
        # self.relation_name_pooler_output = pickle.load(open('./data/ddi/relation_name_pooler_output.pkl','rb'))
    
    def forward(
        self,input_ids,attention_mask,labels
    ):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels = labels
        )
        return outputs.loss,outputs.logits