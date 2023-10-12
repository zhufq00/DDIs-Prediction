from torch import nn
from transformers import RobertaModel

# def init(module, weight_init, bias_init, gain=1):
#     weight_init(module.weight.data, gain=gain)
#     bias_init(module.bias.data)
#     return module

class roberta_actor(nn.Module):

    def __init__(self, args):
        super(roberta_actor, self).__init__()
        self.llm = RobertaModel.from_pretrained(args.pretrained_model_path)
        self.args = args
        self.config = self.llm.config
        self.values = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.ReLU(), nn.Linear(self.config.hidden_size, 1))
    
    def forward(
        self,input_ids,attention_mask
    ):
        batch_size,rel_num,max_length = input_ids.size()
        input_ids = input_ids.reshape(-1,max_length)
        attention_mask = attention_mask.reshape(-1,max_length)
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.last_hidden_state[:,0,:]
        logits = self.values(logits).view(batch_size,-1)
        return logits
        