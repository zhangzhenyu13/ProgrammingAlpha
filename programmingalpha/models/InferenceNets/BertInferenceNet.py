from pytorch_transformers.modeling_bert import BertConfig, BertModel
from torch.nn import CrossEntropyLoss
from torch import nn
import torch
import math
from .InferenceNetBase import InferenceNet
from ..import expandEmbeddingByN

class AttnBertPooler(nn.Module):
    def __init__(self, config):
        super(AttnBertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.activation = nn.Tanh()
        self.hidden_size=config.hidden_size

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0].view(len(hidden_states),-1,1)
        #print("first token tensor",first_token_tensor.size())
        #print("mask",attention_mask.size())
        scores=torch.matmul(hidden_states[:,1:], first_token_tensor)/math.sqrt(self.hidden_size)
        #print("scores",scores.size())
        attn_token_tensor=torch.matmul( hidden_states[:,1:].view(hidden_states.size(0),self.hidden_size,-1), scores )
        #print("attention tensor1",attn_token_tensor.size())
        attn_token_tensor=attn_token_tensor.view( attn_token_tensor.size(0), self.hidden_size )
        #print("attention tensor2",attn_token_tensor.size())

        first_token_tensor=first_token_tensor.squeeze(2)
        pooled_token_tensor=torch.cat((attn_token_tensor,first_token_tensor),dim=-1)
        #attn_token_tensor=attn_token_tensor.unsqueeze(2)
        #pooled_token_tensor=torch.cat((attn_token_tensor,first_token_tensor),dim=1)

        #print("pooled tensor",pooled_token_tensor.size())
        pooled_output = self.dense(pooled_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

#LinkNet
class LinkNet(InferenceNet):

    def __init__(self, model_path=None,num_labels=4 ):
        InferenceNet.__init__(self, model_path, num_labels)
        encoder , config= self._load_model(BertModel, BertConfig)
        self.encoder=encoder
        self.config=config
        self.encoder.embeddings.word_embeddings=expandEmbeddingByN(self.encoder.embeddings.word_embeddings, 2, last=True)

        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, labels=None):
        outputs=self.encoder.forward(input_ids, token_type_ids=token_type_ids)
        pooled_output=outputs[1]
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct_classification = CrossEntropyLoss()
            classfication_loss = loss_fct_classification(logits.view(-1, self.num_labels), labels.view(-1))
            return classfication_loss
        else:
            return logits

#KnowNet
class KnowNet(InferenceNet):
    def __init__(self, model_path=None, num_labels=4):
        super(KnowNet, self).__init__(model_path, num_labels)
        
        self.encoder ,self.config= self._load_model(BertModel, BertConfig)
        self.encoder.embeddings.word_embeddings=expandEmbeddingByN(self.encoder.embeddings.word_embeddings, 2, last=True)

        self.attnpooler=AttnBertPooler(self.config)
        self.classifier = nn.Linear(self.config.hidden_size*2, self.num_labels)
        self.num_labels=self.num_labels


    def forward(self, input_ids, token_type_ids=None, labels=None):
        encoder_outputs=self.encoder.forward(input_ids, token_type_ids=token_type_ids)
        sequence_output = encoder_outputs[0]        

        pooled_output = self.attnpooler(sequence_output)

        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct_classification = CrossEntropyLoss()
            classfication_loss = loss_fct_classification(logits.view(-1, self.num_labels), labels.view(-1))
            return classfication_loss
        else:
            return logits
            