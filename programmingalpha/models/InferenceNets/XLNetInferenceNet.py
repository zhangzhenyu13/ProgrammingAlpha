import torch
from pytorch_transformers import XLNetModel, XLNetConfig
from pytorch_transformers.modeling_utils import SequenceSummary
import programmingalpha
import math
from torch.nn import CrossEntropyLoss
from copy import deepcopy
from programmingalpha.Utility import getLogger
from .InferenceNetBase import InferenceNet
from ..import expandEmbeddingByN

logger = getLogger(__name__)

#LinkNet
class LinkNet(InferenceNet):

    def __init__(self, model_path=None, num_labels=4 ):
        super(LinkNet, self).__init__(model_path, num_labels)
        self.encoder ,self.config= self._load_model(XLNetModel, XLNetConfig)
        self.encoderl.word_embedding=expandEmbeddingByN(self.encoder.word_embedding, 2, last=True)

        self.summary=SequenceSummary(self.config)
        self.classifier  = torch.nn.Linear(self.config.d_model, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, labels=None):
        outputs=self.encoder.forward(input_ids, token_type_ids=token_type_ids)
        pooled_output=self.summary(outputs[0])
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct_classification = CrossEntropyLoss()
            classfication_loss = loss_fct_classification(logits.view(-1, self.num_labels), labels.view(-1))
            return classfication_loss
        else:
            return logits
