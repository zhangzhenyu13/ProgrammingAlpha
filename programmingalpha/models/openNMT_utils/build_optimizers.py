from programmingalpha.Utility import getLogger
import random
import torch
import numpy as np
import onmt
from pytorch_transformers import AdamW
logger= getLogger(__name__)


def buildBertOptimizerW(model,opts):
    #configure optimizer
    logger.info("builing BertAdamW")
    random.seed(1237)
    np.random.seed(7453)
    torch.manual_seed(13171)

    lr = opts.learning_rate

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    bert_optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    
    optim = onmt.utils.optimizers.Optimizer(
        bert_optimizer, learning_rate=lr, max_grad_norm=1)
    
    return optim

