from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import os
from programmingalpha import AlphaConfig, AlphaPathLookUp
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from programmingalpha.models.InferenceNets  import construct_inference_net
from programmingalpha.Utility import getLogger
from .retriever_input_process import   FeatureProcessor
from tqdm import tqdm

logger=getLogger(__name__)

class KnowRetriever(object):

    def __init__(self, config_file):
        # loading config from file
        config=AlphaConfig.loadConfig(config_file)

        ## model parameters

        model=construct_inference_net(name=config.model)

        logger.info("loading weghts for {}".format(config.model))

        # running parameters
        if config.gpu_rank==-1:
            device=torch.device("cpu")
            map_location='cpu'
        else:
            device=torch.device("cuad", config.gpu_rank)
            map_location='cuda:{}'.format(config.gpu_rank)
        self.device=device

        model_state_dict = torch.load(os.path.join(config.model_path, "model.bin"),map_location=map_location)
        model.load_state_dict(model_state_dict)

        model.to(device)

        self.model=model

        logger.info("ranker model init finished!!!")

    def relationPredict(self,features,k=5):
        
        doc_ids=[id for id in range(len(features)) ]

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        eval_data = TensorDataset(input_ids, segment_ids)

        eval_dataloader = DataLoader(eval_data, shuffle=False,batch_size=self.batch_size)

        logits=self.getRelationProbability(eval_dataloader)
        #print("logits is {}".format(logits))

        dprobs=list(map(lambda doc_id, dp:{"Id": doc_id, "dist": int(dp[0]), "prob": float(dp[1]) }, doc_ids, logits ) )
        #print(dprobs)

        dprobs.sort(key=lambda x: x["prob"], reverse=True)
        dprobs.sort(key=lambda x: x["dist"], reverse=False)
        results=dprobs

        #print(results)

        
        return results[:k]

    def getRelationProbability(self,eval_dataloader:DataLoader):
        self.model.eval()
        device=self.device
        logits=[]

        for input_ids, segment_ids in tqdm(eval_dataloader,desc="computing simValue"):
            #logger.info("batch predicting")
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                b_logits = self.model(input_ids, segment_ids)
                #print("b_logits", b_logits)

            b_logits = b_logits.detach().cpu().numpy().tolist()

            logits.extend(b_logits)
        
        #print("logits original", logits)
        logits=map(lambda x:(np.argmax(x),np.max(x)), logits )
        logits=list(logits)
        #print("list logits", logits)
        #logits=np.concatenate(logits,axis=0)

        return logits




from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy
class KnowAlphaHTTPProxy(AlphaHTTPProxy):
    def __init__(self, config_file):
        AlphaHTTPProxy.__init__(self,config_file)
        args=self.args
        self.alphaModel=KnowRetriever(os.path.join(AlphaPathLookUp.ConfigPath, args.model_config_file))
        self.feature_processor=FeatureProcessor(os.path.join(AlphaPathLookUp.ConfigPath, args.processor_config_file))
    
    def processCore(self, data):
        features=self.feature_processor.process(data["question"], data["posts"])

        return self.alphaModel.relationPredict(features)

