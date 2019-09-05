#from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy
from programmingalpha.alphaservices.HTTPServers.tornado_http import AlphaHTTPProxy

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

        model=construct_inference_net(name=config.name)

        logger.info("loading weghts for {}".format(config.name))

        # running parameters
        if torch.cuda.is_available()==False or config.use_gpu==False:
            n_gpu=0
            device=torch.device("cpu")
            map_location='cpu'
        else:
            n_gpu=torch.cuda.device_count()
            if n_gpu<2:
                device=torch.device("cuda:0") 
                map_location='cuda:{}'.format(config.gpu_rank)
            else:
                device= torch.device("cuda")
                map_location='cuda'
                torch.cuda.manual_seed_all(43)
        
        torch.manual_seed(43)
        self.device=device
        logger.info("{}, {} gpus".format(self.device, n_gpu))

        model_state_dict = torch.load(config.dict_path,map_location=map_location)
        model.load_state_dict(model_state_dict)

        model.to(device)

        self.model=model

        self.batch_size=config.batch_size

        logger.info("ranker model init finished!!!")

    def relationPredict(self,features):
        
        doc_ids=[id for id in range(len(features)) ]

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        eval_data = TensorDataset(input_ids, segment_ids)
        sampler=SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=sampler,batch_size=self.batch_size)

        logits=self.getRelationProbability(eval_dataloader)
        #print("logits is {}".format(logits))

        dprobs=list(map(lambda doc_id, dp:{"Id": doc_id, "dist": int(dp[0]), "prob": float(dp[1]) }, doc_ids, logits ) )
        #print(dprobs)

        dprobs.sort(key=lambda x: x["prob"], reverse=True)
        dprobs.sort(key=lambda x: x["dist"], reverse=False)
        results=dprobs

        #print(results)

        
        return results

    def getRelationProbability(self,eval_dataloader:DataLoader):
        self.model.eval()
        device=self.device
        logits=[]
        #print("to device", device)
        for input_ids, segment_ids in tqdm(eval_dataloader,desc="computing dist of <Q-post> pairs"):
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




class KnowAlphaHTTPProxy(AlphaHTTPProxy):
    def __init__(self, config_file):
        AlphaHTTPProxy.__init__(self,config_file)
        args=self.args
        self.alphaModel=KnowRetriever(os.path.join(AlphaPathLookUp.ConfigPath, args.model_config_file))
        self.feature_processor=FeatureProcessor(os.path.join(AlphaPathLookUp.ConfigPath, config_file))
        self.top_K=args.top_K
    def processCore(self, data):
        question, posts= data["question"], data["posts"]
        features=self.feature_processor.process(question, posts)

        res= self.alphaModel.relationPredict(features)
        
        ranks=[]
        for i in range(len(res)):
            post=posts[res[i]["Id"]]
            if "answers" not in post or not post["answers"]:
                continue
            ranks.append(res[i])
            
            if len(ranks)>=self.top_K:
                break

        return ranks[:self.top_K]

