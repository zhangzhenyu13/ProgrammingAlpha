from programmingalpha.alphaservices.KafkaMPI.kafka_node import AlpahaKafkaNode
from programmingalpha.retrievers.relation_searcher import KnowRetriever
from programmingalpha import AlphaConfig, AlphaPathLookUp
from programmingalpha.Utility import getLogger
from .retriever_input_process import   FeatureProcessor
from tqdm import tqdm
from multiprocessing import Pool
import os
logger=getLogger(__name__)

class KnowAlphaKafkaNode(AlpahaKafkaNode):
    def __init__(self, config_file):
        AlpahaKafkaNode.__init__(self,config_file)
        args=self.args
        self.alphaModel=KnowRetriever(os.path.join(AlphaPathLookUp.ConfigPath, args.model_config_file))
        self.feature_processor=FeatureProcessor(os.path.join(AlphaPathLookUp.ConfigPath, config_file))
        self.num_workers=args.num_workers
        self.top_K=args.top_K
    

    def processCore(self, data):
        question, posts= data["question"], data["posts"]
        ranks={
            "question":question,
            "posts":posts[:self.top_K],
            "rank_status": "Failed!"
        }
        try:
            if len(posts)<15:
                features=self.feature_processor.process(question, posts)
            else:
                records=map(lambda q, p: {"question":q, "post":p, "label":"direct"}, [question]*len(posts), posts) 
                with Pool(self.num_workers) as workers:
                    features=workers.map(self.feature_processor.batch_process_core, records)
            
            res= self.alphaModel.relationPredict(features)
            
            posts_sel=[]
            for i in range(len(res)):
                post=posts[res[i]["Id"]]
                if "answers" not in post or not post["answers"]:
                    continue
                #ranks.append(res[i])
                posts_sel.append(post)
                
                if len(posts_sel)>=self.top_K:
                    break
            ranks={
                "question":question,
                "posts":posts_sel[:self.top_K],
                "rank_status":"Finished!"
            }
        except Exception as e:
            logger.error("{}".format(e.args))
        
        return ranks
