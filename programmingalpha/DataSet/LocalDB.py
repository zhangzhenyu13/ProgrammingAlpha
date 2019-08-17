from programmingalpha import DataPath
import os, json
import tqdm
from programmingalpha.Utility import getLogger

logger=getLogger(__name__)

#db skema
class DbSkema(object):
    def __init__(self, name, data_root=DataPath):
        self.name=name
        self.__data_db=os.path.join(data_root,"LocalDB")

        __path=os.path.join(self.__data_db,self.name)
        self.questions=os.path.join(__path,"Questions.json")
        self.answers=os.path.join(__path, "Answers.json")
        self.links=os.path.join(__path, "PostLinks.json")

    def loadCollection(self, path_local):
        collection={}
        with open(path_local, "r", encoding="utf-8") as f:
            data=map(lambda doc: json.loads(doc), f)
        
            for doc in tqdm.tqdm(data, "loading from {}".format(path_local)) :
                Id=doc["Id"]
                collection[Id]=doc
        
        logger.info("loaded {} records from {} : {}".format(len(collection),self.name, path_local))
        return collection



__supported_dbs={
        "ai",
        "datascience",
        "crossvalidated",
        "stackoverflow"
    }

#global local-db path
def getLocalDB(name, data_root=DataPath):
        if name not in __supported_dbs:
            raise RuntimeError("{} is not found in local db:{}".format(name,__supported_dbs))
        
        db=DbSkema(name, data_root=data_root)
        return db

