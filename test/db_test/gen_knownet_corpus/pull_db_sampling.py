from programmingalpha.DataSet.DBLoader import MongoStackExchange
import argparse
import random
import tqdm
from programmingalpha import AlphaPathLookUp
import os
import json
import logging
from collections import Counter
import random
from programmingalpha.Utility import getLogger

logger=getLogger(__name__)



def samplingFeatures():

    collection=docDB.stackdb[collection_name ]
    duplicates=list(collection.find({"label_id":0}).batch_size(args.batch_size))

    size=len(duplicates)
    if args.maxSize>0 and args.maxSize<=size*4:
        size=args.maxSize//4

    query1=[
          {"$match": {"label_id":0}},
          {"$sample": {"size": size}}
        ]

    query2=[
          {"$match": {"label_id":1}},
          {"$sample": {"size": size}}
        ]

    query3=[
          {"$match": {"label_id":2}},
          {"$sample": {"size": size}}
        ]

    query4=[
          {"$match": {"label_id":3}},
          {"$sample": {"size": size}}
        ]
    queries=[query1,query2,query3,query4]

    dataSet=[]
    labels=[]
    for query in queries:
        data_samples=list(collection.aggregate(pipeline=query,allowDiskUse=True))
        for record in tqdm.tqdm(data_samples,desc="{}".format(query)):
            del record["_id"]
            labels.append(record["label_id"])
            dataSet.append(record)
    
    logger.info("laebls:{}".format(Counter(labels)))

    random.shuffle(dataSet)

    sample_file=os.path.join(AlphaPathLookUp.DataPath,"knowNet/data.json")

    logger.info("saving data to "+sample_file)
    with open(sample_file,"w") as f:
        lines=map(lambda record: json.dumps(record)+"\n", dataSet)
        f.writelines(lines)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--db', type=str, default="corpus")
    parser.add_argument('--maxSize', type=int, default=-1)
    parser.add_argument('--task', type=str, default="features")

    args = parser.parse_args()

    docDB=MongoStackExchange(host='10.1.1.9',port=50000)
    docDB.useDB(args.db)
    supported_features={
        "features-bert":"features-bert",
        "features-xlnet":"features-xlnet"
    }
    if args.task not in supported_features:
        raise ValueError("{} are not supported".format(args.task))
    
    collection_name=supported_features[args.task]
    logger.info("task is "+args.task)
    samplingFeatures()
