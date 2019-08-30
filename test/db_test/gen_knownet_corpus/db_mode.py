from programmingalpha.DataSet.DBLoader import MongoStackExchange
from programmingalpha import AlphaConfig, AlphaPathLookUp
import logging
import argparse
import tqdm
import json
import multiprocessing
from programmingalpha.Utility import getLogger

logger = getLogger(__name__)

def init(postData_local):
    global postData
    postData=postData_local

def fetchPostData(q_ids_set, source):
    postsData_local={}

    query={
        "source":source
    }

    for post in tqdm.tqdm(docDB.stackdb["posts"].find(query).batch_size(args.batch_size),desc="loading posts"):

        Id=post["Id"]
        if Id not in q_ids_set:
            continue
        del post["_id"]
        postsData_local[Id]=post

    logger.info("loaded: posts({})".format(len(postsData_local)))

    return  postsData_local


def _genCore(link):

    label=link["label"]
    q1,q2=link["pair"]

    if not (q1 in postData and q2 in postData):
        #if label=='duplicate':
        #    print(link,q1 in questionsData, q2 in questionsData)
        return None

    post1=postData[q1]
    post2=postData[q2]
    
    record={"label":label,"post1":post1,"post2":post2}

    return record

def generateQuestionCorpus(labelData, postData_local):

    res=[]

    counter={'unrelated': 0, 'direct': 0, 'transitive': 0, 'duplicate': 0}
    init(postData_local)
    
    with open(AlphaPathLookUp.DataPath+"Corpus/"+args.source.lower()+"-knowNet.json","w") as f:
        for label_data in tqdm.tqdm(labelData,desc="processing input for knowNet"):
            record=_genCore(label_data)
            if record is not None:
                counter[record["label"]]+=1
                #cache.append(record)
                res.append(json.dumps(record)+"\n")

            if len(res)>args.batch_size:
                f.writelines(res)
                f.flush()
                res.clear()

        #finished running
        if len(res)>0:
            f.writelines(res)
            res.clear()

    logger.info("labels: {}".format(counter))

def main():

    labelData=[]
    q_ids_set=set()
    with open(AlphaPathLookUp.DataPath+"/linkData/"+args.source+"-labelPair.json","r") as f:
        for line in f:
            record=json.loads(line)
            labelData.append(record)
            q_ids_set.update(record["pair"])


    postData_local=fetchPostData(q_ids_set, args.source)

    labelDataNew=[]
    for ld in labelData:
        id1,id2=ld["pair"]
        if id1 not in q_ids_set or id2 not in q_ids_set:
            continue
        labelDataNew.append(ld)

    labels=map(lambda ll:ll["label"],labelData)

    import collections
    logger.info(collections.Counter(labels))

    generateQuestionCorpus(labelDataNew,postData_local)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--source', type=str, default="crossvalidated")


    args = parser.parse_args()

    docDB = MongoStackExchange(host='10.1.1.9', port=50000)
    docDB.useDB("posts")

    logger.info("task source is {}".format(args.source))

    main()
