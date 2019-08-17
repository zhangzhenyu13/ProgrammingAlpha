from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
import json
import logging
import argparse
import tqdm
import multiprocessing

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def init(questionsData_G,answersData_G,indexData_G,copy=True):

    global questionsData,answersData,indexData
    if copy:
        questionsData=questionsData_G.copy()
        answersData=answersData_G.copy()
        indexData=indexData_G.copy()
    else:
        questionsData=questionsData_G
        answersData=answersData_G
        indexData=indexData_G

    logger.info("process {} init".format(multiprocessing.current_process()))


def fetchQuestionData():
    questionsData={}

    for question in tqdm.tqdm(docDB.questions.find().batch_size(args.batch_size),desc="loading questions"):

        Id=question["Id"]

        del question["_id"]
        questionsData[Id]=question


    logger.info("loaded: questions({})".format(len(questionsData)))

    return questionsData

def fetchAnswerData(questionsDataGlobal):
    answersData={}
    #query={
    #    "ParentId":{"$in":list(questionsDataGlobal)}
    #}
    for ans in tqdm.tqdm(docDB.answers.find().batch_size(args.batch_size),desc="loading answers"):

        Id=ans["Id"]

        if  ans["ParentId"] not in questionsDataGlobal:
            continue

        del ans["_id"]
        answersData[Id]=ans

    logger.info("loaded: answers({})".format(len(answersData)))

    return answersData

def fetchIndexData(questionDataGlobal):
    indexData={}
    #query={
    #    "Id":{"$in":list(questionDataGlobal)}
    #}

    for indexer in tqdm.tqdm(docDB.stackdb["QAIndexer"].find().batch_size(args.batch_size),desc="loading indexers"):
        Id=indexer["Id"]

        if Id not in questionDataGlobal:
            continue

        del indexer["_id"]

        indexData[Id]=indexer

    logger.info("loaded: indexer({})".format(len(indexData)))

    return indexData

#generate Core
def _getBestAnswers(q_id):
    answers=[]

    '''get accepted answer id'''    
    accepted_ans_id=-1
    if "AcceptedAnswerId" in questionsData[q_id]:
        accepted_ans_id=questionsData[q_id]["AcceptedAnswerId"]


    'get all the answers'
    ans_idx=indexData[q_id]["Answers"]
    scored=[]
    for id in ans_idx:
        if id!=accepted_ans_id and id in answersData:
            scored.append((id,answersData[id]["Score"]))
    if scored:
        scored.sort(key=lambda x:x[1],reverse=True)
        for i in range(len(scored)):
            id=scored[i][0]
            ans=answersData[id]
            answers.append(ans)

    if accepted_ans_id in answersData:
        ans=answersData[accepted_ans_id]
        answers.append(ans)

    return answers


def _genCore(q_id):

    #get question
    if q_id not in questionsData:
        return None

    question=questionsData[q_id]

    #get answer
    answers=_getBestAnswers(q_id)


    record={"answers":answers,"source":dbName.lower()}
    record.update(question)
    return record


def generateContextAnswerCorpusParallel(questionsDataGlobal,answersDataGlobal,indexDataGlobal):

    cache=[]
    batch_size=args.batch_size
    question_ids=list(questionsDataGlobal.keys())
    batches=[question_ids[i:i+batch_size] for i in range(0,len(question_ids),batch_size)]

    workers=multiprocessing.Pool(args.workers,initializer=init,
                                 initargs=(questionsDataGlobal,answersDataGlobal,indexDataGlobal)
                                 )

    with open(programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-posts.json","w") as f:
        for batch_qids in tqdm.tqdm(batches,desc="processing documents"):

            for record in workers.map(_genCore,batch_qids):
                if record is not None:

                    cache.append(json.dumps(record)+"\n")

            f.writelines(cache)
            cache.clear()


        workers.close()
        workers.join()

def generateContextAnswerCorpus(questionsDataGlobal,answersDataGlobal,indexDataGlobal):

    cache=[]
    question_ids=list(questionsDataGlobal.keys())

    init(questionsDataGlobal,answersDataGlobal,indexDataGlobal,copy=False)

    with open(programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-posts.json","w") as f:
        for q_id in tqdm.tqdm(question_ids,desc="processing documents"):
            record =_genCore(q_id)
            if record is not None:
                cache.append(json.dumps(record)+"\n")

            if len(cache)>args.batch_size:
                f.writelines(cache)
                cache.clear()

        if len(cache)>0:
            f.writelines(cache)
            cache.clear()

def main():

    questionsDataGlobal=fetchQuestionData()
    answersDataGlobal=fetchAnswerData(questionsDataGlobal.keys())
    indexerDataGlobal=fetchIndexData(questionsDataGlobal.keys())

    generateContextAnswerCorpusParallel(questionsDataGlobal,answersDataGlobal,indexerDataGlobal)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)

    parser.add_argument('--db', type=str, default="crossvalidated")

    parser.add_argument('--workers', type=int, default=10)

    args = parser.parse_args()

    docDB=MongoStackExchange(host='10.1.1.9',port=50000)
    dbName=args.db
    docDB.useDB(dbName)

    logger.info("processing db data: {}".format(dbName))

    main()
