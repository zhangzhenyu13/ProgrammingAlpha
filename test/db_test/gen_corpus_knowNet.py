from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
import logging
import argparse
import tqdm
import json
import multiprocessing
from programmingalpha.tokenizers import BertTokenizer
from programmingalpha.Utility.TextPreprocessing import InformationAbstrator

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)

def init(questionsData_G, answersData_G, indexData_G):


    global tokenizer,textExtractor
    never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]","[NUM]","[CODE]")
    tokenizer=BertTokenizer(programmingalpha.ModelPath+"knowledgeSearcher/vocab.txt",never_split=never_split)
    textExtractor=InformationAbstrator(args.postLen,tokenizer)

    filter_funcs={
        "pagerank":textExtractor.page_rank_texts,
        "lexrankS":textExtractor.lexrankSummary,
        "klS":textExtractor.klSummary,
        "lsaS":textExtractor.lsarankSummary,
        "textrankS":textExtractor.textrankSummary,
        "reductionS":textExtractor.reductionSummary
    }
    textExtractor.initParagraphFilter(filter_funcs[args.extractor])

    global questionsData, answersData, indexData
    questionsData=questionsData_G.copy()
    answersData=answersData_G.copy()
    indexData=indexData_G.copy()

    logger.info("process {} init".format(multiprocessing.current_process()))


def fetchQuestionData(q_ids_set):
    questionsData={}
    query={
        "$or":[
            {"FavoriteCount":{"$gte":1}},
         ]
    }

    for question in tqdm.tqdm(docDB.questions.find(query).batch_size(args.batch_size),desc="loading questions"):

        Id=question["Id"]
        if Id not in q_ids_set:
            continue
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
def _getPreprocess(txt,maxLen,cal_lose=False):
    textExtractor.maxClip=maxLen
    if cal_lose:
        original=" ".join(textExtractor.processor.getPlainTxt(txt))
        before_len=len( original.split() )
        if before_len<5:
            #logger.info("bad zero:{}\n=>{}".format(txt,original))
            return "",0
    txt_processed=textExtractor.clipText(txt)

    if cal_lose:
        after_len=len(" ".join(txt_processed).split())
        lose_rate= after_len/before_len
        return txt_processed, lose_rate
    else:
        return txt_processed,None

def _getBestAnswers(q_id,K):
    answers=[]
    ans_id=-1
    if "AcceptedAnswerId" in questionsData[q_id]:
        ans_id=questionsData[q_id]["AcceptedAnswerId"]

        if ans_id in answersData:
            answer=answersData[ans_id]
            K-=1

    ans_idx=indexData[q_id]["Answers"]
    scored=[]
    for id in ans_idx:
        if id!=ans_id and id in answersData:
            scored.append((id,answersData[id]["Score"]))
    if scored:
        scored.sort(key=lambda x:x[1],reverse=True)
        for i in range(min(K-1,len(scored))):
            id=scored[i][0]
            answers.append(answersData[id])

    if K<args.answerNum:
        answers=[answer]+answers


    return answers


def _genCore(link):

    label=link["label"]
    q1,q2=link["pair"]

    if not (q1 in questionsData and q2 in questionsData):
        #if label=='duplicate':
        #    print(link,q1 in questionsData, q2 in questionsData)
        return None

    question1=questionsData[q1]
    question2=questionsData[q2]

    title1=" ".join(textExtractor.tokenizer.tokenize(question1["Title"]))
    title2=" ".join(textExtractor.tokenizer.tokenize(question2["Title"]))

    body1=question1["Body"]
    body2=question2["Body"]

    body1, _ =_getPreprocess(body1, args.questionLen)
    body2, _ =_getPreprocess(body2, args.questionLen)


    question1=[title1]+body1
    question2=[title2]+body2

    #answer1
    answers=_getBestAnswers(q1,args.answerNum)
    answers = list(map(lambda ans: ans["Body"], answers))
    answers = " \n".join(answers)
    answer1,_ = _getPreprocess(answers, args.postLen)

    #answer2
    answers = _getBestAnswers(q2, args.answerNum)
    answers = list(map(lambda ans: ans["Body"], answers))
    answers = " \n".join(answers)
    answer2,_ = _getPreprocess(answers, args.postLen)


    record={"id_1":q1,"id_2":q2,"label":label,"q1":question1,"q2":question2,"answer1":answer1,"answer2":answer2}

    return record

def generateQuestionCorpus(labelData, questionsDataGlobal, answersDataGlobal, indexDataGlobal):

    cache=[]
    batch_size=args.batch_size

    batches=[labelData[i:i+batch_size] for i in range(0,len(labelData),batch_size)]

    workers=multiprocessing.Pool(args.workers,initializer=init,initargs=(questionsDataGlobal, answersDataGlobal, indexDataGlobal) )

    counter={'unrelated': 0, 'direct': 0, 'transitive': 0, 'duplicate': 0}

    with open(programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-knowNet.json","w") as f:
        for batch_labels in tqdm.tqdm(batches,desc="processing input for knowNet"):
            for record in workers.map(_genCore,batch_labels):
                if record is not None:
                    counter[record["label"]]+=1
                    #cache.append(record)
                    cache.append(json.dumps(record)+"\n")

            f.writelines(cache)
            cache.clear()

        workers.close()
        workers.join()

    logger.info("after extratcing informatiev paragraphs: {}".format(counter))

def main():

    labelData=[]
    q_ids_set=set()
    with open(programmingalpha.DataPath+"/linkData/"+args.db.lower()+"-labelPair.json","r") as f:
        for line in f:
            record=json.loads(line)
            labelData.append(record)
            q_ids_set.update(record["pair"])


    questionsDataGlobal=fetchQuestionData(q_ids_set)
    q_ids_set=questionsDataGlobal.keys()
    indexDataGlobal=fetchIndexData(q_ids_set)
    answerDataGlobal=fetchAnswerData(q_ids_set)

    labelDataNew=[]
    for ld in labelData:
        id1,id2=ld["pair"]
        if id1 not in q_ids_set or id2 not in q_ids_set:
            continue
        labelDataNew.append(ld)

    labels=map(lambda ll:ll["label"],labelData)

    import collections
    logger.info(collections.Counter(labels))

    generateQuestionCorpus(labelDataNew,questionsDataGlobal, answerDataGlobal, indexDataGlobal)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--db', type=str, default="crossvalidated")
    parser.add_argument('--questionLen', type=int, default=250)
    parser.add_argument('--postLen', type=int, default=1250)
    parser.add_argument("--answerNum",type=int,default=3)

    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--extractor', type=str, default="lexrankS")

    args = parser.parse_args()

    docDB = MongoStackExchange(host='10.1.1.9', port=50000)
    dbName = args.db
    docDB.useDB(dbName)

    logger.info("task db is {}".format(args.db))

    main()
