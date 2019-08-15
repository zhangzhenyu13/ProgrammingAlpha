from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
import json
import logging
import argparse
import tqdm
import multiprocessing
from programmingalpha.Utility.TextPreprocessing import InformationAbstrator

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def init():
    global textExtractor
    textExtractor=InformationAbstrator(args.maxLength)
    textExtractor.initParagraphFilter(textExtractor.klSummary)


def fetchQuestionData():
    questionsData=[]

    for question in tqdm.tqdm(docDB.questions.find().batch_size(args.batch_size),desc="loading questions"):

        questionsData.append(question)

    logger.info("loaded: questions({})".format(len(questionsData)))

    return questionsData

def fetchAnswerData():
    answersData=[]
    for ans in tqdm.tqdm(docDB.answers.find().batch_size(args.batch_size),desc="loading answers"):

        answersData.append(ans)

    logger.info("loaded: answers({})".format(len(answersData)))

    return answersData


def _genCore(doc_ent):

    doc=doc_ent["Body"]
    '''
    try:
        doc=textExtractor.clipText(doc)
    except:
        print("error")
        print(doc)
        exit(10)
    #'''
    doc=textExtractor.clipText(doc)
    doc=" ".join(doc)

    if "Title" in doc_ent:
        doc=" ".join([doc_ent["Title"],doc])
    
    doc=doc.split()
    if len(doc)<10:
        return None

    doc=" ".join(doc)
    
    return doc.strip()


def generateContextAnswerCorpusParallel(doc_data):

    cache=[]
    batch_size=args.batch_size
    batches=[doc_data[i:i+batch_size] for i in range(0,len(doc_data),batch_size)]

    workers=multiprocessing.Pool(args.workers, initializer=init)

    file=programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-lm{}.txt".format(1 if args.answers else 2)
    with open(file,"w") as f:
        for batch_doc in tqdm.tqdm(batches,desc="processing documents multi-progress"):

            for record in workers.map(_genCore,batch_doc):
                if record is not None:

                    cache.append(record+"\n")

            f.writelines(cache)
            cache.clear()


        workers.close()
        workers.join()


def generateContextAnswerCorpus(doc_data):

    cache=[]
    
    init()
    file=programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-lm{}.txt".format(1 if args.answers else 2)
    with open(file,"w") as f:
        for record in tqdm.tqdm(doc_data,desc="processing documents"):
            record=_genCore(record)
            if record is not None:
                cache.append(record+"\n")

            f.writelines(cache)
            cache.clear()



def main():
    if not args.answers:
        doc_data=fetchQuestionData()
    else:
        doc_data=fetchAnswerData()

    if args.workers<2:
        generateContextAnswerCorpus(doc_data)
    else:
        generateContextAnswerCorpusParallel(doc_data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--db', type=str, default="crossvalidated")
    parser.add_argument("--answers",action="store_true")
    parser.add_argument("--maxLength",type=int, default=500)

    parser.add_argument('--workers', type=int, default=10)
    
    args = parser.parse_args()

    docDB=MongoStackExchange(host='10.1.1.9',port=50000)
    dbName=args.db
    docDB.useDB(dbName)

    logger.info("processing db data: {}".format(dbName))

    main()
