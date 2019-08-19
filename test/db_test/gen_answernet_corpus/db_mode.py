from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
import json
import argparse
import tqdm
import random
from programmingalpha.Utility import getLogger
from programmingalpha.Utility.TextPreprocessing import InformationAbstrator
logger = getLogger(__name__)


def init():

    global textExtractor
    textExtractor=InformationAbstrator(100)

    filter_funcs={
        "pagerank":textExtractor.page_rank_texts,
        "lexrankS":textExtractor.lexrankSummary,
        "klS":textExtractor.klSummary,
        "lsaS":textExtractor.lsarankSummary,
        "textrankS":textExtractor.textrankSummary,
        "reductionS":textExtractor.reductionSummary
    }
    textExtractor.initParagraphFilter(filter_funcs["lexrankS"])

#generate Core
def _getPreprocess(txt,maxLen):
    textExtractor.maxClip=maxLen
    
    txt_processed=textExtractor.clipText(txt)

    return txt_processed

def _getBestAnswers(post):
    answers= post["answers"][:-1]

    if "AcceptedAnswerId" in post :

        acc_ans_id=post["AcceptedAnswerId"]
        if answers and acc_ans_id == answers[-1]["Id"]:
            answer= answers[-1]
            answers.insert(0, answer)

    answers=answers[:args.relative_num]    

    return answers

def _genCore(post):

    answers=_getBestAnswers(post)
    target_answer=answers[0]
    tgt_txt=" ".join(_getPreprocess(target_answer["Body"], args.target_answer_len) )

    if  not tgt_txt or not tgt_txt.strip():
        return None

    src_txts=[]
    q_len=args.question_len -len(post["Title"].split())
    q_txt=post["Title"]
    if q_len>20:
        q_txt+=" ".join(_getPreprocess(post["Body"],  q_len))

    src_txts.append(q_txt)
    
    random.shuffle(answers)
    for i in range(len(answers)):
        answer=answers[i]
        if answer["Id"]==target_answer["Id"]:
            src_txts.append(tgt_txt)
        else:
            ans_txt=" ".join(_getPreprocess(answer["Body"], args.answer_len))
            src_txts.append(ans_txt)
    
    src_txt=" ".join(src_txts)

    record={"src":src_txt, "tgt":tgt_txt}

    return record

def processData():
    init()
    query={
        "AnswerCount" : {"$gte":args.relative_num}
    }

    cache=[]
    with open(programmingalpha.DataPath+"Corpus/seq2seq.json","w") as f:
        for post in tqdm.tqdm(docDB.stackdb["posts"].find(query).batch_size(args.batch_size),desc="loading posts of "):

            record =_genCore(post)
            if record is not None:
                cache.append(json.dumps(record)+"\n")

            if len(cache)>args.batch_size:
                f.writelines(cache)
                cache.clear()
                f.flush()

        if len(cache)>0:
            f.writelines(cache)
            cache.clear()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--relative_num', type=int, default=5)

    parser.add_argument('--target_answer_len', type=int, default=100)
    parser.add_argument('--answer_len', type=int, default=80)
    parser.add_argument('--question_len', type=int, default=80)


    args = parser.parse_args()

    docDB=MongoStackExchange(host='10.1.1.9',port=50000)
    docDB.useDB("posts")

    processData()