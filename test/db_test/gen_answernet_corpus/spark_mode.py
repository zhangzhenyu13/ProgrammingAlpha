import programmingalpha
import json
import argparse
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
        return ""

    src_txts=[]
    q_len=args.question_len -len(post["Title"].split())
    q_txt=post["Title"]
    if q_len>20:
        q_txt+=" ".join(_getPreprocess(post["Body"],  q_len))

    q_txt=" ".join(["[CLS]",q_txt ,"[SEP]"])
    
    src_txts.append(q_txt)
    
    random.shuffle(answers)
    for i in range(len(answers)):
        answer=answers[i]
        if answer["Id"]==target_answer["Id"]:
            src_txts.append(tgt_txt)
        else:
            ans_txt=" ".join(_getPreprocess(answer["Body"], args.answer_len))
            src_txts.append(ans_txt)
    
    src_txt=" ".join(src_txts) +"[SEP]"

    record={"src":src_txt, "tgt":tgt_txt}

    return json.dumps(record)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, required=True)

    parser.add_argument('--relative_num', type=int, default=3)

    parser.add_argument('--target_answer_len', type=int, default=100)
    parser.add_argument('--answer_len', type=int, default=80)
    parser.add_argument('--question_len', type=int, default=80)

    args = parser.parse_args()
    init()

    from pyspark.sql import SparkSession
    spark = SparkSession\
        .builder\
        .appName("gen seq2seq json")\
        .getOrCreate()

    input_file=args.input_file
    output_file=input_file.replace(".json", "-seq2seq.json")
    
    doc_data=spark.read.json(input_file).rdd.filter(lambda post: post["AnswerCount"]>=args.relative_num).map(_genCore).filter(lambda s:s and s.strip())
    doc_data.saveAsTextFile(output_file)