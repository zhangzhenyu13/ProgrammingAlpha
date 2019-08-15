from programmingalpha.DataSet.LocalDB import getLocalDB
import programmingalpha
import json
import logging
import argparse
from programmingalpha.Utility.TextPreprocessing import PreprocessPostContent
import os

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def init():
    global preprocessor
    preprocessor=PreprocessPostContent()


def _genCore(doc_json):
    
    docs=preprocessor.getPlainTxt(doc_json["Body"])
    
    if "Title" in doc_json:
        docs.insert(0, doc_json["Title"])
    
    word_count=0
    words=[]
    res=[]
    for doc in docs:
        w_doc=doc.split()
        word_count+=len(w_doc)

        if word_count> args.maxLength:
            res.append(" ".join(words))
            words=w_doc
            word_count=len(w_doc)
        else:
            words.extend(w_doc)

    doc_str="\n".join(words)
    return doc_str.strip()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--db', type=str, default="crossvalidated")
    parser.add_argument('--data_root', type=str, default="/data/")

    parser.add_argument("--answers",action="store_true")
    parser.add_argument("--maxLength",type=int, default=500)
    
    args = parser.parse_args()

    
    docDB=getLocalDB(args.db, args.data_root)

    logger.info("processing db data: {}".format(docDB.name))
    
    init()
    from pyspark.sql import SparkSession
    spark = SparkSession\
        .builder\
        .appName("gen plain text"+args.tokenizer)\
        .getOrCreate()

    input_file=docDB.answers if args.answers else docDB.questions
    output_file=input_file.replace(".json", ".txt")
    
    doc_data=spark.read.json(input_file).rdd.map(_genCore)

    
