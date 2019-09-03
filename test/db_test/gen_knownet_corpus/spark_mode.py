import argparse
import json
from programmingalpha.retrievers.retriever_input_process import FeatureProcessor
from programmingalpha.Utility import getLogger

logger = getLogger(__name__)


def _get_question_from_post(post):
    data={
        "Title":post["Title"],
        "Body":post["Body"]
    }
    return data

def _get_post_from_post(post):
    data={
        "Title":post["Title"],
        "Body":post["Body"],
        "answers":[]
    }
    if "answers" not in post:
        return data
            
    data["answers"]=post["answers"]
    
    return data

def extract_question_post(record):
    if not record:
        return None

    post1=record["post1"]
    post2=record["post2"]
    
    data=[]

    label=record["label"]

    #q1-p2
    
    data.append(
        {
            "question": _get_question_from_post(post1),
            "post": _get_post_from_post(post2),
            "label":label
        }
    )

    #q2-p1

    data.append(
        {
            "question": _get_question_from_post(post2),
            "post": _get_post_from_post(post1),
            "label":label
        }
    )

    return data

def convert_to_features(json_record):
        records=extract_question_post(json_record)
        if not records:
            return ""
        features=[]
        for record in records:
            feature=processor.batch_process_core(record)
            features.append(feature)
        return features

if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--file', type=str,  required=True)

    args = parser.parse_args()

    processor=FeatureProcessor("knowAlphaService.json")
    
    input_file=args.file
    output_file=input_file.replace(".json","-features")


    from pyspark.sql import SparkSession
    spark = SparkSession\
        .builder\
        .appName("gen question-post pairs features")\
        .getOrCreate()

    doc_data=spark.read.json(input_file).rdd.flatMap( convert_to_features )#.filter(lambda s:s and s.strip())
    doc_data.saveAsTextFile(output_file)
