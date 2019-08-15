from programmingalpha.tokenizers import get_tokenizer
import programmingalpha
import argparse
import os

path_map_tokenizers={
    "bert":programmingalpha.BertBaseUnCased,
    "gpt2": programmingalpha.GPT2Base,
    "xlnet": programmingalpha.XLNetBaseCased,
    "roberta":programmingalpha.RoBertaBase
}

def init():
    global tokenizer
    name= args.tokenizer
    tokenizer=get_tokenizer(path_map_tokenizers[name], name)


def tokenize(text):
    text=text[0]
    #print("trying to tokenize=>",text)
    tokenized_text=tokenizer.tokenizeLine(text)

    return " ".join(tokenized_text)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--tokenizer",type=str,default="bert", choices=["bert", "roberta", "gpt2","xlnet"])
    parser.add_argument("--file",type=str,default="")

    args=parser.parse_args()

    if args.tokenizer not in path_map_tokenizers:
        args.tokenizer="bert"

    inputfile=args.file
    outputfile=inputfile+"-tokenized-"+args.tokenizer

    init()
    from pyspark.sql import SparkSession
    spark = SparkSession\
        .builder\
        .appName("tokenize text with "+args.tokenizer)\
        .getOrCreate()

    tokenized=spark.read.text(inputfile).rdd.map(tokenize)

    tokenized.saveAsTextFile(outputfile)
    spark.stop()
