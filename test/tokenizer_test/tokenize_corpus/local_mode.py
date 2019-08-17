from programmingalpha.tokenizers import get_tokenizer
import programmingalpha
import argparse
import os
import multiprocessing
import tqdm
from programmingalpha.Utility import getLogger
logger=getLogger(__name__)

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
    tokenized_text=tokenizer.tokenizeLine(text, add_sp=False)

    return tokenized_text

def tokenizeParallel(doc_data):

    cache=[]
    batch_size=args.batch_size
    batches=[doc_data[i:i+batch_size] for i in range(0,len(doc_data),batch_size)]

    workers=multiprocessing.Pool(args.workers, initializer=init)

    with open(outputfile,"w") as f:
        for batch_doc in tqdm.tqdm(batches,desc="tokenizing documents multi-progress"):

            for record in workers.map(tokenize,batch_doc):
                if record is not None:

                    cache.append(record+"\n")

            f.writelines(cache)
            cache.clear()

        workers.close()
        workers.join()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--tokenizer",type=str,default="bert", choices=["bert", "roberta", "gpt2","xlnet"])
    parser.add_argument("--file",type=str,default="")
    parser.add_argument("--workers",type=int,default=30)
    parser.add_argument("--batch_size",type=int,default=1000)

    args=parser.parse_args()

    if args.tokenizer not in path_map_tokenizers:
        args.tokenizer="bert"

    inputfile=args.file
    outputfile=inputfile.replace(".txt",".tokenized-"+args.tokenizer)

    with open(inputfile, "r", encoding="utf-8") as f:
        docs=f.readlines()
        doc_data=filter(lambda s: s and s.strip(), docs)
        doc_data=list(doc_data)
        logger.info("loaded {}/{} lines of text".format(len(doc_data), len(docs)))
    tokenizeParallel(doc_data)

    '''s="You can use [NUM] and [CODE] to finish your work."
    init()
    print(tokenizer.tokenizer.vocab_size, len(tokenizer.tokenizer))
    print(tokenizer.tokenizer.tokenize(s))
    s_t=tokenize(s)
    print(s)
    print(s_t)'''
    
