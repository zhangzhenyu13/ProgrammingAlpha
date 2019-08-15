from programmingalpha.tokenizers import get_tokenizer
import programmingalpha
import argparse
import os
import multiprocessing
import tqdm

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
    tokenized_text=tokenizer.tokenizeLine(text, add_sp=False)

    return " ".join(tokenized_text)

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
    outputfile=inputfile+"-tokenized-"+args.tokenizer

    with open(inputfile, "r", encoding="utf-8") as f:
        doc_data=f.readlines()
    tokenizeParallel(doc_data)

    
