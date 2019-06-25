from programmingalpha.DataSet.DBLoader import MongoStackExchange
from tqdm import tqdm
import pickle
import numpy as np
from matplotlib import pyplot

docDB=MongoStackExchange(host='10.1.1.9',port=50000)

def countAnswerLen():
    docDB.useDB('corpus')
    seq2seq=docDB.stackdb['seq2seq']
    answerLength=[]
    retrictedLen=[]
    for record in tqdm(seq2seq.find().batch_size(10000),desc="retrieving seq2seq record"):
        ansL=len(" ".join(record["answer"]).split())
        answerLength.append(ansL)
        if ansL<=200:
            retrictedLen.append(ansL)

    answerLength.sort()

    avg=np.mean(answerLength)
    std=np.std(answerLength)
    hist=np.histogram(answerLength)

    print(avg,std,len(answerLength),len(retrictedLen))
    print(hist)
    x=np.arange(len(answerLength))



    pyplot.plot(x,answerLength)
    pyplot.show()



countAnswerLen()
