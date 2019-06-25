from programmingalpha.DataSet.DBLoader import MongoStackExchange
from programmingalpha.Utility.TextPreprocessing import PreprocessPostContent
def readQueryId(file=""):
    qids=[]
    with open(file,"r") as f:
        for line in f:
            if "query Id :" in line:
                qids.append(int(line.strip()[11:]))
    print(qids)

    return qids

def readSummary(file=""):
    summaries=[]
    tag=False
    with open(file,"r") as f:
        for line in f:
            if "----Summary----" in line:
                tag=True
                continue
            if tag==False:
                continue

            summaries.append(line)
    summaries.pop()

    return " ".join(summaries)

def genResults():
    Qids=readQueryId("../../dataCases/query_list.txt")
    Summy={}
    i=0
    for i in range(100):
        Summy[Qids[i]]=readSummary("../../dataCases/Summary_list/%d.txt"%i)

    print(len(Qids),len(Summy))
    print(Summy)

    Answers={}
    processor=PreprocessPostContent()
    docDB=MongoStackExchange(host="10.1.1.9",port=50000)
    docDB.useDB("stackoverflow")
    for qid in Qids:
        question=docDB.questions.find_one({"Id":qid})
        if not question:
            print("None Error",qid,question)
            continue
        #print(question)
        if "AcceptedAnswerId" in question and question["AcceptedAnswerId"]:
            ans=docDB.answers.find_one({"Id":question["AcceptedAnswerId"]})["Body"]
        else:
            answers=docDB.answers.find({"ParentId":qid})
            answers=list(answers)
            if len(answers)<1:
                print("Error!",qid)
                continue
            answers.sort(key=lambda x:x["Score"],reverse=True)
            ans=answers[0]["Body"]

        ans=processor.getPlainTxt(ans)
        ans=" ".join(ans)
        Answers[qid]={"true":ans,"generated":Summy[qid]}
        print(len(Answers),Answers[qid])
        #break

    with open("../../dataCases/answers.json","w") as f:
        import json
        json.dump(Answers,f)

def splitRefGen():
    with open("../../dataCases/answers.json","r") as f:
        import json
        Answers=json.load(f)
    with open("../../dataCases/refs.txt","w") as f1, open("../../dataCases/sums-50.txt","w") as f2,\
            open("../../dataCases/sums-100.txt","w") as f3,open("../../dataCases/sums-150.txt","w") as f4,\
            open("../../dataCases/sums-200.txt","w") as f5:
        for k in Answers.keys():
            ref,sum=Answers[k]["true"],Answers[k]["generated"]
            ref=" ".join(ref.split())
            sum=sum.split()
            f1.write(ref.strip()+"\n")

            f2.write(" ".join(sum[:50])+"\n")
            f3.write(" ".join(sum[:100])+"\n")
            f4.write(" ".join(sum[:150])+"\n")
            f5.write(" ".join(sum[:200])+"\n")


if __name__ == '__main__':
    #genResults()
    splitRefGen()
