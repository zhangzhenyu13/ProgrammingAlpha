import json
from programmingalpha import AlphaPathLookUp
import argparse
import tqdm

def _get_question_from_post(post):
    question={
        "Title":post["Title"],
        "Body":post["Body"]
    }
    return question

def _get_post_from_post(post):
    data={
        "Title":post["Title"],
        "Body":post["Body"],
        "answers":[]
    }

    for answer in post["answers"]:
        ans={
            "Body":answer["Body"]
        }
        data["answers"].append(ans)
    
    return data

def extract_question_post(record):
    if not record:
        return None

    post1=record["post1"]
    post2=record["post2"]
    
    data=[]

    label=record["label"]

    #q1-p2
    if len(post2["answers"])>0:
        answers=post2["answers"]
        acc_ans_id=-1
        if "AcceptedAnswerId" in post2:
            acc_ans_id=post2["AcceptedAnswerId"]
        if acc_ans_id!=-1:
            ans=answers[-1]
            del answers[-1]
            answers.insert(0, ans)
        data.append(
            {
                "question": _get_question_from_post(post1),
                "post": _get_post_from_post(post2),
                "label":label
            }
        )
    
    #q2-p1
    if len(post1["answers"])>0:
        answers=post1["answers"]
        acc_ans_id=-1
        if "AcceptedAnswerId" in post1:
            acc_ans_id=post1["AcceptedAnswerId"]
        if acc_ans_id!=-1:
            ans=answers[-1]
            del answers[-1]
            answers.insert(0, ans)

        data.append(
            {
                "question": _get_question_from_post(post2),
                "post": _get_post_from_post(post1),
                "label":label
            }
        )

        return data

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--file', type=str,  required=True)

    parser.add_argument('--workers', type=int, default=10)

    args = parser.parse_args()

    

    with open(args.file, "r", encoding="utf-8") as f:
        records=map(lambda line: json.loads(line.strip()), f)
        data=[]
        
        for record in tqdm.tqdm(records, desc="loading data"):
            res=extract_question_post(record)
            if res:
                data.extend(res)            

    with open(args.file.replace(".json","-extracted.json"), "w", encoding="utf-8") as f:
        f.writelines(
            map(lambda record: json.dumps(record)+"\n", data)
        )
