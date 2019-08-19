import argparse
import json
import random 
parser=argparse.ArgumentParser()

parser.add_argument("--file", type=str, required=True)

args=parser.parse_args()
input_file=args.file
src_out=input_file.replace(".json", "-src.txt")
tgt_out=input_file.replace(".json", "-tgt.txt")

with open(input_file, "r", encoding="utf-8") as f:
    docs=list(map(lambda line: json.loads(line.strip()), f))
    random.shuffle(docs)
    srcs= map(lambda doc: doc["src"]+"\n", docs)
    tgts=map(lambda doc: doc["tgt"]+"\n", docs)
    with open(src_out, "w", encoding="utf-8") as f1:
        f1.writelines(srcs)
    with open(tgt_out, "w", encoding="utf-8") as f2:
        f2.writelines(tgts)
