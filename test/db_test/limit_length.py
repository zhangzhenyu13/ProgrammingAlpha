import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--file', type=str, required=True)

parser.add_argument('--length', type=int, default=510)

args = parser.parse_args()

with open(args.file, "r", encoding="utf-8") as f:
    results=map(lambda line: " ".join(line.split()[:args.length])+"\n", f)

    with open(args.file+".{}".format(args.length), "w", encoding="utf-8") as f2:
        f2.writelines(results)