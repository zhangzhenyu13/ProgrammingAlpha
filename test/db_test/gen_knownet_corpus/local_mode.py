import argparse
from copy import deepcopy
import json
from programmingalpha.retrievers.retriever_input_process import FeatureProcessor
from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from programmingalpha.Utility import getLogger
import os

logger = getLogger(__name__)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--file', type=str,  required=True)
    parser.add_argument('--folder', type=str,  required=True)

    args = parser.parse_args()

    processor=FeatureProcessor("knowAlphaService.json")
    

    if args.folder[-1]=="/":
        input_folder=args.folder[:-1]
    else:
        input_folder=args.folder

    input_file=os.path.join(input_folder, args.file)
    output_folder=input_folder+"-features"
    os.makedirs(output_folder, exist_ok=True)
    output_file=os.path.join(output_folder, args.file)

    logger.info("input: {}, {},".format(input_folder, input_file))
    logger.info("output:{}, {}".format(output_folder, output_file))

    features=[]
    with open(input_file,"r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as f2:
        records=map(lambda line: json.loads(line.strip()), f)
        features=map(lambda record: processor.batch_process_core(record), records)
        lines=map(lambda feature: feature.dumps()+"\n", features)
        f2.writelines(lines)
    
    logger.info("finished processing :{} --> {}".format(input_file, output_file))
