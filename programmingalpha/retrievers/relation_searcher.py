from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import random

import programmingalpha
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from programmingalpha.tokenizers import BertTokenizer
from programmingalpha.models.InferenceModels import LinkNet, KnowNet
from pytorch_pretrained_bert import BertConfig
from tqdm import tqdm
from multiprocessing import Pool as ProcessPool
from copy import deepcopy

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self,sourcePair=""):
        self.sourcePair=sourcePair

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a json file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines=[]
            for line in f.readlines():
                lines.append(json.loads(line))
            return lines


class SemanticPairProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        if self.sourcePair is None or self.sourcePair=="":
            data_file="train.json"
        else:
            data_file="train-"+self.sourcePair+".json"

        logger.info("LOOKING AT {}".format(os.path.join(data_dir, data_file)))
        return self._create_examples(
            self._read_json(os.path.join(data_dir, data_file)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if self.sourcePair is None or self.sourcePair=="":
            data_file="test.json"
        else:
            data_file="test-"+self.sourcePair+".json"


        return self._create_examples(
            self._read_json(os.path.join(data_dir, data_file)), "dev")

    def get_labels(self):
        """See base class."""
        return ["duplicate","direct","transitive","unrelated"]



    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []



        for (i, record) in enumerate(lines):

            guid = "%s-%s" % (set_type, i)

            text_a = record["q1"]
            text_b = record["q2"]
            label=record["label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )

        return examples

class QuestionPostProcessor(SemanticPairProcessor):


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []

        for (i, record) in enumerate(lines):

            guid = "%s-%s" % (set_type, i)

            text_a = record["question"]
            text_b = record["post"]
            label=record["label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )

        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convertCore(example:InputExample):

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    feature = InputFeatures(input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id)
    return feature

def worker_initializer(label_map1,tokenizer1,max_seq_length1,copy=True):
    global label_map,tokenizer,max_seq_length
    if copy:
        label_map=deepcopy(label_map1)
        tokenizer=deepcopy(tokenizer1)
        max_seq_length=deepcopy(max_seq_length1)
    else:
        label_map=label_map1
        tokenizer=tokenizer1
        max_seq_length=max_seq_length1


def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer,verbose=2,worker_num=8):
    """Loads a data file into a list of `InputBatch`s."""

    features = []


    if worker_num>1:
        workers=ProcessPool(processes=worker_num,initializer=worker_initializer,initargs=(label_map,tokenizer,max_seq_length))

        batch_size=1000 #configurable variable
        batch_num=len(examples)//batch_size
        if batch_num*batch_size<len(examples):
            batch_num+=1
        batches=[examples[i:i+batch_size] for i in range(0,len(examples),batch_size)]

        for batch in tqdm(batches,desc="convertingBatch"):
            results=workers.map(convertCore,batch)
            features.extend(results)

            #if verbose>0 and len(features)%batch_size==0:
            #    logger.info("loaded {} features".format(len(features)))

        workers.close()
        workers.join()

    else:
        for example in tqdm(examples,desc="converting examples"):
            feature=convertCore(example)
            features.append(feature)

    logger.info("loaded {} features".format(len(features)))

    if verbose>1:
        for ex_index in range(min(5,len(examples))):
            example=examples[ex_index]
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]

            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in features[ex_index].input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in features[ex_index].input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in features[ex_index].segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, features[ex_index].label_id))


    return features


class LinkSearcher(object):

    label_map = {"duplicate":0,"direct":1,"transitive":2,"unrelated":3}
    __default_label="unrelated"

    def initRunningConfig(self, model:KnowNet):

        if self.local_rank == -1 or self.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(self.local_rank != -1), self.fp16))

        self.device=device

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

        if self.fp16:
            model.half()
        model.to(device)
        if self.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        return model

    def __init__(self,model_dir,model_name, no_cuda=False, local_rank=-1, fp16=False, do_lower_case=True, batch_size=8, max_seq_len=1500):
        # running paprameters

        self.no_cuda=no_cuda
        self.local_rank = local_rank
        self.fp16 = fp16
        self.seed = 42

        ## model parameters
        self.do_lower_case = do_lower_case
        self.batch_size = batch_size
        self.max_seq_length=max_seq_len

        buildNet=KnowNet if model_name=='KnowNet' else LinkNet

        model = buildNet(BertConfig(os.path.join(model_dir, model_name + ".json")), num_labels=4)
        logger.info("loading weghts for {}".format(model_name))
        if self.no_cuda:
            map_location='cpu'
        else:
            map_location='cuda'
        model_state_dict = torch.load(os.path.join(model_dir,model_name+".bin"),map_location=map_location)
        model.load_state_dict(model_state_dict)
        self.model=self.initRunningConfig(model)

        self.tokenizer=BertTokenizer.from_pretrained(programmingalpha.BertBasePath, do_lower_case=self.do_lower_case)
        logger.info("ranker model init finished!!!")

    def __getSemanticPair(self,query_doc,docs,doc_ids):
        examples=[]
        for (i,doc) in enumerate(docs):
            guid = "%s-%s" % (doc_ids[i], i)
            text_a = query_doc
            text_b = doc
            label=self.__default_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )

        return examples

    def relationPredict(self,query_doc,docs,k=5):
        doc_texts=[f["text"] for f in docs]
        doc_ids=[f["Id"] for f in docs]
        #doc_ids=np.array(doc_ids, dtype=np.str).reshape((-1,1))

        eval_examples=self.__getSemanticPair(query_doc,doc_texts,doc_ids)

        eval_features = convert_examples_to_features(
            eval_examples, self.label_map, self.max_seq_length, self.tokenizer,0,worker_num=10)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

        logits=self.getRelationProbability(eval_dataloader)
        #print("logits is {}".format(logits))

        dprobs=list(map(lambda doc_id, dp:{"Id": doc_id, "dist": int(dp[0]), "prob": float(dp[1]) }, doc_ids, logits ) )
        #print(dprobs)

        dprobs.sort(key=lambda x: x["prob"], reverse=True)
        dprobs.sort(key=lambda x: x["dist"], reverse=False)
        results=dprobs

        #print(results)

        
        return results[:k]

    def getRelationProbability(self,eval_dataloader:DataLoader):
        self.model.eval()
        device=self.device
        logits=[]

        for input_ids, input_mask, segment_ids in tqdm(eval_dataloader,desc="computing simValue"):
            #logger.info("batch predicting")
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                b_logits = self.model(input_ids, segment_ids, input_mask)
                #print("b_logits", b_logits)

            b_logits = b_logits.detach().cpu().numpy().tolist()

            logits.extend(b_logits)
        
        #print("logits original", logits)
        logits=map(lambda x:(np.argmax(x),np.max(x)), logits )
        logits=list(logits)
        #print("list logits", logits)
        #logits=np.concatenate(logits,axis=0)

        return logits


#recommend top K useful posts
def recommendTopKPosts(knowAlpha:LinkSearcher, questionTxt, postTxts):
    j=random.randint(0, len(postTxts)-1)
    #print(questionTxt)
    #print(postTxts)
    #print(j)
    #print(postTxts[j])
    
    assert "text" in postTxts[j], ("KeyError: text not in post")
    assert "Id" in postTxts[j], ("KeyError: Id not in post")

    return knowAlpha.relationPredict(questionTxt,postTxts)

from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy
class KnowAlphaHTTPProxy(AlphaHTTPProxy):
    def __init__(self, config_file):
        AlphaHTTPProxy.__init__(self,config_file)
        args=self.args
        self.alphaModel=LinkSearcher(args.model_dir, args.model_name,
                                     no_cuda=args.no_cuda, local_rank=args.local_rank, fp16=args.fp16,
                                     do_lower_case=args.do_lower_case, batch_size=args.batch_size, max_seq_len=args.max_seq_len)
    def processCore(self, data):
        return recommendTopKPosts(self.alphaModel, data["question"], data["posts"])

