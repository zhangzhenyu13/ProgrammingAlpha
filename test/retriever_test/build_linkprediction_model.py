from programmingalpha.retrievers.retriever_input_process import  load_features_from_file                                                        
import random
import argparse
from tqdm import tqdm
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torch, numpy as np
import os
from programmingalpha.Utility import getLogger
from programmingalpha.models.InferenceNets import get_inference_net
from programmingalpha import AlphaPathLookUp

logger = getLogger(__name__)


model_path={
    "bert": AlphaPathLookUp.BertBaseUnCased,
    "xlnet": AlphaPathLookUp.XLNetBaseCased
}

num_labels=4
train_step=0
seed=43
#device settings
if torch.cuda.is_available()==False:
        device=torch.device("cpu")
        n_gpu=0
else:
    n_gpu=torch.cuda.device_count()
    if n_gpu<2:
        device=torch.device("cuda:0")
    else:
        device=torch.device("cuda")
        torch.cuda.manual_seed_all(seed)

#random init
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
def mseError(out,values):
    return np.mean(np.square(out-values))

def saveModel(model):
    save_model_file = os.path.join(args.save_dir, "model_{}.bin".format(train_step))

    logger.info("saving model:{}".format(save_model_file))
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save.state_dict())
    

def main():

    if os.path.exists(args.save_dir) and os.listdir(args.save_dir) and args.do_train:
        if args.overwrite==False and input("Save directory ({}) already exists and is not empty, rewrite the files?(Y/N)\n".format(args.output_dir)) not in ("Y","y"):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.save_dir))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)


    # train and eval
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    logger.info("gradient_accumulation_steps {}".format(args.gradient_accumulation_steps))

    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    #preppare train data
    if os.path.exists(args.data_dir)==False:
        raise ValueError("data dir does not exists")

    #load train data
    train_features = load_features_from_file(os.path.join(args.data_dir, "train.json"))
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)

    #load valid data
    eval_features = load_features_from_file(os.path.join(args.data_dir, "valid.json"))
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)


    #configure model running parameters
    t_total = args.max_steps
    num_train_epochs = args.max_steps // (len(train_features) // args.gradient_accumulation_steps) + 1
    
    # Prepare model
    model =get_inference_net( model_path=model_path[args.encoder], name=args.encoder)
    logger.info("{}".format(model))
    model.to(device)
    if n_gpu>1:
        from torch import nn
        model = nn.DataParallel(model)

    # Prepare optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)


        

    def _eval_epoch(model):
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)


        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        i=0
        for input_ids, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, label_ids)

                logits = model(input_ids, segment_ids)

            i+=args.eval_batch_size

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)


            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1


        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        return eval_accuracy, eval_loss

    def _train_epoch(epoch_num):
        saved_at_least_one=False

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", t_total)

        best_eval_acc=0.0

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        model.train()
        for _ in range(num_train_epochs,):
            logger.info("Epoch_num:{}/{}".format(_,num_train_epochs))

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, label_ids = batch
                input_ids=input_ids.to(device)
                segment_ids=segment_ids.to(device)
                label_ids=label_ids.to(device)

                loss = model(input_ids, segment_ids, label_ids)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    train_step += 1

                # save Model
                if (1+step)%args.eval_step_size==0:
                    eval_acc,eval_loss=_eval_epoch(model)
                    if eval_acc>best_eval_acc:
                        saveModel(model)
                        best_eval_acc=eval_acc
                        saved_at_least_one=True


                    logger.info("training step:{}, eval_accuracy:{}, eval_loss:{}, best eval_acc:{}".format(
                        train_step, eval_acc, eval_loss, best_eval_acc)
                    )

        if saved_at_least_one==False:
            saveModel(model)

    #begin training model
    _train_epoch(int(args.num_train_epochs))
   

if __name__ == "__main__":
    dataSource=""

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--encoder",
                        default="bert",
                        type=str,
                        choice=["bert", "bert_attn","xlnet"],
                        help="The name of the pretrained model.")

    parser.add_argument("--data_dir",
                        required=True,
                        type=str,
                        help="The input data dir. containing train and valid data")

    
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="overwrite saved model folder")

    parser.add_argument("--save_dir", required=True, type=str,
                        help="dir to save the trained model")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--train_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_step_size",
                        default=5000,
                        type=int,
                        help="eval model performance after several steps")
    parser.add_argument("--eval_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for eval.")
    
    #optimizer parameters
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    
    
    args = parser.parse_args()
    print(args)
    main()
