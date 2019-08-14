import torch
import torch.nn as nn
import onmt.utils
import programmingalpha
import logging
import numpy as np
import math

from copy import deepcopy

from .BertGen import OnmtBertEncoder, buildBert
from .RoBertaGen import OnmtRobertaEncoder, buildRoberta
from .XLNetGen import OnmtXLNetEncoder, buildXLNet

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class TransformerModel(nn.Module):
    def __init__(self, encoder:onmt.encoders.EncoderBase, decoder:onmt.decoders.DecoderBase):
        nn.Module.__init__(self)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        #print("lengths, src and tgt size is ",lengths.size(),src.size(),tgt.size())
        #src_enc=src.squeeze(2)

        tgt = tgt[:-1]  # exclude last target from inputs
        #src_enc=src.squeeze(2).transpose(0,1)

        enc_state, memory_bank, src_lengths = self.encoder(input_ids=src, lengths=lengths)
        #print("size of encoded layers",memory_bank.size())

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)

        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=src_lengths)
        #print("decoded result",dec_out.size(),attns["std"].size())
        return dec_out, attns


class TextGeneratorModel(object):

    opt=None
    model_opt=None
    fields=None
    gpu_id=0
    device=None

    def __init__(self):
        self.build_model()

    def build_generator(self):
        model_opt=self.model_opt
        fields=self.fields

        from onmt.modules.util_class import Cast

        # Build Generator.
        if not model_opt.copy_attn:

            if model_opt.generator_function == "sparsemax":
                gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
            else:
                gen_func = nn.LogSoftmax(dim=-1)
            generator = nn.Sequential(
                nn.Linear(model_opt.dec_rnn_size,
                          len(fields["tgt"].base_field.vocab)),
                Cast(torch.float32),
                gen_func
            )
            if model_opt.share_decoder_embeddings:
                generator[0].weight = self.transformer.decoder.embeddings.word_embeddings.weight
        else:
            tgt_base_field = fields["tgt"].base_field
            vocab_size = len(tgt_base_field.vocab)
            pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
            generator = onmt.modules.CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)

        return generator

    def use_device(self):
        from onmt.utils.misc import use_gpu
        from torch import cuda
        gpu=use_gpu(self.opt) and cuda.is_available()
        gpu_id=self.gpu_id
        
        if gpu and gpu_id is not None:
            device = torch.device("cuda", gpu_id)
        elif gpu and not gpu_id:
            device = torch.device("cuda")
        elif not gpu:
            device = torch.device("cpu")

        self.device=device
        logger.info("gpu: {}, gpu_id: {}, device: {}".format(gpu, gpu_id, device))

        self.transformer.to(device)
        if self.model_opt.model_dtype == 'fp16':
            self.transformer.half()

        logger.info("device:{},half:{}".format(device,self.model_opt.model_dtype))

    def build_model(self):

        #encoder
        bert=OnmtBertTransformerEncoder.from_pretrained(programmingalpha.BertBasePath)

        def __copyEmbeddings(embeddings:nn.Embedding,index1,index2):
            #print(embeddings.weight.size())
            weight=embeddings.weight.detach().numpy()

            weight[index2]=deepcopy(weight[index1])

            weight=torch.tensor(weight,requires_grad=True)
            embeddings.weight=nn.Parameter(weight,requires_grad=True)

        __copyEmbeddings(bert.embeddings.word_embeddings,0,1)
        __copyEmbeddings(bert.embeddings.word_embeddings,100,0)


        tgt_base_field = self.fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        model_dim= self.model_opt.rnn_size #if hasattr(self.model_opt,"rnn_size") else 768
        max_seq_length=512
        max_relative_position=512
        heads=12
        feed_forwad_size=3072
        #drop_rate=self.model_opt.dropout if type(self.model_opt.drop_out) not in (list, tuple) else self.model_opt.drop_out[0]  #if hasattr(self.model_opt,"dropout") else 0
        layers=self.model_opt.layers #if hasattr(self.model_opt,"layers") else 4
        #tgt embeddings

        encEmbeddings=BertEmbeddingAdapted(embeddings=bert.embeddings, padding_idx=pad_idx, encoder_side=True)
        decEmbeddings=BertEmbeddingAdapted(embeddings=bert.embeddings, padding_idx=pad_idx, encoder_side=False)
        bert.setEmbeddings(encEmbeddings)


        #decoder
        #print(bert.config.__dict__)
        #print("bert.config.attention_probs_dropout_prob=",bert.config.attention_probs_dropout_prob, ", drop_out=", drop_rate)
        transformerDecoder=onmt.decoders.TransformerDecoder(
            num_layers=layers, d_model=model_dim, heads=heads, d_ff=feed_forwad_size,
                         copy_attn=True, self_attn_type="scaled-dot", dropout=bert.config.hidden_dropout_prob, 
                         embeddings=decEmbeddings,
                         max_relative_positions=max_relative_position, attention_dropout=bert.config.attention_probs_dropout_prob,
                         aan_useffn=True
        )

        self.transformer=TransformerModel(bert,transformerDecoder)

        self.transformer.generator=self.build_generator()

        self.use_device()

    def loadModel(self,model_file=None,checkpoint=None):
        assert model_file is not None or checkpoint is not None

        if checkpoint is None:
            # logger.info("loading from model file")
            model_dict=torch.load(model_file)
        else:
            # logger.info("loading from model checkpoint")
            model_dict=checkpoint

        weight_dict=model_dict["model"]
        generator_dict=model_dict["generator"]
        #print(weight_dict.keys())
        #print(generator_dict.keys())
        for k in generator_dict:
            weight_dict["generator."+k]=generator_dict[k]

        transformer_dict=self.transformer.state_dict()
        exlude_dict=tuple()#("encoder.embeddings.token_type_embeddings.weight","encoder.embeddings.position_embeddings.weight",
                     #"decoder.embeddings.position_embeddings.weight")
        restore_dict={}

        layers= self.model_opt.layers #if hasattr(self.model_opt,"layers") else 4
        logger.info("decoder layer num: %d"% layers)

        #self.transformer.load_state_dict(weight_dict)
        #logger.info("weights: {}".format(weight_dict.keys()))
        for k,v in weight_dict.items():
            if k in exlude_dict:
                logger.info("skip :{}".format(k))
                continue
            logger.info("loaded weight: {}".format(k) )
            restore_dict[k]=v
        transformer_dict.update(restore_dict)


        if model_file:
            logger.info("init model weight with "+model_file)
        else:
            logger.info("init model with checkpoint")
