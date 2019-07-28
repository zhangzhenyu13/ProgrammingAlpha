import torch
import torch.nn as nn
import onmt.utils
from pytorch_transformers.modeling_bert import BertModel
import programmingalpha
import logging
import numpy as np
import math
from pytorch_transformers.modeling_bert import BertEmbeddings

from copy import deepcopy

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class BertEmbeddingAdapted(nn.Module):

    def __init__(self, embeddings:BertEmbeddings, padding_idx, encoder_side=False):
        super(BertEmbeddingAdapted, self).__init__()
        self.word_padding_idx=padding_idx
        self.embeddings=embeddings
        self.enc_side=encoder_side

    def forward(self, input_ids, step=None, token_type_ids=None, position_ids=None ):
        '''
        :param input_ids: (len*b*feat) --> same for token_type_ids, position_ids
        :param step:
        :return: (len*b*dim)
        '''
        #print("inputs are:", input_ids, step, token_type_ids, position_ids)
        #print("input",input_ids.size())
        input_ids=input_ids.squeeze(2).transpose(0,1)
        #print("emb input",input_ids.size())

        if token_type_ids is not None and len(token_type_ids.size())>2:
            token_type_ids=token_type_ids.squeeze(2).transpose(0,1)
        if position_ids is not None and len(position_ids.size())>2:
            position_ids=position_ids.squeeze(2).tanspose(0,1)

        embeddings= self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids, position_ids =position_ids)

        if self.enc_side==False:
            embeddings=embeddings.transpose(0,1)
        #print("emb output", embeddings.size())

        return embeddings

class OnmtBertTransformerEncoder(BertModel):

    def setEmbeddings(self,emb):
        self.embeddings=emb

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask=attention_mask.squeeze(2).transpose(0,1)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        outputs = BertModel.forward(self,input_ids=input_ids,
            token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=position_ids,
                                                                                  head_mask=head_mask)

        enc_state, memory_out = outputs[1], outputs[0]
        enc_state= enc_state.transpose(0,1)
        memory_out =memory_out.transpose(0,1)
        #print("encs state and memory out are",enc_state.size(), memory_out.size())
        # return embedding_output.transpose(0,1),encoded_layers.transpose(0,1),lengths

        return enc_state, memory_out

class TransformerModel(nn.Module):
    def __init__(self, encoder:OnmtBertTransformerEncoder, decoder:onmt.decoders.TransformerDecoder):
        nn.Module.__init__(self)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        #print("lengths, src and tgt size is ",lengths.size(),src.size(),tgt.size())
        #src_enc=src.squeeze(2)

        tgt = tgt[:-1]  # exclude last target from inputs
        #src_enc=src.squeeze(2).transpose(0,1)

        enc_state, memory_bank = self.encoder(input_ids=src)
        #print("size of encoded layers",memory_bank.size())

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)

        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        #print("decoded result",dec_out.size(),attns["std"].size())
        return dec_out, attns


class TextGeneratorModel(object):

    opt=None
    model_opt=None
    fields=None
    gpu_id=None
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
        drop_rate=self.model_opt.dropout #if hasattr(self.model_opt,"dropout") else 0
        layers=self.model_opt.layers #if hasattr(self.model_opt,"layers") else 4
        #tgt embeddings

        encEmbeddings=BertEmbeddingAdapted(embeddings=bert.embeddings, padding_idx=pad_idx, encoder_side=True)
        decEmbeddings=BertEmbeddingAdapted(embeddings=bert.embeddings, padding_idx=pad_idx, encoder_side=False)
        bert.setEmbeddings(encEmbeddings)


        #decoder
        transformerDecoder=onmt.decoders.TransformerDecoder(
            num_layers=layers, d_model=model_dim, heads=heads, d_ff=feed_forwad_size,
                         copy_attn=True, self_attn_type="scaled-dot", dropout=drop_rate, embeddings=decEmbeddings,
                         max_relative_positions=max_relative_position
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
        exlude_dict=("encoder.embeddings.token_type_embeddings.weight","encoder.embeddings.position_embeddings.weight",
                     "decoder.embeddings.position_embeddings.weight")
        restore_dict={}

        layers= self.model_opt.layers #if hasattr(self.model_opt,"layers") else 4
        logger.info("decoder layer num: %d"% layers)

        #self.transformer.load_state_dict(weight_dict)
        for k,v in weight_dict.items():
            if k in exlude_dict:
                logger.info("skip :{}".format(k))
                continue
            restore_dict[k]=v
        transformer_dict.update(restore_dict)


        if model_file:
            logger.info("init model weight with "+model_file)
        else:
            logger.info("init model with checkpoint")
