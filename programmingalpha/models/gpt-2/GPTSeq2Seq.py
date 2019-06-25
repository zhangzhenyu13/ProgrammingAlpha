from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import texar as tx
from texar.modules.decoders.transformer_decoders import TransformerDecoder

from utils import model_utils, processor

# pylint: disable=invalid-name, too-many-locals, too-many-statements, no-member
# pylint: disable=too-many-branches

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", "gpt2_pretrained_models/117M/model.ckpt",
                    "Model checkpoint to load model weights from.")
flags.DEFINE_string("config_gpt2", "117M",
                    "The of the GPT-2 config file to use.")
flags.DEFINE_integer("seed", None, "Random seed.")
flags.DEFINE_integer("nsamples", 1, "The number of samples per input.")
flags.DEFINE_integer("batch_size", 1, "The batch size of input.")
flags.DEFINE_integer("max_decoding_length", 100,
                     "The maximun length of generated text.")
flags.DEFINE_float("temperature", 0.7,
                   "Softmax temperature for top-k sample decoding. Must be "
                   "strictly greater than 0. Defaults to 0.7.")
flags.DEFINE_integer("top_k", 40,
                     "The number of top most likely candidates from a vocab "
                     "distribution.")
flags.DEFINE_boolean("is_interactive", False, "Interactive mode or not.")
flags.DEFINE_string("config_format", "json",
                    "The configuration file format. Set to 'json' if the GPT-2 "
                    "config file is in the same format of the official GPT-2 "
                    "config file. Set to 'texar' if GPT-2 config file is in "
                    "Texar format.")


class ContextGeneration(object):
    def __init__(self):
        self.sess=None
        self.proc=None
        self.max_batch_size = 32
        self.max_decoding_length = 300
        self.max_sequence_length = 300

        self.loadModel()

    def loadModel(self,modelPath="gpt2_pretrained_models/117M/"):
        """
        Builds the model and runs
        """

        np.random.seed(123)
        tf.set_random_seed(789)

        ckpt_path = modelPath+"model.ckpt"

        gpt2_config_dir = modelPath
        # Load GPT-2 model configuration
        gpt2_config = model_utils.transform_gpt2_to_texar_config(
            os.path.join(gpt2_config_dir, 'hparams.json'))

        assert self.max_decoding_length <= gpt2_config.decoder["position_size"], (
            "max_decoding_length should be smaller than position size")

        # Create a data pre-processor for, e.g., BPE encoding
        self.proc = processor.get_encoder(gpt2_config_dir)

        # Build the GPT-2 modle
        embedder = tx.modules.WordEmbedder(
            vocab_size=gpt2_config.vocab_size,
            hparams=gpt2_config.embed)


        decoder = TransformerDecoder(
            embedding=embedder.embedding, hparams=gpt2_config.decoder)

        self.context = tf.placeholder(tf.int32, [self.max_batch_size, None])
        self.context_length = tf.placeholder(tf.int32, [self.max_batch_size])

        end_token = self.proc.encoder['<|endoftext|>']

        start_tokens = tf.fill([self.max_batch_size], end_token)

        helper = tx.modules.TopKSampleEmbeddingHelper(
            embedding=embedder,
            start_tokens=start_tokens,
            end_token=end_token,
            top_k=40,
            softmax_temperature=0.7)

        # Generate continuations of context
        self.lm_output, _ = decoder(
            context=self.context,
            context_sequence_length=self.context_length,
            max_decoding_length=self.max_decoding_length,
            helper=helper,
            mode=tf.estimator.ModeKeys.PREDICT)

        self.sess= tf.Session()

        # Load model checkpoint
        model_utils.init_gpt2_checkpoint(self.sess, ckpt_path)
        print("\nFinished loading\n")


    def generateText(self,batch):

        batch_size=len(batch)
        assert batch_size <= self.max_batch_size, (
            "batch_size is larger than max allowed batch size, please feed in several batch")

        context_tokens_padded = self.padEncoding(batch)
        print(context_tokens_padded.shape)

        feed_dict = {
            self.context: context_tokens_padded,
            self.context_length:
                np.repeat(context_tokens_padded.shape[1],self.max_batch_size),
            tx.context.global_mode():tf.estimator.ModeKeys.PREDICT
        }

        generated = 0

        output = self.sess.run(self.lm_output, feed_dict=feed_dict)

        sample_id = output.sample_id
        for i in range(batch_size):

            generated += 1
            print("=" * 40 +
                  " SAMPLE " + str(generated) + " " + "=" * 40)
            si = sample_id[i][context_tokens_padded.shape[1]:]
            print(self.proc.decode(si))

        print("=" * 80)

    def __del__(self):
        self.sess.close()

    def padEncoding(self,texts):
        encoded = [self.proc.encode(raw_text) for raw_text in texts]
        maxLen=max(map(lambda seq:len(seq),encoded))
        seqLen=min(self.max_sequence_length,maxLen)

        data=np.zeros(shape=(self.max_batch_size,seqLen),dtype=int)

        for i in range(len(texts)):
            seq=encoded[i]
            if len(seq)==seqLen:
                data[i]=seq
                continue
            for j in range(min(len(seq),seqLen)):
                data[i][j]=seq[j]
        data=np.abs(data)
        return data

if __name__ == '__main__':
    genGPT2=ContextGeneration()
    batch=["If I understand your question correctly , you can use and regex for this : Example : Output : DEMO",
           "Mark Seemann 's answer gave me the idea for this variant : This is meant to illustrate that instead of letting the Web project reference the DAL it references a separate Composition Root-project that references both DAL and BLL . The composition-root-project has a single class with one method that define the bindings . It gives these additional benefits : I do n't see any big drawbacks .",
           "is an asynchronous operation , so it does n't take effect immediately : enqueues changes to the component state and tells React that this component and its children need to be re-rendered with the updated state [ ... ] Think of as a request rather than an immediate command to update the component . For better perceived performance , React may delay it , and then update several components in a single pass . React does not guarantee that the state changes are applied immediately . does not always immediately update the component . It may batch or defer the update until later . This makes reading right after calling a potential pitfall . Instead , use or a callback [ ... ] , either of which are guaranteed to fire after the update has been applied . Here 's an example of a callback in your context :",
           "The plain HTML way is to put it in a wherein you specify the desired target URL in the attribute . If necessary , set CSS on the form to keep it in the flow with the surrounding text . Instead of in above example , you can also use . The only difference is that the element allows children . You 'd intuitively expect to be able to use analogous with the element , but unfortunately no , this attribute does not exist according to HTML specification . If CSS is allowed , simply use an which you style to look like a button using among others the property ( only Internet Explorer support is currently ( July 2015 ) still poor ) . Or pick one of those many CSS libraries like Bootstrap . If JavaScript is allowed , set the ."
        ]

    genGPT2.generateText(batch)
