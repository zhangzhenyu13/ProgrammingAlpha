# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


LANGUAGE = "english"
SENTENCES_COUNT = 10


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Automatic_summarization"
    html='''
        <p>Well, I do not know what type of features you are giving to your neural network. However, in general, I would go with a single neural network. It seems that you have no limitation in resources for training your network and the only problem is resources while you apply your network. </p>
        <p>The thing is that probably the two problems have things in common (e.g. both types of plates are rectangular). This means that if you use two networks, each has to solve the same sub-problem (the common part) again. If you use only one network the common part of the problem takes fewer cells/weights to be solved and the remaining weights/cells can be employed for better recognition.</p>
        <p>In the end, if I was in your place I would try both of them. I think that is the only way to be really sure what is the best solution. When speaking theoretically it is possible that we do not include some factors.</p>
    '''

    #parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    #parser=HtmlParser.from_string(html, tokenizer=Tokenizer(LANGUAGE), url=None )

    # or for plain text files
    from programmingalpha.Utility.TextPreprocessing import InformationAbstrator
    text_extractor=InformationAbstrator(100)
    text_extractor.initParagraphFilter(text_extractor.lexrankSummary)
    plain_text=" ".join( text_extractor.clipText(html) )

    parser = PlaintextParser.from_string(plain_text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)

