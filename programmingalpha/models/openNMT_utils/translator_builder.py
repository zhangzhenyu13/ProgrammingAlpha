""" Translator Class and builder """
from __future__ import print_function

#import onmt.model_builder
from programmingalpha.models.openNMT_utils import model_builder
import codecs
from onmt.translate.translator import Translator
import onmt

def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    load_test_model = model_builder.load_test_model_ensemble \
        if len(opt.models) > 1 else model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    translator = Translator.from_opt(
        model,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        out_file=out_file,
        report_score=report_score,
        logger=logger
    )
    return translator
