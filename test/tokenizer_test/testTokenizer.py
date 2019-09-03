from programmingalpha.tokenizers import  get_tokenizer

from programmingalpha import AlphaPathLookUp

def testBasicFunctions():
    tokenizer=get_tokenizer(name="gpt2", model_path=AlphaPathLookUp.GPT2Base)
    #tokenizer=get_tokenizer(name="roberta", model_path=programmingalpha.RoBertaBase)
    #tokenizer=get_tokenizer(name="bert", model_path=AlphaPathLookUp.BertBaseUnCased)
    
    print(tokenizer.tokenizer.additional_special_tokens)
    print(tokenizer.tokenizer.added_tokens_encoder)
    #exit(10)

    s="I am fantastic [CODE] supreme [MATH] !"
    print(tokenizer.tokenize(s))
    s_ids=tokenizer.tokenizeLine(s)
    print(s)
    print(s_ids)

    for id in s_ids.split():
        print(tokenizer.decode([id]))

def testVocab():
        
        tokenizer=get_tokenizer(name="bert", model_path=AlphaPathLookUp.BertBaseUnCased)
        ids=[0,1,2,3,4,5,6,7,8,9,10]
        print(tokenizer.decode(ids))
        tokenizer=get_tokenizer(name="xlnet", model_path=AlphaPathLookUp.XLNetBaseCased)
        print(tokenizer.decode(ids) )

if __name__ == "__main__":
        testBasicFunctions()
        #testVocab()
