from programmingalpha.tokenizers import  get_tokenizer

import programmingalpha

def testBasicFunctions():
    tokenizer_gpt=get_tokenizer(name="gpt2", model_path=programmingalpha.GPT2Base)
    #tokenizer_roberta=get_tokenizer(name="roberta", model_path=programmingalpha.RoBertaBase)
    tokenizer=tokenizer_gpt
    print(tokenizer.tokenizer.additional_special_tokens)
    exit(10)

    s="I am fantastic [CODE] supreme [NUM]!"
    s_ids=tokenizer.tokenizeLine(s)
    print(s)
    print(s_ids)

    for id in s_ids.split():
        print(tokenizer.decode([id]))

if __name__ == "__main__":
    testBasicFunctions()

