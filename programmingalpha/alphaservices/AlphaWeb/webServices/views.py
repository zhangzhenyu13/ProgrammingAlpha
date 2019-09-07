from django.shortcuts import render
from django.http import HttpResponse,HttpRequest
from .qamodel import process
from html import escape
import json
from textblob import TextBlob
from programmingalpha.Utility import getLogger
logger=getLogger(__name__)
#def process(q):return "No"
# Create your views here.

def printUserIP(request):
    if 'HTTP_X_FORWARDED_FOR' in request.META:
        ip =  request.META['HTTP_X_FORWARDED_FOR']
    else:
        ip = request.META['REMOTE_ADDR']
    logger.info("get a request from {}".format(ip))

def getAnswer(request:HttpRequest):
    request.encoding='utf-8'
    data=request.body
    #print(type(data),"********requesting post is--->", data)
    data=json.loads(data)
    
    printUserIP(request)


    posts=['not available']*3
    posts.append("...")
    posts.append("not available")

    if 'Title' in data and data["Title"].strip():
        question={
            "Title": data.get("Title"),
            "Body": data.get("Body"),
            "Tags": data.get("Tags")
        }
        if question["Tags"].strip():
            question["Tags"]=list(map(lambda tag: " ".join(tag.strip().split()).replace(' ','-') ,question["Tags"].split(",")) )
        else:
            question["Tags"]=[]
        body=question["Body"]
        sents=[]
        for sent in TextBlob(body).sentences:
            sents.append(
                "<p>"+sent.string+"</p>"
            )
        question["Body"]="".join(sents)
        #print("",question)
        res=process(question )
        posts=list(map(lambda post:post["Title"], res["useful-reading-posts"]))

        answer=res["generated-answers"]["tgt_txt"]
    else:
        answer = 'cannot answer blank questions!!!'
    
    reply ={}
    reply['answer'] = answer
    reply['posts']=posts

    res = HttpResponse(json.dumps(reply))
    res['Access-Control-Allow-Origin'] = '*'
    return res


def index(request):
    printUserIP(request)

    return render(request, 'alpha-QA.html')