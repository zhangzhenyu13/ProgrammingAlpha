from django.shortcuts import render
from django.http import HttpResponse,HttpRequest
from .qamodel import process
from html import escape
import json
#def process(q):return "No"
# Create your views here.


def getAnswer(request:HttpRequest):
    request.encoding='utf-8'
    data=request.body
    #print(type(data),"********requesting post is--->", data)
    data=json.loads(data)

    posts=['not available']*3
    posts.append("...")
    posts.append("not available")

    if 'Title' in data and data["Title"].strip():
        question={
            "Title": data.get("Title"),
            "Body": data.get("Body"),
            "Tags": data.get("Tags")
        }
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

    return render(request, 'alpha-QA.html')