from django.shortcuts import render
from django.http import HttpResponse,HttpRequest
from .qamodel import process
from html import escape
#def process(q):return "No"
# Create your views here.


def getAnswer(request:HttpRequest):
    request.encoding='utf-8'
    #print(request.POST)
    tag=False
    posts=['not available']*5
    if 'Title' in request.POST and request.POST["Title"].strip():
        question={
            "Title": request.POST.get("Title"),
            "Body": request.POST.get("Body"),
            "Tags": request.POST.get("Tags")
        }
        res=process(question )
        posts=list(map(lambda post:post["Title"], res["useful-reading-posts"]))

        answer=res["generated-answers"]["tgt_txt"]
        tag=True
    else:
        answer = 'cannot answer blank questions!!!'

    reply ={}
    if tag:
        reply['answer'] = answer
        for i in range(len(posts)):
            reply['post%d'%(i+1)]=posts[i]

    #print("request body=>",request.body)
    #print("ans is")
    #print(ans)
    return render(request, "alpha-QA.html", reply)

def index(request):

    return render(request, 'alpha-QA.html')