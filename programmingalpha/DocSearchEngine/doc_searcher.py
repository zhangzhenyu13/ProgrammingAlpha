# coding=utf-8
import json

from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy
from programmingalpha.DocSearchEngine.entity.post import PostJSONEncoder2
from programmingalpha.DocSearchEngine.entity.query import Query


class DocSearcherHTTPProxy(AlphaHTTPProxy):
    def __init__(self, config_file):
        AlphaHTTPProxy.__init__(self, config_file)

    def processCore(self, data):

        title = data["title"]
        size = data["size"]

        query = Query(title=title)
        query.search(size=size)
        query.arrange()
        post_results = query.get_results()

        # post_json_list = json.dumps(post_results, cls=PostJSONEncoder2)
       
        post_json_list = []
        for obj in post_results:
            all_text = obj.question_obj.title + ' ' + obj.question_obj.parsed_body
            for i, ans_obj in enumerate(obj.answer_obj_list):
                if i == 3:
                    break
                all_text = all_text + ' ' + ans_obj.parsed_body

            dic = {'id': obj.question_obj.es_id, 'text': all_text}
            post_json_list.append(dic)
        
        results = {}
        results['posts'] = post_json_list
        results['question'] = title

        return results
