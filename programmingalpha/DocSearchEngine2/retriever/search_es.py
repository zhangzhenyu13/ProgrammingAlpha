# -*- UTF-8 -*-
from elasticsearch import Elasticsearch


def search(query, url, size):
    es = Elasticsearch([url])
    doc = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["Title", "Body"]
            }
        }
    }

    results = es.search(index="posts", doc_type="posts", body=doc, size=size)
    return_list = []
    for res in results['hits']['hits']:
        return_list.append(res) # Can get res['_score']

    return return_list


if __name__ == '__main__':
    search_result_list = search("How to use println in java", size=10)
    # Post list
    title_list = []
    score_list = []
    for result in search_result_list:
        result = result['_score']
        score_list.append(result)
        # Question
        # title_list.append(result['question']['Title'])

    print(score_list)
