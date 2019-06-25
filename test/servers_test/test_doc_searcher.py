from programmingalpha.DocSearchEngine.doc_searcher import DocSearcherHTTPProxy


if __name__ == '__main__':
    config_file = "docSearcherService.json"
    server = DocSearcherHTTPProxy(config_file)

    server.start()
