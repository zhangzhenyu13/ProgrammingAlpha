from programmingalpha.DocSearchEngine2.service.http_server import search_server

if __name__ == '__main__':
    config_file = "docSearcherService.json"
    # server = DocSearcherHTTPProxy(config_file)
    #print("init server")
    # server.start()
    search_server(config_file)