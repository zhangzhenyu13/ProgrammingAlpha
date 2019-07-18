from programmingalpha.MainPortal.Requester import RequesterServices

config_file="portalService.json"
print("staring server")
server=RequesterServices(config_file)

server.start()
print("server started")