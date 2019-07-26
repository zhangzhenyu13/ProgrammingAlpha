from programmingalpha.MainPortal.Requester import RequesterServices
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

config_file="portalService.json"
print("staring server")
server=RequesterServices(config_file)

server.start()
print("server started")
