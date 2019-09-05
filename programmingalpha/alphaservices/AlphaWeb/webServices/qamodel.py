from programmingalpha.MainPortal.Requester import RequesterPortal
service_ip="dell-gpu-15.ivic.org.cn"
service_port="12300"
requester=RequesterPortal(service_ip, service_port)

def process(post_data):
    return requester.request(post_data)