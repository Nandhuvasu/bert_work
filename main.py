from env_vars import constants


from wsgiref import simple_server
from wsgiref.simple_server import WSGIRequestHandler

from server.rest_server_falcon import APP



if __name__ == '__main__':
    httpd = simple_server.make_server('0.0.0.0', 8080, APP, handler_class=WSGIRequestHandler)
    httpd.serve_forever()
