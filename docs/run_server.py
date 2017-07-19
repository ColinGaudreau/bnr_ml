import SimpleHTTPServer
import SocketServer
import os
import ssl

os.chdir('build/html')

PORT = 8000
ADDRESS = '130.179.130.2'

Handler = SimpleHTTPServer.SimpleHTTPRequestHandler

httpd = SocketServer.TCPServer((ADDRESS, PORT), Handler)
httpd.socket = ssl.wrap_socket (httpd.socket, 
        keyfile="/Users/colingaudreau/.ssl/key.pem", 
        certfile='/Users/colingaudreau/.ssl/cert.pem', server_side=True)

print "serving at address %s, and port %d" % (ADDRESS, PORT)

try:
	httpd.serve_forever()
except KeyboardInterrupt:
	httpd.server_close()
	print "Server closed."
