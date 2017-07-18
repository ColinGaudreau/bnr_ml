import SimpleHTTPServer
import SocketServer
import os

os.chdir('build/html')

PORT = 8000
ADDRESS = '130.179.130.2'

Handler = SimpleHTTPServer.SimpleHTTPRequestHandler

httpd = SocketServer.TCPServer((ADDRESS, PORT), Handler)

print "serving at address %s, and port %d" % (ADDRESS, PORT)

try:
	httpd.serve_forever()
except KeyboardInterrupt:
	httpd.server_close()
	print "Server closed."
