import socket
s=socket.socket()
s.settimeout(2)
try:
 s.connect(('1.1.1.1',443))
 print('NETWORK_REACHABLE')
 raise SystemExit(1)
except OSError as e:
 print('NETWORK_BLOCKED:'+type(e).__name__)
