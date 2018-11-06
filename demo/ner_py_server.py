#!/usr/bin/env python

from socket import *
from time import ctime
import main_lstm_crt_predict as ner
import time
import json


HOST = ''
PORT = 21567
BUFSIZ = 1024
ADDR = (HOST, PORT)

tcpSerSock = socket(AF_INET, SOCK_STREAM)
tcpSerSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
tcpSerSock.bind(ADDR)
tcpSerSock.listen(5)

def main():
    while True:
        print("waiting for connection...")
        tcpCliSock, addr = tcpSerSock.accept()
        print("...connected from:{}".format(addr))

        while True:
            data = tcpCliSock.recv(BUFSIZ)
            if not data:
                break

            print("rev:{}".format(data.decode()))
            start = time.time()
            result = ner.predict(data.decode())
            end = time.time()
            tcpCliSock.send(result.encode())

            tcpCliSock.close

    tcpSerSock.close

if __name__ == '__main__':
    main()