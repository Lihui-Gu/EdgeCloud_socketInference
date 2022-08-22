# import numpy as np
import time
import pickle
import socket
import multiprocessing

IP = "10.12.11.35"
PORT = 8801
MAX_LAYER = 6


class Data(object):
    def __init__(self, inputData, startLayer, endLayer):
        self.inputData = inputData
        self.startLayer = startLayer
        self.endLayer = endLayer


def run_model(model, queueReceiveData, queueOutputData):
    while True:
        if not queueReceiveData.empty():
            data = queueReceiveData.get()
            output = data.inputData
            startLayer = data.endLayer + 1
            endLayer = MAX_LAYER
            print("Cloud run model from %d to %d Layer" % (startLayer, endLayer))
            for i in range(startLayer, endLayer + 1):
                output += 2
            data = Data(output, startLayer, endLayer)
            queueOutputData.put(data)


def send_data(server, data):
    str = pickle.dumps(data)
    server.send(len(str).to_bytes(length=6, byteorder='big'))
    server.send(str)
    print("data send")


def receive_data(server, queueReceiveData):
    while True:
        print("start to receive")
        length = int.from_bytes(server.recv(6), byteorder='big')
        b = bytes()
        while True:
            value = server.recv(length)
            b += value
            length -= len(value)
            if length == 0:
                break
        data = pickle.loads(b)
        queueReceiveData.put(data)


if __name__ == "__main__":
    # load model
    model = {}
    # use GPU

    x = [1, 1, 1, 1, 0, 0, 0]
    queueReceiveData = multiprocessing.Queue()
    queueOutputData = multiprocessing.Queue()
    pool = multiprocessing.Pool(4)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((IP, PORT))
    print("Cloud prepare to do task")
    server.listen(1)
    conn, _ = server.accept()
    print("start connection")
    dataReceiveProcess = multiprocessing.Process(target=receive_data, name="receive_data",
                                                 args=(conn, queueReceiveData))
    modelRunnerProcess = multiprocessing.Process(target=run_model, name="run_model",
                                                 args=(model, queueReceiveData, queueOutputData))
    dataReceiveProcess.start()
    modelRunnerProcess.start()
    while True:
        if not queueOutputData.empty():
            dataCloud = queueOutputData.get()
            send_data(conn, dataCloud)

