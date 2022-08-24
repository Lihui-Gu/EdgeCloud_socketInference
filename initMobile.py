import numpy as np
import socket
from threading import Thread
import pickle
import multiprocessing

# import io
# import sys
import time

# parameters of model
"""
_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 1
"""

# model path
ALEXNET_MODEL_PATH = ""
VGGNET_MODEL_PATH = ""
# network communication
IP = "10.12.11.35"
PORT = 8801

MAX_LAYER = 6
# global
_data = None


# object define of Data
class Data(object):
    def __init__(self, inputData, startLayer, endLayer):
        self.inputData = inputData
        self.startLayer = startLayer
        self.endLayer = endLayer


def data_load(queueRowData):
    i = 0
    while i < 10:
        queueRowData.put(0)
        print("load data No.%d" % (i))
        time.sleep(0.1)
        i += 1

    # run model on mobile edge


def run_model(model, queueRowData, queueMobileData, startLayer, endLayer):
    while True:
        if not queueRowData.empty():
            print("Mobile run model from %d to %d layer" % (startLayer, endLayer))
            # output = model(inputData, startLayer, endLayer, False)
            output = queueRowData.get()
            for i in range(startLayer, endLayer + 1):
                output += 1
            data = Data(output, startLayer, endLayer)
            if endLayer == MAX_LAYER:
                queueOutputData.put(data.inputData)
            else:
                queueMobileData.put(data)
                print("add data to send cloud")


# send data to cloud
def send_data(client, queueMobileData):
    while True:
        if not queueMobileData.empty():
            data = queueMobileData.get()
            # conver object to string
            str = pickle.dumps(data)
            # convert int to byte, length is 6 bytes and send
            client.send(len(str).to_bytes(length=6, byteorder='big'))
            # send the data
            client.send(str)
            print("send data success")


# receive data from cloud
def receive_data(client, queueOutputData):
    while True:
        length = int.from_bytes(client.recv(6), byteorder='big')
        b = bytes()
        while True:
            value = client.recv(length)
            b = b + value
            length -= len(value)
            if length == 0:
                break
        data = pickle.loads(b)
        queueOutputData.put(data.inputData)
        print("Receive from cloud: %d" % (data.inputData))


# calculate accurency
def test(output, label):
    return (output / label) * 100


if __name__ == "__main__":
    # load model
    model = {}
    # use CPU

    # connect to cloud
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((IP, PORT))
    print("Connect cloud success")
    # define calculate task
    x = [1, 1, 1, 1, 0, 0, 0]
    print("Start calculate")
    startLayer = 0
    endLayer = sum(x) - 1
    # run a thread to load image and label
    # put data into a queue
    queueRowData = multiprocessing.Queue()
    queueMobileData = multiprocessing.Queue()
    queueOutputData = multiprocessing.Queue()

    pool = multiprocessing.Pool(4)
    dataLoaderProcess = multiprocessing.Process(target=data_load, name="data_load",
                                                args=(queueRowData,))
    modelRunnerProcess = multiprocessing.Process(target=run_model, name="run_model",
                                                 args=(model, queueRowData, queueMobileData, startLayer, endLayer))
    dataSendProcess = multiprocessing.Process(target=send_data, name="send_data",
                                              args=(client, queueMobileData))
    dataReceiveProcess = multiprocessing.Process(target=receive_data, name="receive_data",
                                                 args=(client, queueOutputData))

    start = time.time()
    dataLoaderProcess.start()
    modelRunnerProcess.start()
    dataSendProcess.start()
    dataReceiveProcess.start()

    while True:
        if not queueOutputData.empty():
            output = queueOutputData.get()
            acc = test(output, 10)
            end = time.time()
            print("Task finish, cost %f ms with acc: %f" % (end - start, acc))
    client.close()




