import torch
import network
# Edge device load model
if __name__ == '__main__':
    # use cpu
    device = torch.device("cpu")
    # trained model path
    ckpt_path = 'model/deeplabv3plus.pth'
    # load model
    model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16)
    checkpoint = torch.load('model/deeplabv3plus.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    print("load model from %s finished" % ckpt_path)