import os
import torch
import network
from glob import glob
from PIL import Image
from torchvision import transforms as T
from utils import Cityscapes\
# Edge device load model
if __name__ == '__main__':
    # use cpu
    device = torch.device("cpu")
    # trained model path
    ckpt_path = 'model/deeplabv3plus.pth'
    num_classes = 19
    # load model
    model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16)
    checkpoint = torch.load('model/deeplabv3plus.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    print("load model from %s finished" % ckpt_path)
    del checkpoint
    # raw picture crop size
    crop_size = 513
    transform = T.Compose([
        T.Resize(crop_size),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    decode_fn = Cityscapes.decode_target
    img_files = []
    input_file = "data/picture"
    # get filename from dictionary
    if os.path.isdir(input_file):
        for ext in ['png', 'jpg']:
            files = glob(os.path.join(input_file, '**/*.%s' % (ext)), recursive=True)
            if len(files) > 0:
                img_files.extend(files)

    print(img_files)
    with torch.no_grad():
        model = model.eval()
        for img_path in img_files:
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext) - 1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0)
            img = img.to(device)

            # HW
            pred = model(img).max(1)[1].cpu().numpy()[0]
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            colorized_preds.save(os.path.join("result", img_name+'.png'))






