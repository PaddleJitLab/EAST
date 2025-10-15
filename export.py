import torch
from model import EAST


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST().to(device)
    model.eval()
    x = torch.randn(1, 3, 320, 320).to(device)
    try:
        torch.export.export(model, args=(x,))
        print ("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print ("[JIT] torch.export failed.")
        raise e

