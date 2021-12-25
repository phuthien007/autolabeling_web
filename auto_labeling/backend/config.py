from training_class.dataset import *

# setup root
root = "./datasets/images"

list_class = make_datapath_list('train', "training_class\\data\\")[1]
num_thread = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

pretrained = True

def resetPretrain(status):
    pretrained = status