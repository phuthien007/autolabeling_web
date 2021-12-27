from training_class.config import *

'''if your computer has GPU'''


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class ImageTransform():
    def __init__(self, resize=resize, mean=mean, std=std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.5)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


def make_datapath_list(phase='train', rootpath=".\\data\\"):
    target_path = osp.join(rootpath + phase + "\\**")
    path_dict = {}
    list_class = []
    index = 0
    for path_class in glob.glob(target_path):
        list_class.append(path_class.split("\\")[-1])
        for path_img in glob.glob(path_class + "\\**"):
            path_dict.setdefault(index, (path_class.split("\\")[-1], path_img))
            index += 1
    return path_dict, list_class


class MyDataset(tdata.Dataset):
    def __init__(self, file_list, list_class, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.list_class = list_class

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, id):
        img_path = self.file_list.get(id)[1]
        img = Image.open(img_path)
        img_transform = self.transform(img, self.phase)
        return img_transform, self.list_class.index(self.file_list.get(id)[0])


def make_dataloader(batch_size=8, data_train=None, data_val=None):
    train_dataloader = tdata.DataLoader(data_train, batch_size, shuffle=True)
    val_dataloader = tdata.DataLoader(data_val, batch_size, shuffle=False)
    dataloader_dict = {"train": train_dataloader, "val": val_dataloader}
    return dataloader_dict
# print('start in dataset')
list_class = make_datapath_list('train')[1]
# print(list_class)
