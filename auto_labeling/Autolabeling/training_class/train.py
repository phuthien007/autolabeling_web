import datetime
import os.path
import time

import torch.cuda

from training_class.dataset import *
from training_class.config import *

'''if your computer has GPU'''


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def train(net, dataloader_dict, criterior, optimizer, num_epochs, path_save):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: " , device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/ {num_epochs}")

        # move network to device
        net.to(device)

        torch.backends.cudnn.benchmark = True


        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            if epoch == 0 and (phase == "train"):
                continue

            for inputs, labels in tqdm(dataloader_dict[phase]):
                # move inputs, labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # set gradient of optimizer
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    _, predicts = torch.max(outputs, axis=1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(predicts == labels.data)
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc :.4f}")
            if epoch_acc > 0.9 and epoch > num_epochs / 3:
                epoch += num_epochs
                break
    print('save')
    torch.save(net, path_save)
    print('end')

# if __name__ == "__main__":
def process_train(path_dir=''):
    start_time = datetime.datetime.now()
    # tạo tập dataset
    if path_dir != '':
        path_train, list_class = make_datapath_list('train', path_dir)
        path_val = make_datapath_list('val', path_dir)[0]
    else:
        path_train, list_class = make_datapath_list('train')
        path_val = make_datapath_list('val')[0]
    train_dataset = MyDataset(path_train, list_class,
                              transform=ImageTransform(), phase='train')
    val_dataset = MyDataset(path_val, list_class, transform=ImageTransform(),
                            phase='val')
    dataloader = make_dataloader(batch_size=batch_size, data_train=train_dataset, data_val=val_dataset)
    # print(len(dataloader['val'].dataset))

    my_net = make_net(list_class)

    train(net=my_net["net"], dataloader_dict=dataloader, criterior=my_net["criterior"], optimizer=my_net["optimizer"],
          num_epochs=num_epochs, path_save= os.path.join(path_dir, 'my_weight.pt'))

    end_time = datetime.datetime.now()

    print(f"The process train start at {start_time} end have been done at {end_time} ")
