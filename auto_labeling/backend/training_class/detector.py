from training_class.train import *
from PIL import  Image
class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):

        max_id = np.argmax(out.detach().numpy())
        if (out.detach().numpy()[0][max_id] < 0.5):
            max_id = -1
        # predict_label = self.class_index[max_id]
        return max_id

# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(img, list_class, model):

    predictor = Predictor(list_class)

    # prepare network
    # net = models.vgg16(pretrained=True, progress=True)
    # net.training_class[6] = nn.Linear(in_features=4096, out_features=len(list_class))
    # net.eval()

    # prepare input img
    transform = ImageTransform()
    img = transform(img, phase="test")
    img = img.unsqueeze_(0)
    img = img.to(device)
    #predict
    output = model(img)
    out = predictor.predict_max(output)
    return  out


#
# img = Image.open('C:\\Users\\Thien Phu\\Documents\\tài liệu học Hust\\20211\Project I\\auto_labeling-20211211T124600Z-001\\auto_labeling\\backend\\datasets\images\\65523440_1065519380300550_476553361957584896_n.jpg')
# out = predict(img, list_class)
# print(list_class[out])