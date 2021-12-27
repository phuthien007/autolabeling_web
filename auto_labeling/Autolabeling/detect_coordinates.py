import datetime
import os
import shutil
import threading

import cv2
import torch.hub

from training_class import detector
from config import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def npf32u8(np_arr):
    # intensity conversion
    if str(np_arr.dtype) != 'uint8':
        np_arr = np_arr.astype(np.float32)
        np_arr -= np.min(np_arr)
        np_arr /= np.max(np_arr)  # normalize the data1 to 0 - 1
        np_arr = 255 * np_arr  # Now scale by 255
        np_arr = np_arr.astype(np.uint8)
    return np_arr


def opencv2pil(opencv_image):
    opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    opencv_image_rgb = npf32u8(opencv_image_rgb)  # convert numpy array type float32 to uint8
    pil_image = Image.fromarray(opencv_image_rgb)  # convert numpy array to Pillow Image Object
    return pil_image


def extract_coordonates(model='',classifier='', threshold=0.5, img_path='./datasets/bus.jpg'):
    # load picture
    frame = cv2.imread(img_path)

    height, width, channels = frame.shape

    # detect
    # model = model.to(device)
    # frame = frame.to(device)

    detections = model(frame)

    # format data1
    results = detections.pandas().xyxy[0]
    results['class'] = -1
    for index in range(len(results['class'])):
        cropped_image = frame[int(results['ymin'][index]):int(results['ymax'][index]),
                        int(results['xmin'][index]):int(results['xmax'][index])]
        cropped_image = npf32u8(cropped_image)
        im = opencv2pil(cropped_image)
        id_class = detector.predict(im, list_class, classifier)
        if id_class != -1:
            results['class'][index] = id_class

    results['xmin'] = results['xmin'] / width
    results['xmax'] = results['xmax'] / width
    results['ymin'] = results['ymin'] / height
    results['ymax'] = results['ymax'] / height

    results['x-center'] = (results['xmin'] + results['xmax']) / 2
    results['y-center'] = (results['ymin'] + results['ymax']) / 2
    results['width'] = results['xmax'] - results['xmin']
    results['height'] = results['ymax'] - results['ymin']

    results.drop(['xmin', 'ymin', 'xmax', 'ymax', 'name'], axis=1, inplace=True)

    # filter data1 which have conf > threshold
    for item in results.index:
        if results['confidence'][item] < threshold:
            results = results.drop(item)
    results.drop('confidence', axis=1, inplace=True)
    results = results.to_dict(orient='records')
    label_path = img_path[0:img_path.rindex('.')] + '.txt'
    if os.path.exists(label_path):
        return
    else:
        save_image(label_path, results)


def edit_label(url):
    with open(url, "r") as f:
        lst = [i.split(" ") for i in f.read().split("\n")]
        lst.remove(lst[-1])
        for item in range(len(lst)):
            lst[item][0] = '0'
            lst[item] = ' '.join(lst[item])

        res = '\n'.join(lst)

    with open(url, "w") as f:
        f.write(res)


def save_image(label_path, result):
    f = open(label_path, 'w')
    for i in range(len(result)):
        if result[i].get('class') != -1:
            f.write(str(result[i].get('class')) + ' ')
            f.write(str(result[i].get('x-center')) + ' ')
            f.write(str(result[i].get('y-center')) + ' ')
            f.write(str(result[i].get('width')) + ' ')
            f.write(str(result[i].get('height')) + ' ')
            if i != len(result):
                f.write('\n')
    f.close()


def split_label(root_img, root_label):
    if not os.path.exists(root_label + "/labels"):
        os.mkdir(root_label + "/labels")
    for url in os.listdir(root_img):
        if url[url.rindex("."):] == '.txt':
            if os.path.exists(root_label + "/labels/" + url):
                os.remove(root_label + "/labels/" + url)
            if url == "classes.txt":
                print("delete classes")
                os.remove(root_img + "/" + url)
            else:
                shutil.move(root_img + "/" + url, root_label + "/labels")
                print("Successfully remove file ", url, " to labels ")


# if __name__ == '__main__':
def process_main(path_dir):
    classifier = torch.load(os.path.join(path_dir, 'my_weight.pt'))
    classifier = classifier.to(device)
    # load weight yolov5
    if(pretrained):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=pretrained)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='static/upload_data/best.pt')



    time_start = datetime.datetime.now()
    list_image = []
    for url in tqdm(os.listdir(root)):
        if url[url.rindex(".") + 1:] != "txt":
            list_image.append(os.path.join(root, url))
    else:
        with open((root + "//classes.txt"), "w") as f:
            for item in list_class:
                f.write(item + "\n")
    for index in tqdm(range(0, len(list_image), num_thread)):
        list_thread = []
        for j in range(num_thread):
            try:
                t1 = threading.Thread(target=extract_coordonates, args=(model,classifier, 0.2, list_image[index]), )
            except:
                t1 = None
            list_thread.append(t1)
        for j in range(num_thread):
            if list_thread[j] != None:
                list_thread[j].start()
        for j in range(num_thread):
            if list_thread[j] != None:
                list_thread[j].join()
    print("Time last: ", datetime.datetime.now() - time_start)

    # split_label("datasets/images", "datasets/" )
