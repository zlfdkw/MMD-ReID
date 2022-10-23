import os


class DATASET_RGBIR():

    def __init__(self, root, print_info=False):
        self.root = root
        self.train_dir = os.path.join(root, 'bounding_box_train')
        self.gallery_dir = os.path.join(root, 'bounding_box_test2')
        self.query_dir = os.path.join(root, 'query2')

        self.check_dir_exist()

        self.train_data, self.train_id_num, self.train_img_num, self.train_cam_num \
            = self.get_data(self.train_dir, relabel=True)
        self.gallery_data, self.gallery_id_num, self.gallery_img_num, self.gallery_cam_num \
            = self.get_data(self.gallery_dir, relabel=False)
        self.query_data, self.query_id_num, self.query_img_num, self.query_cam_num \
            = self.get_data(self.query_dir, relabel=False)

        if print_info:
            self.print_statistics_info()

    def get_data(self, dir, relabel=False):
        label_set = set()
        cam_set = set()
        data = list()
        for image in os.listdir(dir):
            label, camid, _ = image.split('_')
            label, camid = int(label), int(camid[1:])   # camid starts from 0
            assert 1 <= label <= 600
            #assert 0 <= camid <= 7
            assert 1 <= camid <= 8
            label_set.add(label)
            cam_set.add(camid)
            data.append((os.path.join(dir, image), label, camid))
        if relabel:
            new_data = list()
            label2label = {label: new_label for new_label, label in enumerate(label_set)}
            for info in data:
                path, label, camid = info
                new_label = label2label[label]
                new_data.append((path, new_label, camid))
            data = new_data

        return data, len(label_set), len(data), len(cam_set)

    def check_dir_exist(self):
        if not os.path.exists(self.root):
            raise Exception('Error path: {}'.format(self.root))
        if not os.path.exists(self.train_dir):
            raise Exception('Error path:{}'.format(self.train_dir))
        if not os.path.exists(self.gallery_dir):
            raise Exception('Error path:{}'.format(self.gallery_dir))
        if not os.path.exists(self.query_dir):
            raise Exception('Error path:{}'.format(self.query_dir))

    def print_statistics_info(self):
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(self.train_id_num, self.train_img_num, self.train_cam_num))
        print("  query    | {:5d} | {:8d} | {:9d}".format(self.query_id_num, self.query_img_num, self.query_cam_num))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(self.gallery_id_num, self.gallery_img_num, self.gallery_cam_num))
        print("  ----------------------------------------")

if __name__ == '__main__':
    dataset = DATASET_RGBIR("C:\code\\vscode\MMD-ReID\data_set\RGBNT\\rgbir", False)
    train_data = dataset.train_data
    test_data = dataset.gallery_data
    query_data = dataset.query_data
    for i, data in enumerate(train_data):
        print(data)
        if (i == 299):
            break
    for i, data in enumerate(test_data):
        print(data)
        if (i == 29):
            break
    for i, data in enumerate(query_data):
        print(data)
        if (i == 29):
            break



        