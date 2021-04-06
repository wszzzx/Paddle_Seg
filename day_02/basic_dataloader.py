import os
import random
import numpy as np
import cv2
import paddle.fluid as fluid

class BasicDataLoader():
    def __init__(self,
                 image_folder,
                 image_list_file,
                 transform=None,
                 shuffle=True):
        self.imade_floder = image_folder
        self.image_list_file = image_list_file
        self.transform = transform
        self.datalist = self.read_list()

    def read_list(self):
        data_list = []
        with open(self.image_list_file) as f:
            for line in f.readlines():
                img = line.strip().split()[0]
                grouptruth = line.strip().split()[1]
                data_list.append((img,grouptruth))
        return data_list
    def preprocess(self, data, label):
        h,w,c = data.shape
        h_gt,w_gt = label.shape
        assert h==h_gt,"Error"
        assert w==w_gt,"Error"
        if self.transform:
            data,label = self.transform(data,label)
        label = label[:,:,np.newaxis]
        return data,label


    def __len__(self):
        print(len(self.datalist))
        return len(self.datalist)

    def __call__(self):
        return self.datalist




def main():
    batch_size = 5
    place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        # TODO: craete BasicDataloder instance
        basic_dataloader = BasicDataLoader()
        image_folder="./dummy_data"
        image_list_file="./dummy_data/list.txt"

        # TODO: craete fluid.io.DataLoader instance
        dataloader = fluid.io.DataLoader(capacity=1)

        # TODO: set sample generator for fluid dataloader
        dataloader.batch_sampler(batch_size=batch_size,dataloader = basic_dataloader)


        num_epoch = 2
        for epoch in range(1, num_epoch+1):
            print(f'Epoch [{epoch}/{num_epoch}]:')
            for idx, (data, label) in enumerate(dataloader):
                print(f'Iter {idx}, Data shape: {data.shape}, Label shape: {label.shape}')

if __name__ == "__main__":
    main()
