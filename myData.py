# -*- coding:utf-8 -*-

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import itertools
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class dataProcess(object):
    def __init__(self, out_rows, out_cols, train_path="E:/oneplus/oneplus/", train_label="E:/oneplus/oneplus/",
                 val_path="E:/oneplus/", val_label="E:/oneplus/",
                 test_path="E:/oneplus/oneplus/", test_label='E:/oneplus/oneplus/', npy_path="./npydata", img_type="png",batch_size = 2):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.train_path = train_path
        self.train_label = train_label
        self.img_type = img_type
        self.val_path = val_path
        self.val_label = val_label
        self.test_path = test_path
        self.test_label = test_label
        self.npy_path = npy_path
        self.batch_size = batch_size

    def label2class(self, label):
        x = np.zeros([self.out_rows, self.out_cols, 3])
        for i in range(self.out_rows):
            for j in range(self.out_cols):
                x[i, j, int(label[i][j])] = 1  # 属于第m类，第三维m处值为1
        return x


    def getTrainPath(self):
        traintxtpath = "E:/oneplus/oneplus/train.txt"
        ft = open(traintxtpath, 'r')
        trainimg_pathlist = []
        trainmask_pathlist = []
        tdata = ft.readlines()
        for i,line in enumerate(tdata):
            #训练图像的位置数组
            trainimgpath, trainmaskpath = line[:-1].split()
            trainimg_pathlist.append(self.train_path + trainimgpath)
            trainmask_pathlist.append(self.train_path + trainmaskpath)
        #assert len(trainimg_pathlist) == len(trainmask_pathlist)
        ft.close()
        return trainimg_pathlist,trainmask_pathlist

    
    def getTestPath(self):
        testtxtpath = "E:/oneplus/oneplus/val.txt"  # 验证过程
        tt = open(testtxtpath, 'r')
        testimg_pathlist = []
        testmask_pathlist = []
        tedata = tt.readlines()
        for j,line in enumerate(tedata):
            #val图像的位置数组
            testimgpath, testmaskpath= line[:-1].split()
            testimg_pathlist.append(self.test_path + testimgpath)
            testmask_pathlist.append(self.test_path + testmaskpath)
        #assert len(testimg_pathlist) == len(testmask_pathlist)
        tt.close()
        return  testimg_pathlist,testmask_pathlist


    def trainGen(self)
        trainimg_pathlist,trainmask_pathlist = self.getTrainPath()
        testimg_pathlist,testmask_pathlist = self.getTestPath()
        #imgs:图片的路径列表
        #labels:mask图的路径列表
        imgs = trainimg_pathlist + testimg_pathlist
        labels = trainmask_pathlist + testmask_pathlist
        zipped = itertools.cycle(zip(imgs, labels))
        #imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3),dtype=np.uint8)
        #imglabels = np.ndarray((len(labels),self.out_rows,self.out_cols,3),dtype=np.uint8)

        while True:
            imgdatas = []
            imglabels = []
            for _ in range(self.batch_size):
                im,msk = next(zipped)
                img = load_img(im, grayscale=False, target_size=[512, 512])
                label = load_img(msk, grayscale=True, target_size=[512, 512])
                img = img_to_array(img)
                label = self.label2class(img_to_array(label))
                img = img/255.
                label = label/255.
                imgdatas.append(img)
                imglabels.append(label)
            yield np.array(imgdatas),np.array(imglabels)
            

    def valGen(self):
        print('loading the test img to predict')
        valtxt = "E:/oneplus/oneplus/te.txt"
        f = open(valtxt,'r')
        data = f.readlines()
        imgpathlist = []
        for tline in data:
            tpath = tline.split("\n")[0]
            imgpathlist.append(tpath)
        imgdatas = np.ndarray((len(imgpathlist), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        for i in range(len(imgpathlist)):
            imgpath = imgpathlist[i]
            #print(imgpath)
            img_path = str(self.val_path + imgpath)
            imgpathlist.append(img_path)
            img = load_img(img_path,grayscale=False,target_size=[512,512])
            img = img_to_array(img)
            imgdatas[i] = img

        imgdatas = imgdatas.astype('float32')
        imgdatas /=255.
        print("loading the test img done")
        return imgdatas


    def create_train_data(self):
        i = 0
        print('Creating training images...')
        imgs0 = sorted(glob.glob(self.train_path+"/*."+self.img_type))
        imgs1 = sorted(glob.glob(self.test_path+"/*."+self.img_type))
        imgs = imgs0 + imgs1
        labels0 = sorted(glob.glob(self.train_label+"/*."+self.img_type))
        labels1 = sorted(glob.glob(self.test_label + "/*." + self.img_type))
        labels = labels0 + labels1
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        imglabels = np.ndarray((len(labels), self.out_rows, self.out_cols, 12), dtype=np.uint8)
        print(len(imgs), len(labels))

        for x in range(len(imgs)):
            imgpath = imgs[x]
            labelpath = labels[x]
            img = load_img(imgpath, grayscale=False, target_size=[512, 512])
            label = load_img(labelpath, grayscale=True, target_size=[512, 512])
            img = img_to_array(img)
            label = self.label2class(img_to_array(label))
            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1

        print('loading done')
        np.save(self.npy_path + '/camvid_train.npy', imgdatas)
        np.save(self.npy_path + '/camvid_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

        
    def create_test_data(self):
        i = 0
        print('Creating test images...')
        imgs = glob.glob(self.val_path + "/*." + self.img_type)
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        testpathlist = []

        for imgname in imgs:
            testpath = imgname
            testpathlist.append(testpath)
            img = load_img(testpath, grayscale=False, target_size=[512, 512])
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1

        txtname = './results/camvid.txt'
        with open(txtname, 'w') as f:
            for i in range(len(testpathlist)):
                f.writelines(testpathlist[i] + '\n')
        print('loading done')
        np.save(self.npy_path + '/camvid_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

        
    def load_train_data(self):
        print('load train images...')
        imgs_train = np.load(self.npy_path + "/camvid_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/camvid_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/camvid_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test



if __name__ == "__main__":
    mydata = dataProcess(512, 512)
    mydata.trainGen()
    mydata.valGen()
    #mydata.create_train_data()
    #mydata.create_test_data()
