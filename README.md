# tf-unet-for-rgb-images

myData used for preprocess images：

    1、valGen & TrainGen:通过直接读取图片训练模型。
    
    2、create：将所有需要训练的图片保存为数组格式，存为npy文件。
       load:加载npy文件传入模型中。（这样做有个弊端：如果训练集过大，而电脑内存不够，电脑会卡住。）
Unet_camvid defined the architecture of model & train/test the model：

    1、iou meaniou定义了一个模型的检测标准。iou=预测和GT之交/预测和GT之并。
    2、模型结构：改动了 loss='categorical_crossentropy' 这个是针对多标签问题；如果是二分类，用 loss='binary_crossentropy'。
    3、训练和测试的方法其实应该单独写在其他的PY文件中，我这个写得太冗杂了。
