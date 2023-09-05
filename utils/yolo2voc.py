import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import cv2
import shutil
import argparse

def marge_txt(path1,path2,imagesets_path): #path1 path2 对应train和val的txt
    out_txt = imagesets_path + '/' + 'trainval.txt'
    f_out = open(out_txt, 'a+')
    f_1 = open(path1)
    f_2 = open(path2)
    f_out.write(f_1.read() + '\n')
    f_1.close()
    f_out.write(f_2.read() + '\n')
    f_2.close()
    f_out.close()
    
class YOLO2VOCConvert:
    def __init__(self, txts_path, xmls_path, imgs_path,imagesets_path,data_set,image_out_path): # data_set代表数据集的种类 是train还是val
        self.txts_path = txts_path   # 标注的yolo格式标签文件路径
        self.xmls_path = xmls_path   # 转化为voc格式标签之后保存路径
        self.imgs_path = imgs_path   # 读取读片的路径个图片名字，存储到xml标签文件中
        self.imagesets_path = imagesets_path #这个代表输出 VOC格式中ImageSets格式文件夹的路径
        self.data_set = data_set     # 代表对应集合是训练集还是验证集
        self.image_out_path = image_out_path #图像输出对应的地址 
        self.classes = ['person']  # 从所有的txt文件中提取出所有的类别， yolo格式的标签格式类别为数字 0,1,...

    
    # writer为True时，把提取的类别保存到'./Annotations/classes.txt'文件中
    def search_all_txt_classes(self, writer=False):
        # 读取每一个txt标签文件，取出每个目标的标注信息
        all_names = set()
        txts = os.listdir(self.txts_path)
        # 使用列表生成式过滤出只有后缀名为txt的标签文件
        txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']
        #print(len(txts), txts)
        # 11 ['0002030.txt', '0002031.txt', ... '0002039.txt', '0002040.txt']
        for txt in txts:
            txt_file = os.path.join(self.txts_path, txt)
            with open(txt_file, 'r') as f:
                objects = f.readlines()
                for object in objects:
                    object = object.strip().split(' ')
                    #print(object)  # ['2', '0.506667', '0.553333', '0.490667', '0.658667']
                    all_names.add(int(object[0]))

        print("all lables", all_names, "all txts %d" % len(txts))

        return list(all_names)
    
    def search_all_img_classes(self, writer=False):
        # 读取每一个txt标签文件，取出每个目标的标注信息
        all_names = set()
        imgs = os.listdir(self.imgs_path)
        # 使用列表生成式过滤出只有后缀名为txt的标签文件
        imgs = [img for img in imgs if txt.split('.')[-1] == 'jpg']
        #print(len(txts), txts)
        # 11 ['0002030.txt', '0002031.txt', ... '0002039.txt', '0002040.txt']
        for img in imgss:
            img_file = os.path.join(self.imgs_path, img)
            with open(txt_file, 'r') as f:
                objects = f.readlines()
                for object in objects:
                    object = object.strip().split(' ')
                    #print(object)  # ['2', '0.506667', '0.553333', '0.490667', '0.658667']
                    all_names.add(int(object[0]))
            # print(objects)  # ['2 0.506667 0.553333 0.490667 0.658667\n', '0 0.496000 0.285333 0.133333 0.096000\n', '8 0.501333 0.412000 0.074667 0.237333\n']

        print("all imgs", all_names, "all imgs %d" % len(imgs))

        return list(all_names)

    def yolo2voc(self):
        
        # 创建一个保存xml标签文件的文件夹
        if not os.path.exists(self.xmls_path):
            os.makedirs(self.xmls_path)

        # 把上面的两个循环改写成为一个循环：
        imgs = os.listdir(self.imgs_path)
        txts = os.listdir(self.txts_path)
        
        imgs = [img for img in imgs if img.split('.')[-1] == 'jpg']
        txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']  
        map_imgs_txts = [(img, txt) for img, txt in zip(imgs, txts)]
        
        for img_name, txt_name_useless in map_imgs_txts:
            img_number = (img_name.split('.')[0]+'.txt') # 这一步的img-number 可以作为输出 后续输出成txt文件夹的依据
            txt_name = img_number   #改进
            
            # 从这里开始 对图片进行复制
            if not os.path.exists(self.image_out_path):
                os.makedirs(self.image_out_path)             
            image_in_path =  os.path.join(self.imgs_path, img_name)
            image_out_path = os.path.join(self.image_out_path, img_name)
            shutil.copy(image_in_path,image_out_path)
            
            # 这里开始 输出每张图片的标签
            if self.data_set == 'train':
                train_txt_path = self.imagesets_path + '/' + 'train.txt'
                if not os.path.exists(self.imagesets_path):
                    os.makedirs(self.imagesets_path)
                file_train_txt = open(train_txt_path, 'a')
                file_train_txt.write(img_name.split('.')[0])
                file_train_txt.write('\n')
                file_train_txt.close()
                
            if self.data_set == 'val':
                val_txt_path = self.imagesets_path + '/' + 'val.txt'
                if not os.path.exists(self.imagesets_path):
                    os.makedirs(self.imagesets_path)
                file_val_txt = open(val_txt_path, 'a')
                file_val_txt.write(img_name.split('.')[0])
                file_val_txt.write('\n')
                file_val_txt.close()            
            
            if img_number in txts:
            
                # 读取图片的尺度信息
                print("reading image:", img_name)
                print("cur img path is ",os.path.join(self.imgs_path, img_name))  # 后添加的
                
                img = cv2.imread(os.path.join(self.imgs_path, img_name))
                height_img, width_img, depth_img = img.shape
                #print(height_img, width_img, depth_img)   # h 就是多少行（对应图片的高度）， w就是多少列（对应图片的宽度）

                # 获取标注文件txt中的标注信息
                all_objects = []
                txt_file = os.path.join(self.txts_path, txt_name)
                print("cur txt path is", txt_file)                               #后添加的
                with open(txt_file, 'r') as f:
                    objects = f.readlines()
                    for object in objects:
                        object = object.strip().split(' ')
                        all_objects.append(object)
                        #print(object)  # ['2', '0.506667', '0.553333', '0.490667', '0.658667']

                # 创建xml标签文件中的标签
                xmlBuilder = Document()
                # 创建annotation标签，也是根标签
                annotation = xmlBuilder.createElement("annotation")

                # 给标签annotation添加一个子标签
                xmlBuilder.appendChild(annotation)

                # 创建子标签folder
                folder = xmlBuilder.createElement("folder")
                # 给子标签folder中存入内容，folder标签中的内容是存放图片的文件夹，例如：JPEGImages
                folderContent = xmlBuilder.createTextNode(self.imgs_path.split('/')[-1])  # 标签内存
                folder.appendChild(folderContent)  # 把内容存入标签
                annotation.appendChild(folder)   # 把存好内容的folder标签放到 annotation根标签下

                # 创建子标签filename
                filename = xmlBuilder.createElement("filename")
                # 给子标签filename中存入内容，filename标签中的内容是图片的名字，例如：000250.jpg
                filenameContent = xmlBuilder.createTextNode(txt_name.split('.')[0] + '.jpg')  # 标签内容
                filename.appendChild(filenameContent)
                annotation.appendChild(filename)

                # 把图片的shape存入xml标签中
                size = xmlBuilder.createElement("size")
                # 给size标签创建子标签width
                width = xmlBuilder.createElement("width")  # size子标签width
                widthContent = xmlBuilder.createTextNode(str(width_img))
                width.appendChild(widthContent)
                size.appendChild(width)   # 把width添加为size的子标签
                # 给size标签创建子标签height
                height = xmlBuilder.createElement("height")  # size子标签height
                heightContent = xmlBuilder.createTextNode(str(height_img))  # xml标签中存入的内容都是字符串
                height.appendChild(heightContent)
                size.appendChild(height)  # 把width添加为size的子标签
                # 给size标签创建子标签depth
                depth = xmlBuilder.createElement("depth")  # size子标签width
                depthContent = xmlBuilder.createTextNode(str(depth_img))
                depth.appendChild(depthContent)
                size.appendChild(depth)  # 把width添加为size的子标签
                annotation.appendChild(size)   # 把size添加为annotation的子标签

                # 每一个object中存储的都是['2', '0.506667', '0.553333', '0.490667', '0.658667']一个标注目标
                for object_info in all_objects:
                    # 开始创建标注目标的label信息的标签
                    object = xmlBuilder.createElement("object")  # 创建object标签
                    # 创建label类别标签
                    # 创建name标签
                    imgName = xmlBuilder.createElement("name")  # 创建name标签
                    imgNameContent = xmlBuilder.createTextNode(self.classes[int(object_info[0])])
                    imgName.appendChild(imgNameContent)
                    object.appendChild(imgName)  # 把name添加为object的子标签

                    # 创建pose标签
                    pose = xmlBuilder.createElement("pose")
                    poseContent = xmlBuilder.createTextNode("Unspecified")
                    pose.appendChild(poseContent)
                    object.appendChild(pose)  # 把pose添加为object的标签

                    # 创建truncated标签
                    truncated = xmlBuilder.createElement("truncated")
                    truncatedContent = xmlBuilder.createTextNode("0")
                    truncated.appendChild(truncatedContent)
                    object.appendChild(truncated)

                    # 创建difficult标签
                    difficult = xmlBuilder.createElement("difficult")
                    difficultContent = xmlBuilder.createTextNode("0")
                    difficult.appendChild(difficultContent)
                    object.appendChild(difficult)

                    # 先转换一下坐标
                    # (objx_center, objy_center, obj_width, obj_height)->(xmin，ymin, xmax,ymax)
                    x_center = float(object_info[1])*width_img + 1
                    y_center = float(object_info[2])*height_img + 1
                    xminVal = int(x_center - 0.5*float(object_info[3])*width_img)   # object_info列表中的元素都是字符串类型
                    yminVal = int(y_center - 0.5*float(object_info[4])*height_img)
                    xmaxVal = int(x_center + 0.5*float(object_info[3])*width_img)
                    ymaxVal = int(y_center + 0.5*float(object_info[4])*height_img)



                    # 创建bndbox标签(三级标签)
                    bndbox = xmlBuilder.createElement("bndbox")
                    # 在bndbox标签下再创建四个子标签(xmin，ymin, xmax,ymax) 即标注物体的坐标和宽高信息
                    # 在voc格式中，标注信息：左上角坐标（xmin, ymin） （xmax, ymax）右下角坐标
                    # 1、创建xmin标签
                    xmin = xmlBuilder.createElement("xmin")  # 创建xmin标签（四级标签）
                    xminContent = xmlBuilder.createTextNode(str(xminVal))
                    xmin.appendChild(xminContent)
                    bndbox.appendChild(xmin)
                    # 2、创建ymin标签
                    ymin = xmlBuilder.createElement("ymin")  # 创建ymin标签（四级标签）
                    yminContent = xmlBuilder.createTextNode(str(yminVal))
                    ymin.appendChild(yminContent)
                    bndbox.appendChild(ymin)
                    # 3、创建xmax标签
                    xmax = xmlBuilder.createElement("xmax")  # 创建xmax标签（四级标签）
                    xmaxContent = xmlBuilder.createTextNode(str(xmaxVal))
                    xmax.appendChild(xmaxContent)
                    bndbox.appendChild(xmax)
                    # 4、创建ymax标签
                    ymax = xmlBuilder.createElement("ymax")  # 创建ymax标签（四级标签）
                    ymaxContent = xmlBuilder.createTextNode(str(ymaxVal))
                    ymax.appendChild(ymaxContent)
                    bndbox.appendChild(ymax)

                    object.appendChild(bndbox)
                    annotation.appendChild(object)  # 把object添加为annotation的子标签
                    
                f = open(os.path.join(self.xmls_path, txt_name.split('.')[0]+'.xml'), 'w')
                xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
                f.close()
                
def YOLO_To_VOC(source_dir, source_dir_val, mark_1, mark_2): #从两个大文件夹路径到合并为一个文件夹路径 source_dir2的文件会合并到source_dir_1中 第一个被当做训练集 第二个为验证集
    
    txts_path = os.path.join(source_dir,'labels', mark_1) # source_dir/labels/train      
    mark_voc_folder = source_dir.split('/')[-1] + '_voc' #将最后一级目录添加_voc的后缀 表示数据集种类为VOC
    xmls_path = os.path.join(os.path.dirname(source_dir),mark_voc_folder, 'VOC2012', 'Annotations') # /source/Annotations 
    imgs_path = os.path.join(os.path.join(source_dir,'images', mark_1))
    imagesets_path = os.path.join(os.path.dirname(source_dir),mark_voc_folder,'VOC2012', 'ImageSets','Main')
    image_out_path = os.path.join(os.path.dirname(source_dir),mark_voc_folder, 'VOC2012', 'JPEGImages')
    yolo2voc_obj1 = YOLO2VOCConvert(txts_path, xmls_path, imgs_path, imagesets_path, 'train', image_out_path)
    yolo2voc_obj1.yolo2voc() # 这一步完成txt到xml文件的格式转换
    
    # !!! txts_path为输入的txt文档路径 imgs_path为输入的imgs路径  xmls imagesets image_out 为对应VOC格式的三个文件夹输出
    txts_path_val = os.path.join(source_dir_val,'labels', mark_2) # source_dir/labels/train       指定这个训练集为后续的验证集
    mark_voc_folder_val = source_dir_val.split('/')[-1] + '_voc'
    imgs_path_val = os.path.join(os.path.join(source_dir_val,'images', mark_2))
    
    print("start changing val set")
    yolo2voc_obj2 = YOLO2VOCConvert(txts_path_val, xmls_path, imgs_path_val, imagesets_path, 'val', image_out_path)
    yolo2voc_obj2.yolo2voc() # 这一步完成txt到xml文件的格式转换  
    
    txt_train_set = imagesets_path + '/' + 'train.txt'
    txt_val_set   = imagesets_path + '/' + 'val.txt'
    print("start marging txt file")
    
    marge_txt(txt_train_set, txt_val_set, imagesets_path)  
    print("end") 
    
if __name__ == '__main__':
    #source_dir_train = '/userhome/voc_mini/voc_1/voc_1_1' #这个文件夹代表原始 yolo格式文件夹 对应的train_set
    #mark_1 = 'train'
    
    #source_dir_val = '/userhome/voc_mini/voc_1/voc_1_2'
    #mark_2 = 'train'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_train', type=str, default=' ', help='path for train subset/saved path') #该路径为最终输出保存对应的路径
    parser.add_argument('--mark_train', type=str, default='train', help='which subset for path_train')
    parser.add_argument('--path_val', type=str, default=' ', help='path for val subset')
    parser.add_argument('--mark_val', type=str, default='val', help='which subset for path_val')
    opt = parser.parse_args()
    #source_dir_train = '/userhome/viped'
    #mark_1 = 'train'
    #source_dir_val = '/userhome/viped'
    #mark_2 = 'val'
    source_dir_train = opt.path_train
    mark_1 = opt.mark_train
    source_dir_val = opt.path_val
    mark_2 = opt.mark_val
    
    YOLO_To_VOC(source_dir_train, source_dir_val, mark_1, mark_2)

    

        
  
