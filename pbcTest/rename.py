
import os
 
ROOT_DIR = os.path.abspath("")
img_path = ROOT_DIR
imglist = os.listdir(img_path)
#print(filelist)
i = 0
for img in imglist:
    i+=1

    if img.endswith('.jpg'):
        print(i)
        src = os.path.join(os.path.abspath(img_path), img) #原先的图片名字
        dst = os.path.join(os.path.abspath(img_path), '%d' % i + '.jpg') #根据自己的需要重新命名,可以把'E_' + img改成你想要的名字
        os.rename(src, dst) #重命名,覆盖原先的名字