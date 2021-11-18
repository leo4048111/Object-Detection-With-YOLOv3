import tkinter as tk
from tkinter import *
from tkinter.filedialog import askdirectory, askopenfilename
from PIL import Image, ImageTk
from infer import infer
import os

img = None
currentImagePath = None
outputImageDir = None
imgList = []
curImagePos = 0

def resize(w, h, w_box, h_box, pil_image):
  '''
  resize a pil_image object so it will fit into
  a box of size w_box times h_box, but retain aspect ratio
  对一个pil_image对象进行缩放，让它在一个矩形框内，还能保持比例
  '''
  f1 = 1.0*w_box/w # 1.0 forces float division in Python2
  f2 = 1.0*h_box/h
  factor = min([f1, f2])
  #print(f1, f2, factor) # test
  # use best down-sizing filter
  width = int(w*factor)
  height = int(h*factor)
  return pil_image.resize((width, height), Image.ANTIALIAS)


def selectFile():
    return

def openImg(p):
    imgOpen = Image.open(p)
    w, h = imgOpen.size
    imgOpen = resize(w, h, displayw,displayh,imgOpen)
    global img
    img = ImageTk.PhotoImage(imgOpen)
    labelImg['image'] = img
    return

def selectImgCallback():
    global currentImagePath
    currentImagePath = tk.filedialog.askopenfilename(title='Select Image File',
                                          filetypes=[('JPG files','*.jpg'),
                                                     ('PNG files','*.png')])
    imgPath.set(currentImagePath)
    if(currentImagePath):
        openImg(currentImagePath)
    return

def selectDirCallback():
    _imgDir = tk.filedialog.askdirectory(title='Select Image Directory')
    imgDir.set(_imgDir)
    global imgList
    global curImagePos
    curImagePos = 0
    imgList = []
    for filename in os.listdir(_imgDir):
        filetype = os.path.splitext(filename)[1]
        if filetype == '.jpg' or filetype == '.png':
            imgList.append(_imgDir+'/'+filename)
    return

def selectOutputDirCallback():
    global outputImageDir
    outputImageDir = tk.filedialog.askdirectory(title='Select Output Directory')
    outputDirStrVal.set(outputImageDir)
    return

def inferButtonCallback():
    if outputImageDir == None:
        errorText = tk.Label(statFrame, text='Critical Error: Output Directory Not Selected!', bg='red')
        errorText.after(1000, errorText.destroy)
        errorText.grid(row=4, column=0, sticky='w')
        return

    infer(currentImagePath, outputImageDir, openImg)
    out_path = outputImageDir
    out_path += '/' + os.path.splitext(os.path.basename(currentImagePath))[0]+'-result.jpg'
    outputImageStrVal.set(out_path)
    return

def prevCallback():
    if len(imgList) == 0:
        return
    global curImagePos
    global currentImagePath
    curImagePos = curImagePos -1
    if curImagePos <0:
        curImagePos = len(imgList) - 1
    currentImagePath = imgList[curImagePos]
    imgPath.set(currentImagePath)
    openImg(currentImagePath)
    return

def nextCallback():
    if len(imgList) == 0:
        return
    global curImagePos
    global currentImagePath
    curImagePos = curImagePos + 1
    if curImagePos >= len(imgList):
        curImagePos = 0
    currentImagePath = imgList[curImagePos]
    imgPath.set(currentImagePath)
    openImg(currentImagePath)
    return

if __name__ == '__main__':
    window = tk.Tk()
    window.resizable(width=False, height=False)
    window.title('Object Detection')

    displayw = 720
    displayh = 480
    ##init control frame
    controlFrame = tk.Frame(width=150, height=displayh, bg='white')
    controlFrame.grid(row=0, column=0,)
    controlFrame.grid_propagate(0)

    ##init display frame
    displayFrame = tk.Frame(width=displayw, height=displayh, bg='grey')
    displayFrame.grid(row=0, column=1)
    displayFrame.grid_propagate(0)

    ##init stat frame
    statFrame = tk.Frame(width=displayw+150, height=100, bg='white')
    statFrame.grid(row=1, column=0, columnspan=2)

    ##init stats
    outputDirStrVal = StringVar()
    outputLabel = tk.Label(statFrame, text='Output Directory:')
    outputLabel.grid(row=0, column=0, sticky='w')
    outputDirText = tk.Label(statFrame, textvariable=outputDirStrVal, bg='grey')
    outputDirText.grid(row=0, column=1, sticky='w')

    imgPath = StringVar()
    imgPathLabel = tk.Label(statFrame, text='Image Path:')
    imgPathLabel.grid(row=1, column=0, sticky='w')
    imgPathText = tk.Label(statFrame, textvariable=imgPath, bg='grey')
    imgPathText.grid(row=1, column=1, sticky='w')

    imgDir = StringVar()
    imgDirLabel = tk.Label(statFrame, text='Image Dir:')
    imgDirLabel.grid(row=2, column=0, sticky='w')
    imgDirText = tk.Label(statFrame, textvariable=imgDir, bg='grey')
    imgDirText.grid(row=2, column=1, sticky='w')

    outputImageStrVal = StringVar()
    outputImagePathLabel = tk.Label(statFrame, text='Output Image:')
    outputImagePathLabel.grid(row=3, column=0, sticky='w')
    outputImagePathText = tk.Label(statFrame, textvariable=outputImageStrVal, bg='red')
    outputImagePathText.grid(row=3, column=1, sticky='w')

    ##Init selection button
    selectImg = tk.Button(controlFrame,
                          text='Select Image...',
                          command=selectImgCallback,
                          height=5,
                          width=20)
    selectImg.grid(row=1, column=0, sticky='w', pady=1)

    ##Init selection dir button
    selectDir = tk.Button(controlFrame,
                          text='Select Dir...',
                          command=selectDirCallback,
                          height=5,
                          width=20)

    selectDir.grid(row=2, column=0, sticky='w', pady=1)

    ##Init output dir selection button
    selectOutputDir = tk.Button(controlFrame,
                                text='Select Output Dir...',
                                command=selectOutputDirCallback,
                                height=5,
                                width=20)
    selectOutputDir.grid(row=3, column=0, sticky='w', pady=1)

    ##Init inference button
    inferButton = tk.Button(controlFrame,
                            text='INFER',
                            command=inferButtonCallback,
                            height=5,
                            width=20)
    inferButton.grid(row=4, column=0, sticky='w', pady=1)

    ##Init prev/next button
    npFrame = tk.Frame(controlFrame, width=150)
    npFrame.grid(row=5, column=0)
    prevButton = tk.Button(npFrame,
                           text='Prev',
                           height=5,
                           width=8,
                           command=prevCallback)
    prevButton.grid(row=0, column=0, sticky='w')
    nextButton = tk.Button(npFrame,
                           text='Next',
                           height=5,
                           width=8,
                           command=nextCallback)
    nextButton.grid(row=0, column=1, sticky='w')

    ##Init Image
    labelImg = tk.Label(window, image=None, bg='grey')
    labelImg.grid(row=0, column=1, sticky='nesw')

    #add to loop queue
    window.mainloop()