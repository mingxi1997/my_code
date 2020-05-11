import tkinter as tk
import threading
import random
import time
import mp3player
import cam_demo
import cv2
from PIL import Image,ImageTk


def Opencv2Canvas(imgCV_in, canva, layout="null"):
    """
    Showimage()是一个用于在tkinter的canvas控件中显示OpenCV图像的函数。
    使用前需要先导入库
    import cv2 as cv
    from PIL import Image,ImageTktkinter
    并注意由于响应函数的需要，本函数定义了一个全局变量 imgTK，请不要在其他地方使用这个变量名!
    参数：
    imgCV_in：待显示的OpenCV图像变量
    canva：用于显示的tkinter canvas画布变量
    layout：显示的格式。可选项为：
        "fill"：图像自动适应画布大小，并完全填充，可能会造成画面拉伸
        "fit"：根据画布大小，在不拉伸图像的情况下最大程度显示图像，可能会造成边缘空白
        给定其他参数或者不给参数将按原图像大小显示，可能会显示不全或者留空
    """
    global imgTK
    canvawidth = int(canva.winfo_reqwidth())
    canvaheight = int(canva.winfo_reqheight())
    sp = imgCV_in.shape
    cvheight = sp[0]  # height(rows) of image
    cvwidth = sp[1]  # width(colums) of image
    if (layout == "fill"):
        imgCV = cv2.resize(imgCV_in, (canvawidth, canvaheight), interpolation=cv2.INTER_AREA)
    elif (layout == "fit"):
        if (float(cvwidth / cvheight) > float(canvawidth / canvaheight)):
            imgCV = cv2.resize(imgCV_in, (canvawidth, int(canvawidth * cvheight / cvwidth)),
                               interpolation=cv2.INTER_AREA)
        else:
            imgCV = cv2.resize(imgCV_in, (int(canvaheight * cvwidth / cvheight), canvaheight),
                               interpolation=cv2.INTER_AREA)
    else:
        imgCV = imgCV_in
    imgCV2 = cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
    current_image = Image.fromarray(imgCV2)  # 将图像转换成Image对象
    imgTK = ImageTk.PhotoImage(image=current_image)  # 将image对象转换为imageTK对象
    canva.create_image(0, 0, anchor=tk.NW, image=imgTK)

window = tk.Tk()
window.title("object detection system")  # 窗口标题
width = 800
height = 600
screenwidth = window.winfo_screenwidth()
screenheight = window.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, (screenheight-height)/2)
window.geometry(alignstr)
# window.geometry('800x600')  # 窗口大小
window["background"] = "green"
window.overrideredirect(True)
# window.iconbitmap('spider_128px_1169260_easyicon.net.ico') #窗口图标
# window.resizable(0, 0)  # 防止用户调整尺寸和静止最大化
# 设置窗口居中
label_pic = tk.Canvas(window, bg='black')  # 创建标签
label_pic.place(x=0, y=1, width=600, height=600, anchor=tk.NW)  # 放置标签

label_bottle = tk.Label(window, text="瓶子个数：")  # 创建标签
label_bottle.place(x=601, y=1, width=100, height=30, anchor=tk.NW)  # 放置标签
text_bottle = tk.Text(window, font=('Helvetica', '14', 'bold'))  # 创建标签
text_bottle.insert('insert', '0')
text_bottle.place(x=702, y=1, width=100, height=30, anchor=tk.NW)  # 放置标签
text_bottle.tag_configure("center", justify='center')
text_bottle.tag_add("center", 0.0, "insert")

label_cup = tk.Label(window, text="杯子个数：")  # 创建标签
label_cup.place(x=601, y=32, width=100, height=30, anchor=tk.NW)  # 放置标签
text_cup = tk.Text(window, font=('Helvetica', '14', 'bold'))  # 创建标签
text_cup.insert('insert', '0')
text_cup.place(x=702, y=32, width=100, height=30, anchor=tk.NW)  # 放置标签
text_cup.tag_configure("center", justify='center')
text_cup.tag_add("center", 0.0, "insert")

label_man = tk.Label(window, text="人的个数：")  # 创建标签
label_man.place(x=601, y=63, width=100, height=30, anchor=tk.NW)  # 放置标签
text_man = tk.Text(window, font=('Helvetica', '14', 'bold'))  # 创建标签
text_man.insert('insert', '0')
text_man.place(x=702, y=63, width=100, height=30, anchor=tk.NW)  # 放置标签
text_man.tag_configure("center", justify='center')
text_man.tag_add("center", 0.0, "insert")

def updatedata():
    while start_flag:
        text_bottle.delete('1.0', 'end')
        var_bottle =str(random.randint(0, 9))
        text_bottle.insert('insert',var_bottle )

        text_cup.delete('1.0', 'end')
        var_cup = str(random.randint(0, 9))
        text_cup.insert('insert', var_cup)

        text_man.delete('1.0', 'end')
        var_man = str(random.randint(0, 9))
        text_man.insert('insert', var_man)
        mp3.play([var_bottle,var_cup,var_man])
        if start_flag==False:
            mp3.stop()
            break
        time.sleep(10)

hit_stop =True
state_text=""
def getdata(bottle_count,cup_count,man_count,img,fps):
    global state_text
    text_bottle.delete('1.0', 'end')
    text_bottle.insert('insert', bottle_count)
    text_cup.delete('1.0', 'end')
    text_cup.insert('insert', cup_count)
    text_man.delete('1.0', 'end')
    text_man.insert('insert', man_count)
    Opencv2Canvas(img, label_pic)
    text=bottle_count+cup_count+man_count
    if text != state_text:
        state_text = text
        mp3.play([bottle_count, cup_count, man_count])


yolov3 = cam_demo.Yolov3Manager(getdata)
def btn_start():
    global hit_stop
    global start_flag
    global yolov3
    if hit_stop:
        hit_stop=False
        var_start.set('停止')
        # threading.Thread(target=updatedata).start()
        threading.Thread(target=yolov3.start).start()
    else:
        hit_stop = True
        start_flag = False
        label_pic.delete(tk.ALL)
        yolov3.stop()
        var_start.set('开始')
var_start = tk.StringVar()
var_start.set('开始')
button_start = tk.Button(window, textvariable=var_start, command=btn_start, cursor='hand2')
button_start.place(x=800 - 200, y=600 - 60-60, width=200, height=60, anchor=tk.NW)


mp3 = mp3player.mp3player()
start_flag = True
def btn_close():
    global start_flag
    global mp3
    global yolov3
    start_flag = False
    mp3.stop()
    yolov3.stop()
    window.destroy()

button_close = tk.Button(window, text='退出', command=btn_close, cursor='hand2')
button_close.place(x=800 - 200, y=600 - 60, width=200, height=60, anchor=tk.NW)






window.mainloop()
