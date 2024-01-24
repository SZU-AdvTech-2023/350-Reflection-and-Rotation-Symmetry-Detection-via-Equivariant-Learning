import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import Label

loss_file=open('./images_truth/loss2.txt')
loss_list=loss_file.readlines()

class App:
    def __init__(self, master):
        self.master = master
        master.title("Image Viewer")
        master.title('基于等变学习的图像对称性检测')
        master.geometry("680x580+150+50")
        # 创建标签和按钮

        self.image_path = None
        self.button1 = tk.Button(master, text="打开图片",
                                 command=lambda: [self.open_image(True,None,10,50,"输入图像")])
        self.button1.place(x=200, y=10)

        self.button2 = tk.Button(master, text="对称检测",
                                 command=lambda: [self.open_image(False,
                                './images_truth/%s_truth.jpg' % (self.image_path.split('/')[-1][:-4]),225,50,"真实对称轴"),
                                                  self.show_more_images(),
                                                  self.set_label(True)])
        self.button2.place(x=320, y=10)


        self.set_label(False)
        self.img_list=['./images/border.png','./images/border.png','./images/border.png']
        self.open_image(False,'./images/border.png',10,50,"输入图像")
        self.open_image(False,'./images/border.png',225,50,'真实对称轴')
        self.show_more_images()

    def set_label(self, open=True):
        if not open:
            self.text1 = "角度误差:"
            self.text2 = "中心偏移误差:"
        else:
            index=int(self.image_path.split('/')[-1][5:-4])-1
            line=loss_list[index][:-1].split()
            a, b = line[1], line[2]
            c, d = line[3], line[4]
            self.text1 = "角度误差:"+a+"(阈值:"+b+")"
            self.text2 = "中心偏移误差:" + c + "(阈值:" + d + ")"
        self.label1 = tk.Label(self.master, text=self.text1, font=('微软雅黑', 12))
        self.label1.place(x=450, y=120)
        self.label2 = tk.Label(self.master, text=self.text2, font=('微软雅黑', 12))
        self.label2.place(x=450, y=150)

    def open_image(self,open=True,filepath=None,x=None,y=None,text=None):
        # 打开文件对话框
        if open:
            file_path = filedialog.askopenfilename()
            # 判断是否选择了文件
            if file_path:
                # 打开图片
                img = Image.open(file_path)
                #'./demo/pred/%s_%s.png' % (im_path[0].split('/')[-1][:-4]
                #./images_result/refs_001_reflection_axis.png
                #print(file_path.split('/')[-1][:-4])#[:-4]是排除.png或.jpg
                self.img_list=[]
                self.img_list.append('./images_result/%s_reflection_axis.png'% (file_path.split('/')[-1][:-4]))
                self.img_list.append('./images_heatmap/%s_reflection_heatmap.png' % (file_path.split('/')[-1][:-4]))
                self.img_list.append('./drawhough_2/%s_symmetry.jpg' % (file_path.split('/')[-1][:-4]))
                # 调整图片大小
                img = img.resize((200, 200))

                # panel = Label(master=root)
                # panel.photo = ImageTk.PhotoImage(img)  # 将原本的变量photo改为panel.photo
                # Label(master=self.master, image=panel.photo, text="input image", anchor='center', fg='black',
                #       font=('微软雅黑', 12), compound='top').grid(row=0, column=0,padx=(10,0),pady=(60,0))
                # 显示图片
                photo = ImageTk.PhotoImage(img)
                self.label = tk.Label(self.master,text=text, anchor='center', fg='black',
                       font=('微软雅黑', 12), compound='top')
                self.label.place(x=x, y=y)
                self.label.config(image=photo)
                self.label.image = photo

                #保存图片路径
                self.image_path = file_path.split('/')[-1]
        else:
            file_path = filepath
            if file_path:
                # 打开图片
                img = Image.open(file_path)
                # 调整图片大小
                img = img.resize((200, 200))
                photo = ImageTk.PhotoImage(img)
                self.label = tk.Label(self.master, text=text, anchor='center', fg='black',
                                      font=('微软雅黑', 12), compound='top')
                self.label.place(x=x, y=y)
                self.label.config(image=photo)
                self.label.image = photo

    def load_img(self, index, item):
        # 打开图片。
        # resize()：示例图片太大，这里缩小一些。
        img = Image.open(item).resize((200, 200))
        # 引用：添加一个Label，用来存储图片。使用PanedWindow也行。
        panel = Label(master=root)
        panel.photo = ImageTk.PhotoImage(img)  # 将原本的变量photo改为panel.photo

        # 图片用Label来显示，参数master改不改为panel都行，这里就不改了。
        # 注意：参数image改为panel.photo
        list=['黑白得分图','热力图','预测对称轴']
        Label(master=self.master, image=panel.photo, text=list[index],anchor='center',fg='black',
              font=('微软雅黑',12),compound='top').grid(row=0, column=index,padx=(10,0),pady=(300,0))

        # 使用for循环添加图片，enumerate：获取元素与其索引值
    def show_more_images(self):
        # 总共3张图片
        img_list = self.img_list
        for index, item in enumerate(img_list):
            self.load_img(index, item)  # 执行函数
root = tk.Tk()
app = App(root)
root.mainloop()