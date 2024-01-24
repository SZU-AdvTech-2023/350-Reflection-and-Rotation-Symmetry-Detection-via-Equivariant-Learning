import cv2
import numpy as np
import math

def loss(pa,pb, H,W):
    x1, y1, x2, y2 = pa
    x3, y3, x4, y4 = pb
    # 两条直线的斜率
    slope1 = (y2-y1)/(x2-x1-0.01)
    slope2 = (y4-y3)/(x4-x3-0.01)
    # 计算两条直线的夹角
    angle = np.arctan((slope1 - slope2) / (1 + slope1 * slope2))
    # 将弧度转换为角度
    angle_degree = np.degrees(angle)
    Angle=math.fabs(round(angle_degree,2))
    # 输出结果
    #print("两条直线的夹角为：", angle_degree, "度")

    #计算两条直线的中心点
    c1x = (x1+x2)/2
    c1y = (y1+y2)/2
    c2x = (x3+x4)/2
    c2y = (y3+y4)/2
    #print(c1x,c1y, c2x,c2y)
    #print("中心点距离：", pow(( (c1x-c2x)*(c1x-c2x) + (c1y-c2y)*(c1y-c2y) ),0.5))
    Dis=round(pow(( (c1x-c2x)*(c1x-c2x) + (c1y-c2y)*(c1y-c2y) ),0.5),2)
    #print("中心点距离评估标准",0.025*min(W,H))
    return Angle,3,Dis,round(0.025*min(W,H),2)

def loss1(pa,pb, H,W):
    x1, y1, x2, y2 = pa
    x3, y3, x4, y4 = pb
    # 两条直线的斜率
    slope1 = (y2-y1)/(x2-x1-0.01)
    slope2 = (y4-y3)/(x4-x3-0.01)
    # 计算两条直线的夹角
    angle = np.arctan((slope1 - slope2) / (1 + slope1 * slope2))
    # 将弧度转换为角度
    angle_degree = np.degrees(angle)
    Angle=math.fabs(round(angle_degree,2))

    #计算两条直线的中心点
    c1x = (x1+x2)/2
    c1y = (y1+y2)/2
    c2x = (x3+x4)/2
    c2y = (y3+y4)/2
    len1 = round(pow(( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) ),0.5))
    len2 = round(pow(((x3 - x4) * (x3 - x4) + (y3 - y4) * (y3 - y4)), 0.5))
    #print("中心点距离：", pow(( (c1x-c2x)*(c1x-c2x) + (c1y-c2y)*(c1y-c2y) ),0.5))
    Dis=round(pow(( (c1x-c2x)*(c1x-c2x) + (c1y-c2y)*(c1y-c2y) ),0.5),2)
    #print("中心点距离评估标准",0.025*min(W,H))
    return Angle,3,Dis,round(0.2*min(len1,len2),2)

truth_file=open("./images_truth/groudtruth.txt")
test_file=open("./images_truth/1_symmetry.txt")
H_W_file=open('./images_truth/height_width.txt')
loss_file=open('./images_truth/loss2.txt','w')
truth_list=truth_file.readlines()
test_list=test_file.readlines()
H_W_list=H_W_file.readlines()
sum1=0
sum2=0
angle_list=[]
dis_list=[]

for i in range(1,96):
    truth=truth_list[i][:-1] #[:-1]去除换行符
    test=test_list[i][:-1]
    H_W=H_W_list[i][:-1]
    p1,p2=list(map(int,truth.split(' ')[1].split(','))), list(map(int,truth.split(' ')[2].split(',')))
    p3,p4 = list(map(int, test.split(' ')[1].split(','))), list(map(int, test.split(' ')[2].split(',')))
    H,W = list(map(int,H_W.split(" ")))
    #print(i+1)
    Angle,A_min,Dis,D_Min=loss1(p1+p2,p3+p4,H,W)
    loss_file.write('{:2} {:6} {:2} {:6} {:6}\n'.format(i+1,Angle,A_min, Dis,D_Min))
    if Angle <= A_min:
        sum1=sum1+1
        angle_list.append(i+1)
    if Dis <= D_Min:
        sum2=sum2+1
        dis_list.append(i + 1)
print(sum1,sum2)
print(angle_list)
print(dis_list)

easy=[4,3,9,8,11, 13, 33 ,19,25,27, 29,30,40,41,45, 51,57,79,89,93]
normal=[2,12,17,21,34, 43,46,52, 64 ,61, 72,75,77,81,83, 85,87,94,95,96]
hard=[5,6,7,10,26, 28,31, 14 ,35,49, 53,54, 56 ,68,74, 80,84,86,90,92]
easy_sum_angle=0
normal_sum_angle=0
hard_sum_angle=0
easy_sum_dis=0
normal_sum_dis=0
hard_sum_dis=0
easy_angle, easy_dis=[],[]
normal_angle, normal_dis=[],[]
hard_angle, hard_dis=[],[]
for i in easy:
    if i in angle_list:
        easy_sum_angle+=1
        easy_angle.append(i)
    if i in dis_list:
        easy_sum_dis+=1
        easy_dis.append(i)
for i in normal:
    if i in angle_list:
        normal_sum_angle+=1
        normal_angle.append(i)
    if i in dis_list:
        normal_sum_dis+=1
        normal_dis.append(i)
for i in hard:
    if i in angle_list:
        hard_sum_angle+=1
        hard_angle.append(i)
    if i in dis_list:
        hard_sum_dis+=1
        hard_dis.append(i)

print(easy_sum_angle,normal_sum_angle,hard_sum_angle)
print(easy_sum_dis,normal_sum_dis,hard_sum_dis)
print(easy_angle)
print(easy_dis)#无14 40
print(normal_angle)#无52 56
print(normal_dis)#无17 43 56 61
print(hard_angle)#无33 54 64 68
print(hard_dis)#无33 68 90