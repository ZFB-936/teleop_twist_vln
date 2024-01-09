#!/usr/bin/env python
#-*- coding: UTF-8 -*-

from __future__ import print_function
import sys, select, termios, tty
# sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
# sys.path.append('/home/zfb/catkin_ws/src/')
import roslib; roslib.load_manifest('teleop_twist')
import rospy
import PIL
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
#from yolov3_pytorch_ros.msg import BoundingBoxes
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge, CvBridgeError
import Tkinter as tk
from lib.utils import num2colors
from PIL import ImageTk
from PIL import Image as PIL_Image
from seq2seq import *
import time
import cv2
import numpy as np
import os
import re
import threading

msg = """
text: stop by the laptop, go to the microwave, go to the laptop, stop by the microwave
Action: forward,  backward, left, right
"""

bounding_boxes = []
cv_image_RGB = None
img0 = None
cv_image_god = None
img2 = None
instruction_count = 4
action_count = 0
cv_image_depth = None
instruction_path = None
text = None
dr = None

moveBindings = {
		'forward':(1,0,0,0),
		'backward': (-1, 0, 0, 0),
		'left': (0, 0, 0, 1),
		'right': (0, 0, 0, -1),
		}

contexts = {
		#'cup': (0, 0),
		'laptop: (0, 0),'
		}

def PublishInfoCallback(msg):
	global bounding_boxes
	#rospy.loginfo("this is Person Publisher messgae:[name:%s]", msg.bounding_boxes)
	bounding_boxes = msg.bounding_boxes

def callback_color(data):
	global cv_image_RGB
	global img0
	try:
		cv_image_RGB = CvBridge().imgmsg_to_cv2(data, "bgr8")
	except CvBridgeError as e:
		print(e)
	img0 = cv2.resize(cv_image_RGB, (1440, 810), interpolation=cv2.INTER_AREA)
	# print(img0.shape)
	img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGBA)
	img0 = PIL.Image.fromarray(img0)  # 将图像转换成Image对象
	img0 = PIL.ImageTk.PhotoImage(img0)
	img1.configure(image=img0)
	img1.image = img0
	img1.update()

def callback_god(data):
	global cv_image_god
	global img2
	try:
		cv_image_god = CvBridge().imgmsg_to_cv2(data, "bgr8")
	except CvBridgeError as e:
		print(e)


	cv_image_god = cv2.resize(cv_image_god, (320, 270), interpolation=cv2.INTER_AREA)
	cv_image_god = cv2.copyMakeBorder(cv_image_god, top=5, bottom=5, left=5, right=5,
									  borderType=cv2.BORDER_CONSTANT, value=[2, 2, 2])
	cv2.putText(cv_image_god, 'External Camera', (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	zeros = np.zeros((cv_image_god.shape), dtype=np.uint8)
	zeros_mask = cv2.rectangle(zeros, (25, 220), (300, 265), color=(51,0,255), thickness=-1)
	cv_image_god = cv2.addWeighted(cv_image_god, 1, zeros_mask, 0.5, 0.5)
	img2 = cv2.cvtColor(cv_image_god, cv2.COLOR_BGR2RGBA)
	img2 = PIL.Image.fromarray(img2)
	img2 = PIL.ImageTk.PhotoImage(img2)
	img3.configure(image=img2)
	img3.image = img2
	img3.update()

def callback_depth(data):
	global cv_image_depth
	
	cv_image_depth = CvBridge().imgmsg_to_cv2(data, "passthrough")
	cv_image_depth = np.array(cv_image_depth, dtype=np.float32)
	cv2.normalize(cv_image_depth, cv_image_depth, 0, 1, cv2.NORM_MINMAX)
	cv_image_depth = cv_image_depth
	#cv_image = cv2.resize(cv_image_depth, (640, 480), interpolation=cv2.INTER_AREA)
	# depth_array = np.array(cv_image_depth, dtype=np.uint16)
	# count +=1
	# depth_csv = str(count) + ".csv"
	# np.savetxt(depth_csv, cv_image_depth, delimiter=',')
	# cv2.namedWindow("Image window", 0);
	# cv2.resizeWindow("Image window", 224, 224);
	# cv2.imshow("Image window", cv_image_depth)
	# cv2.waitKey(3)

def Send_text(ipt):  # button函数方法
	global bounding_boxes, instruction_path, instruction_count, action_count, text
	settings = termios.tcgetattr(sys.stdin)
	pub = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
	rospy.init_node('teleop_twist')
	text = ipt.get()  # 获取输入的值
	print(text, end='\t')
	# instruction_path = "./data/image/" + text + '_' + str(instruction_count)
	# mkdir(instruction_path)  # makedirs 创建文件时如果路径不存在会创建这个路径
	# print(instruction_path)
	# instruction_count += 1
	# action_count = 0
	text = re.sub(r"([,.!?])", r" ", text)
	text = text.replace("  ", " ")
	evaluate_text_t = threading.Thread(target=evaluate_text)
	evaluate_text_t.start()

	# with open(instruction_path + "/instruction.txt", "wb") as f:
	# 	f.write(text)
	# 	f.write("\t")

# bypass the bucket go to the microwave
# bypass the bucket stop by the microwave
# from the left side of the bucket, go to the microwave

def action_move(action):
	if action in moveBindings.keys():
		x = moveBindings[action][0]
		th = moveBindings[action][3]
		print(action, end=' ')
		for i in range(20):
			twist = Twist()
			twist.linear.x = x * 0.1
			twist.angular.z = th * 0.18
			pub.publish(twist)
			time.sleep(0.1)

def ward(ipt):  # button 函数方法

	pub = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
	rospy.init_node('teleop_twist')
	x = 0
	th = 0
	Action = ipt

	if Action in moveBindings.keys():
		x = moveBindings[Action][0]
		th = moveBindings[Action][3]

	for i in range(20):
		twist = Twist()
		twist.linear.x = x * 0.1
		twist.angular.z = th * 0.2
		pub.publish(twist)
		time.sleep(0.1)

	# with open(instruction_path + "/instruction.txt", "ab") as f:
	# 	f.write(Action)
	# 	f.write(" ")

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
	else:
		print("---  There is this folder!  ---")

def get_dateset():
	global instruction_path, action_count, dr
	now = time.strftime(" %H:%M:%S", time.localtime())
	file = instruction_path + "/" + str(action_count)

	mkdir(file)  # 调用函数
	cv2.imwrite(file +'/RGB' + '.jpg', cv_image_RGB) #

	depth_array = np.array(cv_image_depth, dtype=np.float32)
	cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
	cv2.imwrite(file +'/Deep' + '.jpg', depth_array * 255)

	np.savetxt(file +'/scan' + '.csv', dr, delimiter=',')

	action_count += 1

transf_RGB = transforms.Compose([
    #transforms.CenterCrop(270),
    transforms.ToTensor(),
])

def transform_image(image, c):
	if c == 1:
		image = cv2.resize(image, (360, 270), interpolation=cv2.INTER_AREA)
		image = PIL_Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		image_input = transf_RGB(image)
	else:
		image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
		transf = transforms.ToTensor()
		image_input = transf(image)  # tensor数据格式是torch(C,H,W)

	return image_input.unsqueeze(dim=0).to(device)

def evaluate(encoder, decoder, sentence, max_length=30):
	global cv_image_RGB, cv_image_depth, dr

	with torch.no_grad():

		input_tensor = tensorFromSentence(input_lang, sentence)
		input_length = input_tensor.size()[0]
		encoder_hidden = encoder.initHidden()
		decoder_hidden = decoder.initHidden()
		encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

		for ei in range(input_length):
			encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
			encoder_outputs[ei] += encoder_output[0, 0]
		decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
		decoder_hidden = encoder_hidden
		decoded_words = []
		for di in range(max_length):
			RGB = transform_image(cv_image_RGB, 1)
			Deep = transform_image(cv_image_depth, 0)
			scan = torch.tensor(dr, dtype=torch.float32, device=device).view(1, -1)
			decoder_output, decoder_hidden = decoder(
				decoder_input, decoder_hidden, encoder_hidden, RGB, Deep, scan)
			topv, topi = decoder_output.data.topk(1)
			if topi.item() == EOS_token:
				decoded_words.append('<EOS>')
				print('ok')
				break
			else:
				Action = output_lang.index2word[topi.item()]
				decoded_words.append(Action)
				action_move(Action)
				time.sleep(2)

			decoder_input = topi.squeeze().detach()

	return decoded_words

def evaluate_text():
	global text
	encoderC = torch.load('checkpoints/encoder.pt')
	decoderC = torch.load('checkpoints/decoder.pt')
	output_words = evaluate(encoderC, decoderC, text, 30)  # Je suis aveugle.
	return output_words

def scan_callback(data):
	global dr
	i, ar = 0, 0
	dr = np.zeros((1, 64), np.float32)
	for r in data.ranges:
		i += 1
		ar += r
		if i % 7 == 0:
			dr[0][int(i/7) - 1] = ar / 7
			ar = 0

	dr = dr[0][16:48]

	#dr = dr[0][16:48]
	# if scan_display % 100 == 0:
	# 	scan_display = 0
	# 	for c in range(1, 68):
	# 		colors = num2colors(dr[0][c])
	# 		t = tk.Label(window, text=" ", bg=colors)
	# 		#t = tk.Text(width=1, height=2, bg='#ed3232')
	# 		t.place(x=50 + c * 10, y=600)

	#print(len(data.ranges))
	# print(dr)

# def get_scan():  # button函数方法
# 	global dr
# 	np.savetxt('a.csv', dr, delimiter=',')
# 	my_matrix = np.loadtxt('a.csv', delimiter=',')
# 	print(my_matrix.size)
# 	print(my_matrix)

if __name__=="__main__":

	#settings = termios.tcgetattr(sys.stdin)
	pub = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
	rospy.init_node('teleop_twist')
	speed = rospy.get_param("~speed", 0.5)
	turn = rospy.get_param("~turn", 1.0)
	status = 0

	try:
		print(msg)
		while(1):
			window = tk.Tk()
			window.title('VLN')
			window.geometry('1480x920+300+200')  # 窗口大小
			img1 = tk.Label(window, image=img0, width=1440, height=810)
			img1.place(x=20, y=20)
			img3 = tk.Label(window, image=img2, width=330, height=280)
			img3.place(x=1100, y=20)

			ipt = tk.Entry(window, show=None, font=('华文行楷', 20), width=70)  # show代表显示，如果是输入密码的话show='*'，这样显示就是*号
			ipt.insert(0, "go to the microwave and turn right stop by the door")
			ipt.place(x=20, y=850)  #ipt.pack()  # 布局

			b = tk.Button(window, text='Send text', width=15, height=2, command=lambda: Send_text(ipt))  # 定义一个button，text为button，command为其绑定一个函数方法
			b.place(x=1200, y=850)

			# forward_B = tk.Button(window, text='forward', width=10, height=2, command=lambda: ward('forward'))
			# forward_B.place(x=1220, y=200)
			#
			# backward_B = tk.Button(window, text='backward', width=10, height=2, command=lambda: ward('backward'))
			# backward_B.place(x=1220, y=400)
			#
			# left_B = tk.Button(window, text='left', width=10, height=2, command=lambda: ward('left'))
			# left_B.place(x=1050, y=300)
			#
			# right_B = tk.Button(window, text='right', width=10, height=2, command=lambda: ward('right'))
			# right_B.place(x=1390, y=300)
			#
			# context_B = tk.Button(window, text='get context', width=10, height=2, command=lambda: Context())
			# context_B.place(x=1400, y=500)
			#
			# get_dateset_B = tk.Button(window, text='get image', width=10, height=2, command=lambda: get_dateset())
			# get_dateset_B.place(x=1100, y=500)
			#
			# get_scan_B = tk.Button(window, text='get scan', width=10, height=2, command=lambda: get_scan())
			# get_scan_B.place(x=1300, y=500)
			rospy.Subscriber('/kinect2/qhd/image_color', Image, callback_color)
			#rospy.Subscriber('/detections_image_topic', Image, callback_color)
			#rospy.Subscriber('/kinect2/hd/image_color', Image, callback_color)
			rospy.Subscriber('/kinect2/sd/image_depth', Image, callback_depth)
			rospy.Subscriber("/scan", LaserScan, scan_callback, queue_size=1)
			rospy.Subscriber("/usb_cam/image_raw", Image, callback_god)
			window.mainloop()  # 显示窗口

	except Exception as e:
		print(e)

	finally:
		twist = Twist()
		twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
		twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
		pub.publish(twist)
		#termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
