
# -*- coding: utf-8 -*-
colors = ['#540d0d' ,'#9c1414' ,'#d92727' ,'#ed3232' ,'#f27777' ,'#f7adad']

def num2colors(num):
	if num < 0.5:
		return colors[0]
	elif num < 1:
		return colors[1]
	elif num < 1.5:
		return colors[2]
	elif num < 2:
		return colors[3]
	elif num < 2.5:
		return colors[4]
	else:
		return colors[5]

'''
def get_deep(xmin, xmax, ymin, ymax):
	global cv_image_depth
	depth = 0
	x = (xmin + xmax) / 2
	y = (ymin + ymax) / 2
	deep = cv_image_depth[y, x]
	for j in range((y - 5) , (y + 6)):
		for i in  range((x - 5) , (x + 6)):
			depth += cv_image_depth[j, i]
	deep = depth / 121
	return deep


def Context():  # button函数方法

	global bounding_boxes
	context_vetor = [0] * 4
	context_str = "0 0 0 0"
	rospy.Subscriber("/detected_objects_in_image", BoundingBoxes, PublishInfoCallback)
	for bounding_boxe in bounding_boxes:
		c = bounding_boxe.Class
		if c in contexts.keys():
			c = contexts[c][0]
			if c == 0:

				context_vetor[0 + 4 * c] = 1
				context_vetor[1 + 4 * c] = (bounding_boxe.xmin + bounding_boxe.xmax) / 2
				context_vetor[2 + 4 * c] = (bounding_boxe.ymin + bounding_boxe.ymax) / 2
				context_vetor[3 + 4 * c] = get_deep(bounding_boxe.xmin, bounding_boxe.xmax, bounding_boxe.ymin,
													bounding_boxe.ymax)
			context_str = str(context_vetor[0 + 4 * c]) + ' ' + str(context_vetor[1 + 4 * c]) + ' ' + str(context_vetor[2 + 4 * c]) + ' ' + str(context_vetor[3 + 4 * c])

	print(context_str)
	return context_str

'''