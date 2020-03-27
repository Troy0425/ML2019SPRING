import os , random
dir_path = os.path.dirname(os.path.realpath(__file__))

fp = open("train_labels.csv", 'r')
fp.readline()
out = open("annotations", 'w')
out2 = open("val_annotations", 'w')

res = 0


for line in fp:
	arr = line.split(",")
	path = dir_path+"/train/" + arr[0]
	if (arr[5] == "1\n"):
		for i in range(1,5):
			arr[i] = float(arr[i])
		x1,x2,y1,y2 = int(arr[1]),int(arr[1]+arr[3]),int(arr[2]),int(arr[2]+arr[4])
		if (res < 20998):
			out.write(path+",%d,%d,%d,%d,pneumonia\n"%(x1,y1,x2,y2))
		else:
			out2.write(path+",%d,%d,%d,%d,pneumonia\n"%(x1,y1,x2,y2))
	else:
		if (res < 20998):
			out.write(path+",,,,,\n")
		else:
			out2.write(path+",,,,,\n")
	res += 1



		


