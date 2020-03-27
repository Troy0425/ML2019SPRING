out = open("./out.csv", "w")
out.write("patientId,x,y,width,height,Target\n")

inp = open("./ans.csv", "r")
inp.readline()
for line in inp:
	arr = line.split(",")
	img = arr[0]
	pixel = arr[1].split(" ")
	total = len(pixel) // 5
	i = 0
	if (total == 0):
		out.write(img+".png,,,,,0\n")
	else:
		for i in range(total):
			out.write(img+".png,")
			for j in range(1,5):
				out.write(pixel[j+i*5] + ",")
			out.write("1\n")


