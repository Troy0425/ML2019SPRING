if [ "$#" -ne 1 ]; then
	echo "Usage : Select one model from ../model"	
	exit 0
fi


SIZE_OF_IMAGE=412

python ../src/keras_retinanet/bin/convert_model.py $1 ../src/model1.h5

cd ../src

python predict.py "$SIZE_OF_IMAGE"
python trans.py
python bbox_to_rle.py

rm out.csv
rm ans.csv
mv overall.csv ../ans.csv
rm ./model1.h5
