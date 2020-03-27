cd ..
python keras-retinanet/keras_retinanet/bin/_train.py --backbone=resnet101 --batch-size=8 --steps=500 --epochs=55 --image-min-side=412 --random-transform csv data/annotations data/classes #--val-annotations=data/val_annotations

echo "If you modified the image-min-side parameter, please also modified the SIZE_OF_IMAGE in predict.sh OuO"