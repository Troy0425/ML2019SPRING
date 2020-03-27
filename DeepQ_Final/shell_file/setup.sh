cd ../src
pip install numpy --user
pip install -r ../requirements.txt
pip install . --user
python setup.py build_ext --inplace
cd ../data
python preprocess.py
