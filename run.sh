# train
conda activate {your env}

python train.py --config_file configs/RGBNT201/Signal.yml
python train.py --config_file configs/MSVR310/Signal.yml
python train.py --config_file configs/RGBNT100/Signal.yml

#test
conda activate {your env}

python test.py --config_file configs/RGBNT201/Signal.yml
python test.py --config_file configs/MSVR310/Signal.yml
python test.py --config_file configs/RGBNT100/Signal.yml