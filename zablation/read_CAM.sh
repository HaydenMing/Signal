1. yml 修改 
    TEST:IMS_PER_BATCH: 64

2. make_model.py
    cam_label = x['cam_label']
    view_label = x['view_label']

3. bash 

conda activate envs
python CAM.py --config_file configs/RGBNT201/GMR.yml