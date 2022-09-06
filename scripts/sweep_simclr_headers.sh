# Strong augmentations
python src/experiment.py ; 
python src/experiment.py ++exp_name=SimCLR_Linear ++model_cfg.header_cfg.projection_type=Linear ; 
python src/experiment.py ++exp_name=SimCLR_None ++model_cfg.header_cfg.projection_type=None ; 
python src/experiment.py ++exp_name=SimCLR_RandomProjection ++model_cfg.header_cfg.projection_type=Direct ;
python src/experiment.py ++exp_name=SimCLR_RandomProjection ++model_cfg.header_cfg.projection_type=RandomProjection ;