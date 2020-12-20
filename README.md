# nepali_asr
Nepali Automatic Speech Recognition using pytorch


usage: train.py [-h] --save_model_path SAVE_MODEL_PATH
                [--load_model_from LOAD_MODEL_FROM]
                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                [--load_x LOAD_X] [--load_y LOAD_Y] [--logdir LOGDIR]

optional arguments:
  -h, --help            show this help message and exit
  --save_model_path SAVE_MODEL_PATH
                        path to save model
  --load_model_from LOAD_MODEL_FROM
                        path to load a pretrain model to continue
                        training
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        check path to resume from
  --load_x LOAD_X       path to load a tensor x file
  --load_y LOAD_Y       path to load a tensor label file
  --logdir LOGDIR       path to save logs
