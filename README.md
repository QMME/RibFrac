# RibFrac

  This is the project for EE228.

  Put the data in folder `input`.
  
  To run 
  
      python train.py --train_image_dir <path/to/train/images> --train_label_dir <path/to/train/labels> --val_image_dir <path/to/val/image> --val_label_dir <path/to/val/labels> --epoch 100 --save_dir checkpoints 
  
  To predict the results on certain dataset, run

      python predict.py --image_dir <path/to/image/folder> --model_path <path/to/model/dir> --pred_dir <path/to/store/results>

  To evaluate the results (if you have ground truth), run
      
      python evaluate.py --gt_dir <path/of/ground/truth> --pred_dir <path/of/pred/results> --clf False
      
  We have our pretrained model in folder `checkpoints`. The results of prediction on validation dataset are in `pred_val.zip`. The results of prediction on test dataset are in `pred_test.zip`. 
