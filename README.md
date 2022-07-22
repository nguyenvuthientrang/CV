# CV
This [repository](https://github.com/nguyenvuthientrang/CV) contains all of our code for DSAI | K64 | HUST | Computer Vison project.

## Dataset

To download the dataset, run this code:

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
mkdir PascalVOC12
mv VOCdevkit/VOC2012/* PascalVOC12
cd PascalVOC12
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug_Visualization.zip
wget http://cs.jhu.edu/~cxliu/data/list.zip
unzip SegmentationClassAug.zip
unzip SegmentationClassAug_Visualization.zip
unzip list.zip
mv list splits
```
Refactor the dataroot as follow:

```
data_root/
    --- VOC2012/
        --- Annotations/
        --- ImageSet/
        --- JPEGImages/
        --- SegmentationClassAug/
        --- saliency_map/
    --- ADEChallengeData2016
        --- annotations
            --- training
            --- validation
        --- images
            --- training
            --- validation
```

Download [SegmentationClassAug](https://drive.google.com/file/d/17ylg3RHZCQRyGVk6rcmkAjcMi6jeuXLr/view?usp=sharing) and [saliency_map](https://drive.google.com/file/d/1NDPBKbg5aoCismuU9R_IJ9cJp5ncww-M/view?usp=sharing)

## Perform Training

Clone the needed branch and customize the command:
```
python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0,1 --crop_val --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly --pseudo --pseudo_thresh ${THRESH} --freeze --bn_freeze --unknown --w_transfer --amp --mem_size ${MEMORY}
```

To run the original version (network frozen version), clone the branch main and run this command:
```
$ python main.py --data_root [your data root] --model deeplabv3_resnet101 --gpu_id 0,1 --crop_val --lr 0.01 --batch_size 16 --train_epoch 50 --loss_type bce_loss --dataset voc --task 5-5 --overlap --lr_policy poly --pseudo --pseudo_thresh 0.7 --freeze --bn_freeze --unknown --w_transfer --amp --mem_size 100 --val_interval 100
```

To run the variant version (network masking using hard attention), clone the branch hat and run this command:
```
$ python main.py --data_root [your data root] --model deeplabv3_resnet101 --overlap --gpu_id 0,1 --crop_val --lr 0.01 --bb_lr 0.0001 --batch_size 16 --train_epoch 50 --loss_type bce_loss --dataset voc --task 2-1 --lr_policy poly --pseudo --pseudo_thresh 0.7 --freeze --bn_freeze --unknown --w_transfer --amp --mem_size 100 --val_interval 100 --lamb 1
```

#### References
Our implementation is based on
[clovaai/SSUL](https://github.com/clovaai/SSUL), [k-gyuhak/CLOM](https://github.com/k-gyuhak/CLOM)
