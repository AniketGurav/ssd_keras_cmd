# keras-frcnn
A command line version of https://github.com/rykov8/ssd_keras

## Papers and ideas
Single shot multi-box detector (SSD: https://arxiv.org/pdf/1512.02325.pdf) is one of the state-of-the-art techniques for object detection. The technique uses multi-level aggressive feature maps simultaneously to train and predict object bboxes and classes, which is efficient and  highly adaptive to object sizes. However, in order to accelarate the algorithm, the default prior boxes are pre-computed, which limits the input image size to some constant (300*300 in this repository) and the network structure cannot be modified conveniently. If it is a big probelm for you, I recommend RetinaNet (https://arxiv.org/pdf/1708.02002.pdf) which can achieve similar results and not suffer from those limitations.

## Requirements
* Keras 2
* Python 2
* Opencv 2+

## Pretrained models
* https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA

## Training
* The data should be transformed to VOC2012 format for training.

## How to train
* Simply specify the parameters
```python
python train.py --pkl_path [path to .pkl file] --input_path [path to training images]
```

## How to test
* Simply specify the test data path
```python
python predict.py --model_path [path to model] --input_path [path to test images]  --output_path [path to results]
```

## Prediction results
<p>
  <img src="https://github.com/shuuchen/ssd_keras/blob/master/predict_results/69.jpg" height="432" width="432" />
</p>

## Any questions
Please feel free to contact me if you have any questions. Wechat users can add me via dushuchen1022wind for direct communication.

## Thanks
https://github.com/rykov8/ssd_keras

## License
* Released under the MIT license. See LICENSE for details.
