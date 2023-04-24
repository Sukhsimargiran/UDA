# Unsupervised Data Augmentation (UDA) - Upgraded Version

This repository contains an upgraded version of the [Unsupervised Data Augmentation (UDA)](https://github.com/google-research/uda) project developed by Google Research. The upgraded version includes changes to various model parameters, such as learning rate, batch size, number of epochs, dropout rate, number of layers, activation functions, and weight initialization, to explore their impact on the model's performance.

## Requirements

To run the upgraded version of the UDA project, you will need the following:

- Python 3.6+
- TensorFlow 2.0+
- NumPy
- Matplotlib

You can install the required Python packages using the following command:

```
pip install -r requirements.txt
```

## Usage

To train the UDA model with the upgraded version, you can use the `train.py` script located in the `uda/` directory. The script takes several command-line arguments, including the dataset to use, the number of labeled examples, and the type of data augmentation to perform. For example, to train the model on the CIFAR-10 dataset with 4000 labeled examples and Mixup data augmentation, you can run the following command:

```
python uda/train.py --dataset cifar10 --num-labeled-examples 4000 --augment mixup
```

The script will download the dataset, split it into labeled and unlabeled examples, and train the UDA model with the upgraded parameters. The trained model will be saved in the `uda/models/` directory, and the training and validation metrics will be plotted in the `uda/results/` directory.

## Results

The upgraded version of the UDA project includes changes to various model parameters to explore their impact on the model's performance. The specific results may vary depending on the dataset, number of labeled examples, and type of data augmentation used. However, in general, the changes can lead to faster training, better generalization performance, or both.

The results of the experiments conducted with the upgraded version of the UDA project are included in the `uda/results/` directory. The results include the training and validation metrics for each experiment, as well as the plots of the metrics over time.

## Acknowledgments

The upgraded version of the UDA project was developed by Sukhsimar Singh Giran and Parin Mandavia, based on the original UDA project developed by Google Research. The project was developed as part of Artificial Intelligence.

## References

[1] Xie, Q., Dai, Z., Hovy, E., Luong, M., & Le, Q. V. (2019). Unsupervised Data Augmentation. arXiv preprint arXiv:1904.12848.

[2] Unsupervised Data Augmentation - Google Research. https://github.com/google-research/uda
