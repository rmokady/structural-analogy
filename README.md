# Structural-analogy from a Single Image Pair
Pytorch implementation for the paper "Structural-analogy from a Single Image Pair".

Abstract: The task of unsupervised image-to-image translation has seen substantial advancements in recent years through the use of deep neural networks.
Typically, the proposed solutions learn the characterizing distribution of two large,
unpaired collections of images, and are able to alter the appearance of a given
image, while keeping its geometry intact. In this paper, we explore the capabilities of neural networks to understand image structure given only a single pair
of images, A and B. We seek to generate images that are structurally aligned:
that is, to generate an image that keeps the appearance and style of B, but has
a structural arrangement that corresponds to A. The key idea is to map between
image patches at different scales. This enables controlling the granularity at which
analogies are produced, which determines the conceptual distinction between
style and content. In addition to structural alignment, our method can be used
to generate high quality imagery in other conditional generation tasks utilizing
images A and B only: guided image synthesis, style and texture transfer, text translation as well as video translation.

For more details, please refer to the Paper

## Code:

### Prerequisites:
Python 3.7, Pytorch 1.4.0, argparse, Pillow 7.0.0, Scipy 1.4.1, skimage 0.16.2, numpy

### Structural Analogy:
You can train using the following command:
```
python train.py --input_a ./208.jpg --input_b ./209.jpg --gpu_id 0 --out ./output0/ --beta 10.0 --alpha 1.0
```
For other images, just replace input_a and input_b.

## Citation
If you found this work useful, please cite.

## Contact
For further questions, ron.mokady@gmail.com or sagiebenaim@gmail.com.

## Acknowledgements
This implementation is heavily based on https://github.com/tamarott/SinGAN.




