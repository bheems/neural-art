## Faster neural doodle

This is my try on drawing with neural networks, which is faster than [Alex J. Champandard's version](https://github.com/alexjc/neural-doodle), and similar in quality. This approach is based on [neural artistic style method](http://arxiv.org/abs/1508.06576) (L. Gatys), whereas Alex's version uses [CNN+MRF approach](http://arxiv.org/abs/1601.04589) of Chuan Li.

It takes several minutes to redraw `Renoir` example using GPU and it will easily fit in 4GB GPUs. If you were able to work with [Justin Johnson's code for artistic style](https://github.com/jcjohnson/neural-style) then this code should work for you too. 

## Requirements
- torch
- torch.cudnn (optional)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5)
- python + numpy + scipy + h5py + sklearn

Tested with python2.7 and latest `conda` packages.
## Do it yourself

First download VGG-19.
```
cd data/pretrained && bash download_models.sh && cd ../..
```

Use this script to get intermediate representations for masks. 
```
python get_mask_hdf5.py --n_colors=4 --style_image=data/Renoir/style.png --style_mask=data/Renoir/style_mask.png --target_mask=data/Renoir/target_mask.png
```

Now run doodle.
```
th fast_neural_doodle.lua -masks_hdf5 masks.hdf5 -vgg_no_pad
```

And here is the result.
![Renoir](data/Renoir/grid.png)
First row: original, second -- result.

And Monet.
![Renoir](data/Monet/grid.png)

## Misc
- Supported backends: 
	- nn (CPU/GPU mode)
	- cudnn
	- clnn (not tested yet..)
 
- When using `-backend cudnn` do not forget to switch `-cudnn_autotune`.

## Acknowledgement

The code is heavily based on [Justin Johnson's great code](https://github.com/jcjohnson/neural-style) for artistic style.