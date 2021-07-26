# GraphMDN: Leveraging graph structure and deep learning to solve inverse problems

This code implements the work in our paper of the same title, which can be found on [arXiv](https://arxiv.org/abs/2010.13668). 

This code "repository" contains the code necessary to train and evaluate the GraphMDN model introduced in the accompanying text. The code will be made public at a later date via github repository.

This code extends the [SemGCN model](https://github.com/garyzhao/SemGCN/)  to incorporate MDN structured outputs, and uses data preprocessing code from [VideoPose3d](https://github.com/facebookresearch/VideoPose3D)

## Dataset setup
You can find the instructions for setting up the Human3.6M and results of 2D detections in [`data/README.md`](data/README.md). The code for data preparation is borrowed from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).

### Evaluating our pretrained model

To evaluate, run:
```
python main_eval.py --evaluate checkpoint/trained_model/ckpt_best.pth.tar
```

### Training from scratch
If you want to reproduce the results of our pretrained models, run the following commands.

```
python main_mdn.py
```

Additionaly training parameters and arguments can be seen in the common/arguments.py file.
