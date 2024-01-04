python -m hawp.ssl.predict --device cpu --ckpt checkpoints/hawpv3-imagenet-03a84.pth --threshold 0.2 --img $1 --saveto ./output/ --ext json --disable-show
