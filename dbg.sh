python -m hawp.ssl.predict --device cpu --ckpt checkpoints/hawpv3-imagenet-03a84.pth --threshold 0.02 --img duh.jpg --saveto ./ --ext json --disable-show && open duh.png
