python -m hawp.ssl.predict --device cpu --ckpt checkpoints/hawpv3-imagenet-03a84.pth --width 512 --height 512 --cta $3 --threshold $4 --img $1 --saveto $2 --ext json --disable-show
