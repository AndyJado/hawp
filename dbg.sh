python -m hawp.ssl.predict --device cpu --ckpt checkpoints/hawpv3-imagenet-03a84.pth --cta $3 --threshold $2 --img $1 --saveto ./test/ --ext json --disable-show
