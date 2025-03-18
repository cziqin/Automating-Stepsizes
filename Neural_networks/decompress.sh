# run this script in the `*/data/imagenet` directory
cd ./data/imagenet
tar -xzf ILSVRC2012_devkit_t12.tar.gz

mkdir train && cd train
tar -xf ../ILSVRC2012_img_train.tar
for f in *.tar; do 
    mkdir -p "${f%.tar}";
    tar -xf "$f" -C "${f%.tar}";
    rm "$f";
done
cd ..

mkdir val && cd val
tar -xf ../ILSVRC2012_img_val.tar