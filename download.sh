
FILE=$1

if [ $FILE == "data" ]; then

    # CelebA images and attribute labels
    URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
    ZIP_FILE=./data/celeba.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/
    rm $ZIP_FILE

if [ $FILE == "data" ]; then

    # pretrained model
    URL=https://www.dropbox.com/s/n1vxfdlbrbt4gk4/pretrained.zip?dl=0
    ZIP_FILE=./model/pretrained.zip
    mkdir -p ./model/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./model/
    rm $ZIP_FILE
