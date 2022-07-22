#! /bin/bash
opencv_directory=./opencv
acl_directory=./acl

if [ -d $opencv_directory -o -d $acl_directory ]
then
    mv $acl_directory $opencv_directory/modules/
fi

cd $opencv_directory

build_directory=$PWD/build
if [ ! -d $build_directory ]
then
    mkdir -p build
fi
cd build

cmake ..
make -j
while [ $? != 0 ]
do
    make -j
done

for var in $@
do 
    if [ $var == "ACLTEST" ]
    then
        cd bin
        ./opencv_test_acl
    fi
done