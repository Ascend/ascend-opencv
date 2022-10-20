# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

for var in $@
do 
    if [ $var == "-x86" ]
        then
	    cmake .. -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-Bsymbolic
	else
	    cmake ..
    fi
done

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
