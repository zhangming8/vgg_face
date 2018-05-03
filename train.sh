/data/Experiments/caffe/build/tools/caffe train -solver solver_VGG_FACE.prototxt --weights=VGG_FACE.caffemodel -gpu all 2>&1 | tee logs/vgg_face.log
