TOOLS=/home/jhoffman/caffe-weak-fcn/build/tools
SOLVER=solver.prototxt
DATE=$(date +"%F-%T")
LOG=logs/log_file_${DATE}.txt

REF='/x/jhoffman/VGG16_full_conv.caffemodel'
CMD="$TOOLS/caffe.bin train -solver ${SOLVER} -weights ${REF} 2>$LOG" 
echo $CMD
echo ""
GLOG_logtostderr=1 ${TOOLS}/caffe.bin train -solver "${SOLVER}" -weights "${REF}" 2>"${LOG}"

echo "Done."


