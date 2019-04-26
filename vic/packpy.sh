export PYTHON_ROOT=~/Python
curl -O https://www.python.org/ftp/python/3.5.7/Python-3.5.7.tgz
tar -xvf Python-3.5.7.tgz
rm Python-3.5.7.tgz

# compile into local PYTHON_ROOT
pushd Python-3.5.7
./configure --prefix="${PYTHON_ROOT}" --enable-unicode=ucs4
make
make install
popd
rm -rf Python-3.5.7

# install pip
pushd "${PYTHON_ROOT}"
curl -O https://bootstrap.pypa.io/get-pip.py
bin/python3 get-pip.py
rm get-pip.py
bin/pip3 install pydoop
bin/pip3 install tensorflow
bin/pip3 install tensorflowonspark
zip -r Python.zip *

# Note: add any extra dependencies here, e.g.
# ${PYTHON_ROOT}/bin/pip install pydoop
popd

hdfs dfs -put ${PYTHON_ROOT}/Python.zip 

export PYTHON_ROOT=~/Python
curl -O https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz
tar -xvf Python-3.6.8.tgz


# compile into local PYTHON_ROOT
pushd Python-3.6.8
./configure --prefix="${PYTHON_ROOT}" --enable-unicode=ucs4
make
make install
popd


# install pip
pushd "${PYTHON_ROOT}"
curl -O https://bootstrap.pypa.io/get-pip.py
bin/python3 get-pip.py
rm get-pip.py
bin/pip3 install pydoop
bin/pip3 install tensorflow
bin/pip3 install tensorflowonspark
zip -r Python.zip *

# Note: add any extra dependencies here, e.g.
# ${PYTHON_ROOT}/bin/pip install pydoop
popd

hdfs dfs -put ${PYTHON_ROOT}/Python.zip 






spark-submit \
--master yarn \
--queue ${QUEUE} \
--driver-class-path ~/hpc/sparklens.jar \
--deploy-mode client \
--num-executors 4 \
--executor-memory 4G \
--archives hdfs:///user/root/Python.zip#Python,mnist/mnist.zip#mnist \
TensorFlowOnSpark/examples/mnist/mnist_data_setup.py \
--output mnist/csv2 \
--format csv2

hdfs dfs -rm -r mnist_model stream_data temp

hdfs dfs -mkdir stream_data 



spark-submit \
--master yarn \
--deploy-mode client \
--driver-class-path ~/hpc/sparklens.jar \
--queue ${QUEUE} \
--py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/streaming/mnist_dist.py \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.streaming.stopGracefullyOnShutdown=true \
--archives hdfs:///user/${USER}/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
TensorFlowOnSpark/examples/mnist/streaming/mnist_spark.py \
--images stream_data \
--format csv2 \
--mode train \
--model mnist_model

hdfs dfs -mkdir temp
hadoop fs -cp mnist/csv2/train/* temp
hadoop fs -cp temp/part-00000 stream_data
hadoop fs -cp temp/part-00001 stream_data
hadoop fs -cp temp/part-00002 stream_data
hadoop fs -cp temp/part-00003 stream_data
hadoop fs -cp temp/part-00004 stream_data
hadoop fs -cp temp/part-00005 stream_data
hadoop fs -cp temp/part-00006 stream_data
hadoop fs -cp temp/part-00007 stream_data
hadoop fs -cp temp/part-00008 stream_data
hadoop fs -cp temp/part-00009 stream_data


spark-submit \
--master yarn \
--deploy-mode client \
--driver-class-path ~/hpc/sparklens.jar \
--queue ${QUEUE} \
--py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/streaming/mnist_dist.py \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs:///user/${USER}/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
TensorFlowOnSpark/examples/mnist/streaming/mnist_spark.py \
--images stream_data \
--format csv2 \
--mode inference \
--model mnist_model \
--output predictions/batch

hadoop fs -rm -r -skipTrash predictions/* stream_data/* temp
hadoop fs -mkdir temp 
hadoop fs -cp mnist/csv2/test/* temp

hadoop fs -cp temp/part-00000 stream_data
hadoop fs -cp temp/part-00001 stream_data
hadoop fs -cp temp/part-00002 stream_data
hadoop fs -cp temp/part-00003 stream_data
hadoop fs -mv temp/part-00004 stream_data
hadoop fs -cp temp/part-00005 stream_data
hadoop fs -cp temp/part-00006 stream_data
hadoop fs -cp temp/part-00007 stream_data
hadoop fs -cp temp/part-00008 stream_data
hadoop fs -cp temp/part-00009 stream_data



823) listening for reservations at ('10.1.4.4', 35717)


to stop:

python3
import sys
import tensorflowonspark.reservation as reservation
host = sys.argv[1]
port = int(sys.argv[2])
addr = (host, port)
client = reservation.Client(addr)
client.request_stop()
client.close()

# no streaming
spark-submit \
--master yarn \
--deploy-mode client \
--queue ${QUEUE} \
--num-executors 4 \
--executor-memory 4G \
--archives hdfs:///user/${USER}/Python.zip#Python,mnist/mnist.zip#mnist \
TensorFlowOnSpark/examples/mnist/mnist_data_setup.py \
--output mnist/csv \
--format csv


spark-submit \
--master yarn \
--deploy-mode client \
--driver-class-path ~/hpc/sparklens.jar \
--queue ${QUEUE} \
--py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/spark/mnist_dist.py \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs:///user/root/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
TensorFlowOnSpark/examples/mnist/spark/mnist_spark.py \
--images mnist/csv/train/images \
--labels mnist/csv/train/labels \
--mode train \
--model mnist_model_nos

spark-submit \
--master yarn \
--deploy-mode client \
--driver-class-path ~/hpc/sparklens.jar \
--queue ${QUEUE} \
--num-executors 2 \
--executor-memory 25G \
--py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/spark/mnist_dist.py \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs:///user/root/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
TensorFlowOnSpark/examples/mnist/spark/mnist_spark.py \
--images mnist/csv/test/images \
--labels mnist/csv/test/labels \
--mode inference \
--model mnist_model_nos \
--output predictions_nos

 hdfs dfs -rm -r /user/root/predictions_nos ; spark-submit  --master yarn --deploy-mode client --queue ${QUEUE} --driver-class-path ~/hpc/sparklens.jar --py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/spark/mnist_dist.py --conf spark.yarn.maxAppAttempts=1 --archives hdfs:///user/root/Python.zip#Python --conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS TensorFlowOnSpark/examples/mnist/spark/mnist_spark.py --images mnist/csv/test/images --labels mnist/csv/test/labels --mode inference --model mnist_model_nos --output predictions_nos
