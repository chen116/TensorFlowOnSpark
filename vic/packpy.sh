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
--jars hdfs:///user/${USER}/tensorflow-hadoop-1.10.0.jar \
TensorFlowOnSpark/examples/mnist/mnist_data_setup.py \
--output mnist/tfr \
--format tfr

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
hadoop fs -cp temp/part-00004 stream_data
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
--model mnist_model_nos2

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
--images mnist/csv/test/images \
--labels mnist/csv/test/labels \
--mode inference \
--model mnist_model_nos2 \
--output predictions_nos2

 hdfs dfs -rm -r /user/root/predictions_nos ; spark-submit  --master yarn --deploy-mode client --queue ${QUEUE} --driver-class-path ~/hpc/sparklens.jar --py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/spark/mnist_dist.py --conf spark.yarn.maxAppAttempts=1 --archives hdfs:///user/root/Python.zip#Python --conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS TensorFlowOnSpark/examples/mnist/spark/mnist_spark.py --images mnist/csv/test/images --labels mnist/csv/test/labels --mode inference --model mnist_model_nos --output predictions_nos


--keras mnist/keras




spark-submit \
--master yarn \
--py-files TensorFlowOnSpark/tfspark.zip \
/user/root/${TFoS_HOME}/examples/mnist/mnist_data_setup.py \
--output ${TFoS_HOME}/mnist/csv \
--format csv


${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 12G ${MASTER}


spark-submit \
--master yarn \
--py-files TensorFlowOnSpark/tfspark.zip \
--archives hdfs:///user/root/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
TensorFlowOnSpark/examples/mnist/keras/mnist_mlp_estimator.py \
--input_mode spark \
--images hdfs://gpu2:9000/user/root/mnist/csv/train/images \
--labels hdfs://gpu2:9000/user/root/mnist/csv/train/labels \
--epochs 1 \
--model_dir hdfs://gpu2:9000/user/root/mnist_model_keras \
--tensorboard


spark-submit \
--master yarn \
--py-files TensorFlowOnSpark/tfspark.zip \
--archives hdfs:///user/root/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
TensorFlowOnSpark/examples/mnist/keras/mnist_mlp_estimator.py \
--input_mode tf \
--model_dir hdfs://gpu2:9000/user/root//mnist_model_keras_tfr \
--epochs 5 \

spark-submit \
--master yarn \
--py-files TensorFlowOnSpark/tfspark.zip \
--archives hdfs:///user/root/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_HDFS \
${TFoS_HOME}/examples/mnist/keras/mnist_mlp_estimator.py \
--input_mode tf \
--model_dir mnist_model_keras_tf  \
--epochs 5 \
--tensorboard

spark-submit --master yarn \
--py-files TensorFlowOnSpark/tfspark.zip \
--archives hdfs:///user/root/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
--jars hdfs:///user/${USER}/tensorflow-hadoop-1.10.0.jar \
TensorFlowOnSpark/examples/mnist/keras/mnist_inference.py \
--images_labels hdfs://gpu2:9000/user/root/mnist/tfr/test \
--export hdfs://gpu2:9000/user/root/mnist_model_keras_tfr/export/serving/* /  \
--output hdfs://gpu2:9000/user/root/predictions_keras/


normal yarn non streaming

spark-submit \
--master yarn \
--deploy-mode client \
--queue ${QUEUE} \
--py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/tf/mnist_dist.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs:///user/${USER}/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
TensorFlowOnSpark/examples/mnist/tf/mnist_spark.py \
--images mnist/tfr/train \
--format tfr \
--mode train \
--model mnist_model_tfr

spark-submit \
--master yarn \
--deploy-mode client \
--queue ${QUEUE} \
--py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/tf/mnist_dist.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs:///user/${USER}/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
TensorFlowOnSpark/examples/mnist/tf/mnist_spark.py \
--images mnist/tfr/test \
--mode inference \
--model mnist_model_tfr \
--output predictions_tfr