## Table of contents

- [Serving using tensorflow](#serving-using-tensorflow)
    -[Mini batching](#mini-batching)
    -[Some positive points](#some-positive-points)
- [Edge device serving using tensorflowlite](#edge-device-serving-using-tensorflowlite)
- [Deploying models on AWS](#deploying-models-on-aws)


## Serving using tensorflow
This [video](https://youtu.be/264nTqrPCJQ) explains things in detail. The high level steps are as follows:
* Save the model
This is exporting model as a protobuf, which is it has log of all essences like variables etc.
```python
import tensorflow as tf

tf.save_model.save(model, expoert_dir="saved_models/", signatures=None)
```
* Use docker for serving
```bash
docker pull tensorflow/serving

docker run -p 84500:8500 saved_models/ /models/models -r MODEL_NAME=my_model \
-t tensorflow/serving

docker run -t tensorflow/serving:latest-gpu

```
Rest post requests can be made to the exposed port to receive the required data in json format.
There is another method using gRPC which is more efficient.

REST api can also be implemented in flask.

### Mini batching
Mini batching is the key feature of tensorflow serving which is difficult to do in Flask.
When multiple clients keep requesting the server, tensorflow serving puts them in a mini batch.

### Some positive points
* Automatic update of model load and unload because there is a 2 second polling.
* NVIDIA TensorRT used to quantize from float32 to float8 during inference.This can be added as an option.


## Edge device serving using tensorflowlite
Positive points of tflite <br>
* New lighter version of tensorflow where models can be saved as TFlite flatbuffer using TFlite.
* Support for keras API.
* It is cross platform so can be used for android as well as iOS.

We might not want to score model on the cloud. Transfer restrictions etc. we dont want cloud to do everything.




## Deploying models on AWS
We can save the weights and json separately.
Weights can be saved as h5 model.save_weights(".h5") to save the weights and then model.to_json(".json") to save the 
 structure. <br>
Formats are:
* yaml - structure only
* json - structure only
* h5 - structure only
* h5 - weights only
* A protocol buffer or protobuf is a format used by google for deploying models.

### Things required for AWS deployment
```python
import boto3 #interface into sagemaker from python

role = get_execution_role()
```
AWS requires direct keras. Not the one inside tensorflow.
Apart from this we must also consider the following for exporting the models.
```python
model_version = "1"
export_dir = "export/Servo" + model_version
```
then create the build
```python
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

build  = builder.SavedModelBuilder(exportdir)

signature = predict_signature_def(inputs={"inputs":loaded_model.input}, outputs = {"score":loaded_model.output})
```
This saves as a .pb or protobuf. It is going to create a whole directory essentially.

```python
from keras import backend as K
with K.get_session() as sess:
    build.add_meta_graph_and_variables(
    sess=sess, tags=[tags_constants.SERVING, signature_def_map={"serving_default":signature}]    
)
    build.save()
```
Tar the complete directory and upload to S3
```python
import sagemaker

sagemaker_session = sagemaker.Session()
inputs = sagemaker_session.upload_data(path="model.tar.gz", key_prefix="model")
```

Actual deployment. <br>
There is a bug in sagemaker and entrypoint needs an empty train.py file.
```python
!touch train.py

from sagemaker.tensorflow.model import TensorFlowModel
sagemaker_model = TensorflowModel(model_data="s3://"+sagemaker_session.default_bucket()+"model/model.tar.gz",
role=role,
framework_version="1.12",
entry_point="train.py")
```

```python
predictor = sagemaker_model.deploy(initial_instance_count=1,
instance_type="ml.m4.xlarge")
```

```python
predictor.endpoint
```

Using the deployed model

```python
endpoint_name="output from above command"
import sagemaker
from sagemaker.tensorflow.model import TensorFlowModel
predictor = sagemaker.tensorflow.model.TensorFlowPredictor(endpoint_name, sagemaker_session)

data = train_X[0:]

# to run stuff on local (inside AWS)
client = boto3.client("runtime.sagemaker")
response = client.invoke_endpoint(EndpointName=endpoint_name, Body=json.dumps(data))
response_body = response["Body"]


# outside AWS
boto3.client("runtime.sagemaker", region_name="us-east-1",
aws_access_key_id="",
aws_secret_access_key="")
# these access keys comes from IAM (identity and access management)
```