## Table of contents

- [Serving using tensorflow](#serving-using-tensorflow)
    -[Mini batching](#mini-batching)
    -[Some positive points](#some-positive-points)
- [Edge device serving using tensorflowlite](#edge-device-serving-using-tensorflowlite)

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
* New lighter version of tensorflow where models can be saved as TFlite flatbuffer using TFlite.
* Support for keras API.
