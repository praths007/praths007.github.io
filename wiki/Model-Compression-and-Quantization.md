## Table of Contents

* [Why is model compression required?](#why-is-model-compression-required?)
* [Compression techniques for DNN](#compression-techniques-for-DNN)

## Why is model compression required?

Usually for on-edge-devices which are small and can fit very little computing power because of their size, model compression becomes necessary. Generally in the case of DNN model compression is of utmost importance. In the final scenario when the model is used to make predictions, it is called model inference.

After deployment a trade-of between some of the key performance metrics have to be measured:

Accuracy, latency, memory use, throughput, cost, model size etc.



## Compression techniques for DNN

Goal is to find a NN with same or similar accuracy by decreasing size of the model.

- Low-precision inference
  - Heuristically use lower precision numbers to represent weights and activations, like from 32bit to 8bit. 
  - Usage of scale quantization to lower precision. Convert tensors with decimals with 2 place precision to integer 
  values.
- Pruning
  - Remove activations that are zero or close to zero. A threshold can be used for doing this.
  - Eg. in Keras we can use layers.get_weight and layers.set_weight to get weights and quantize if required.
- Old school compression
  - Zipping with g-zip or using other lossless techniques to weights.
- Knowledge distillation
  - Use ensemble of smaller model to distil the output of the larger teacher model.





