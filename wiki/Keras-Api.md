## Table of Contents
* [Keras](#tensorflow)
   * [General](#general)
   * [Known issues](#known-issues)
   * [Regularization techniques](#regularization-techniques)
   * [Saving Models](#saving-models)
   * [Callbacks](#callbacks)
        * [Early Stopping](#early-stopping)
   * [Model Selection](#model-selection)
   * [Accuracy & Predictions](#accuracy--predictions)
   * [Models](#models)

## Keras
### General
* Bias is the constant term in y=mx+c. Where m is the weight or Gradient and c is the constant or the bias. A bias is
needed to shift the curve from the origin. This helps in fitting data across any region.

### Known issues
* The weights are randomized before we start training a NN. So the output is going to be different and the error for
train will also be different each we run NN. Some of ways to overcome this is using bootstrap aggregating, dropouts etc.

### Regularization techniques
* Weight sharing- as done in CNN's, applying the same filters across the image.

* Data Augmentation- Augmenting existing data and generate synthetic data with generative models

* Large amount of training data- thanks to ImageNet etc.

* Pre-training- For example say Use ImageNet learnt weights before training classifier on say Caltech dataset.

* The use of RelU's in Neural Nets by itself encourages sparsity as they allow for zero activations. In fact for more complex regions in feature space use more RelU's, deactivate them for simple regions. So basically vary model complexity based on problem complexity.

### Saving Models
Specifically in keras models can be saved in 3 formats viz. yaml, json and hdf5. hdf5 models are saved with weights,
for the others it is just string.
```python
model.save("path/to/file.yml/.h5/.json")
```
For serving models saved_model is used. The model is saved as a protobuf.
```python
from tensorflow import saved_model

saved_model.save(obj, "path/to/save", signatures, options)
```

### Callbacks
Used while model.fit(). For multiple functions such as:

#### Early Stopping
* Rather than deciding how many epochs we need to train the NN. We keep a separate validation set and train the NN such
that the val loss stops decreasing.
* Generally the training set has better loss compared to validation because the NN has those values and thus has learnt
those values.
```Python
from tensorflow.keras.callbacks import EarlyStopping
moniter = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(X, y, validation_data=(X_test, y_test), callbacks=[moniter], epochs=1000)
```
