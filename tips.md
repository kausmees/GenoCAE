# Tips on hyperparameter tuning and training setup


It's difficult to give rules of thumb for what will work in the optimization procedure of GCAE. We've found that it can be quite data-dependent, and are ourselves in the process of performing hyperparameter tuning for our application.

Below are some suggested values that might be good to start out with.

### Model architecture

The example model [models/M1.json](models/M1.json) is very similar to the one we have described in [1], which we have found works well for dimensionality reduction.

The table below lists some different architecture settings we have tried on the same basic model structure. These can be experimented with further, as well as changing the types of layers and their order.

We have had difficulty getting larger models (both with more layers and more convolutional filters) to work well on the data set in [1], but have by no means done an exhaustive evaluation yet.

Creating smaller models (less filters and more maxpooling in particular) might be a good option to obtain something that's faster to train. We've found that smaller, simpler models can give quite good results also, especially for smaller data sets.



#### Model architecture settings

| Parameter      | Values |
| ----------- | ----------- |
|kernel size convolution | 3, 5, 8, 10 |
|pool size maxpool | 2, 3, 5 |
|stride maxpool | 2, 3 |
|number of filters/kernels convolution | 8, 16, 32 |
|dropout rate | 0.0, 0.01, 0.1 |
|number of units in fully-connected layers  | 15, 25, 50, 75 |


    
### Hyperparameter values

The table below describes a hyperparameter space that might be good to begin in. Depening on your data, other values might give better results though.


| Parameter      | Values |
| ----------- | ----------- |
|learning rate | 0.00032, 0.001, 0.0032, 0.01 |
|exponential decay rate | 0.92, 0.94, 0.96 |
|decay every X epochs | 10, 30  |
|regularization factor | 1e-06, 1e-07, 1e-08 |
|noise std  |  0.01, 0.1, 1.0, 10|
|batch size  | 30, 45, 60, 100|



### Other tips

#### Loss function and normalization
In our experiments, the combination of **norm_mode = genotypewise01** in **data_opts** and loss function **CategoricalCrossentropy** (as in [train_opts/ex3.json](train_opts/ex3.json)) gives the best results.


#### Use the validation loss to get an idea of overfitting
Look at the validation loss, either on the generated plots or in tensorboard. If it starts to increase early, even though the train loss is decreasing, increasing the size of the regularization factor usually helps. I've found that the smaller the data set, particularly with few (< 1000) samples, the more models tend to overfit, and the larger regularization is required. Possibly several orders of magnitude larger than the ones in the table above. Increasing the noise, both **noise_std** in the **train_opts** and **sparsifies** in **data_opts** might also help.



## References

 * [1] Ausmees, Kristiina and Nettelblad, Carl. "A deep learning framework for characterization of genotype data." bioRxiv (2020). [here](https://www.biorxiv.org/content/10.1101/2020.09.30.320994v1.abstract)

