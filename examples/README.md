### Examples

#### 1. Evolving a target bitmap

The goal here is to evolve a target bitmap image that is not known. The learning signal is the similarity between any image and the target. the fitness function was defined to return this similarity measure. The population evolves over time to ultimately generate the target image.

Target image               |  Evolved image
:-------------------------:|:-------------------------:
<img src="https://github.com/madvn/TFSearch/blob/master/examples/evolImage/targetImage.bmp" width="150"/>  |  <img src="https://github.com/madvn/TFSearch/blob/master/examples/evolImage/bestEvolvedImage.bmp" width="150"/>



#### 2. Evolving to generate neural network oscillators

The goal here is to optimize CTRNNs to have intrinsic oscillatory dynamics. The fitness function was defined to measure the average absolute difference in neural network activity in consecutive time-steps during the course of a simulation.

Fitness over time               |  Evolved oscillator
:-------------------------:|:-------------------------:
<img src="https://github.com/madvn/TFSearch/blob/master/examples/CTRNN_oscillator/results/fitsvsGens.png" width="600"/>  |  <img src="https://github.com/madvn/TFSearch/blob/master/examples/CTRNN_oscillator/results/outputs.png" width="600"/>


#### 3. Evolving neural networks to discriminate MNIST

The goal here is to classify the MNIST digits. The fitness function was defined as a network's performance on a random batch of 20000 samples from the dataset.
