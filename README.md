# Poisson Blending with adapted gradients

This library is a simple demonstration of using Optimal Transport Mapping Estimation [1]
in a context of seamless copy between images. See [1] for details over the method

## Installation

The Library has been tested on Linux and MacOSX. Among other classical dependencies, it requires the installation of POT, the Python Optimal Transport library (https://github.com/rflamary/POT)

- Numpy (>=1.11)
- Scipy (>=0.17)
- Matplotlib (>=1.5)
- Pyamg (>=3.1)
- POT (>=1.0)

If you want to execute the video demo, then you also need to have OpenCV for python installed.



## Examples

One notebook is provided as example of use:

* [Example](https://github.com/ncourty/PoissonGradient/blob/master/test.ipynb)

The video demo is available in the test_video.py file.




## References


[1] M. Perrot, N. Courty, R. Flamary, A. Habrard, "Mapping estimation for discrete optimal transport", Neural Information Processing Systems (NIPS), 2016.
