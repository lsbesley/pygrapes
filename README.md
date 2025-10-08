# PyGRAPES
PyGRAPES (**Py**torch **gra**zing incidence **p**tychographic **e**ngine for **s**urfaces and nanostructures) is a versatile pytorch-based multislice framework for 3D reconstruction and simulation of ptychographic data in grazing incidence.


PyGRAPES is best suited for grazing incidence and reflection geometry ptychographic datasets, supports multiple incidence angles, and multiple rotational angles about the surface normal, allowing users to specify complex imaging geometries and perform iterative reconstructions from ptychographic datasets. The framework is written in mind for ease of use and accessibility for ptychographic reconstruction, being able to be run from a jupyter notebook (with a demo notebook provided) or from a simple and small set of python commands. 

PyGRAPES is tested on GPU and is strongly reccomended to run on GPU, but should run on CPU. 

<img src="documentation/GIXP_Geom_Diags.png" alt="Geometries in grazing incidence for 3D reconstruction" width="600"/>



## Installation
You can either clone the repository, or run:

```bash
pip install git+https://github.com/lsbesley/pygrapes.git
```
## Demos

You can run Demo_sim_Script_1incangle-pygrapes.ipynb, a jupyter notebook in the main folder to run a simulated ptychographic reconstruction of a test structure, to see an example of how PyGRAPES .

Alternatively, you can run Demo_sim_Script_1incangle_scratch.ipynb which is messier, but does not require PyGRAPES to be installed explicitly, only other existing dependencies (torch, matplotlib, numpy, time, h5py, etc in the first cell of the notebook)

## Requirements
### Hardware 
PyGRAPES is strongly _reccomended_ to run on GPU, but in principle could run on CPU by changing the default device in pytorch. This has not been tested. 
For the demo scripts, the default settings require at least 5 GB VRAM and an NVIDIA GPU with CUDA. The software was built/tested on an RTX 3090 with 24 GB VRAM.

### Package dependencies
- Pytorch
- Numpy
- matplotlib
- skimage.draw
- scipy.ndimage

