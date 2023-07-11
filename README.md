![LOGO](https://github.com/DIG-Kaust/PyPOCS/blob/main/asset/logo.png)

Reproducible material for **Why POCS works, and how to make it better - Ravasi M., Luiken N.** submitted to Geophysics.


## Project structure
This repository is organized as follows:

* :open_file_folder: **pypocs**: python library containing routines for POCS interpolation with various solvers;
* :open_file_folder: **data**: folder containing data;
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);
* :open_file_folder: **scripts**: set of python scripts used to run multiple experiments ...

## Notebooks
The following notebooks are provided:

- :orange_book: ``POCS_Overthrust-Irregular3D.ipynb``: notebook performing interpolation of a shot gather of the Overthrust dataset from irregularly sampled inlines;
- :orange_book: ``POCS_Overthrust-FullIrregular3D.ipynb``: notebook performing interpolation of a shot gather of the Overthrust dataset from an irregularly sampled 2d-grid;
- :orange_book: ``POCS_Overthrust-Summary.ipynb``: notebook visualizing the results of the various algorithms used in the two previous notebooks.
- :orange_book: ``POCS_Overthrust-Offthegrid3D_windows.ipynb``: notebook performing interpolation of a shot gather of the Overthrust dataset from irregularly sampled inlines with dithering (using disjoint windowed approach);
- :orange_book: ``POCS_Overthrust-Offthegrid3D_sliding.ipynb``: notebook performing interpolation of a shot gather of the Overthrust dataset from irregularly sampled inlines with dithering (using global, sliding-window approach).

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

Remember to always activate the environment by typing:
```
conda activate pypocs
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.


## Citing :newspaper: :trophy:
If you find this library useful for your work, please cite the following papers

```
@article{ravasi2022,
	title={Why POCS works, and how to make it better},
	authors={M. Ravasi, N. Luiken},
	journal={ArXiv},
	year={2022}
}
```
