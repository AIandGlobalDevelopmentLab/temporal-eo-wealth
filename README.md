# Time series of satellite imagery improve deep learning estimates of neighborhood-level poverty in Africa (IJCAI 2023)
**[Lab](https://liu.se/en/research/global-lab-ai)** | 
**[Paper](https://www.ijcai.org/proceedings/2023/0684.pdf)** | 
**[Appendix](./IJCAI_23_supplementary_material.pdf)** | 
**[Video](https://ijcai-23.org/video/?vid=39005554)** |
**[Generated maps](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2F9DINV4)**

This is the official repository for the IJCAI 2023 paper 
"_Time series of satellite imagery improve deep learning estimates of neighborhood-level poverty in Africa_".  

Authors: 
[Markus Pettersson](https://markuspettersson.com),
[Mohammad Kakooei](https://se.linkedin.com/in/mohammad-kakooei-33118211a),
[Julia Ortheden](https://se.linkedin.com/in/julia-ortheden-6a0623134), 
[Fredrik D. Johansson](https://fredjo.com), 
[Adel Daoud](https://AdelDaoud.se).

## Apptainer environment

In order to improve reproducability, we ran all of our code using a single [Apptainer (previously known as Singularity)](https://apptainer.org) container. This container can be built using the included recipe file [apptainer_recipe.def](./apptainer_recipe.def) as described in the apptainer documentation. Make sure you include the image path you select, e.g. `path/to/image/location.sif`, in your version of the configuration file [config.ini](./config_sample.ini). 

To execute a .py script, simply run

```bash
$ apptainer run path/to/image/location.sif -nv path/to/script/file.py --script_args
```

in order to run one of the jupyter notebooks, you can start a jupyter lab session by running

```bash
$ apptainer exec path/to/image/location.sif -nv jupyter
```

## Running trained single- and multi-frame models

Steps:

1. Set up your local paths and other environment variables in the [config.ini](./config.ini) file.

2. Download the satellite data, calculate the dataset variables and prepare the cross-validation folds as outlined in the [preprocessing](./preprocessing) directory.

3. Make predictions for the different pretrained models by running [inference_model.py](./inference/inference_model.py). In case your system is equipped with [Slurm](https://slurm.schedmd.com/documentation.html), you can simply run the [inference_model.sh](./inference/inference_model.sh) script 

4. Generate the figures as presented in the paper by running the [evaluate_results/model_evaluation.ipynb](evaluate_results/model_evaluation.ipynb) and [evaluate_results/ts_effect.ipynb](evaluate_results/ts_effect.ipynb) notebooks.

## Acknowledgements
Preprocessing and evaluation code in this repository takes a lot of inspiration from the [work by Yeh et al.](https://doi.org/10.1038/s41467-020-16185-w), creators of the architecture we call "single-frame model". You can find their codebase [here](https://github.com/chrisyeh96/africa_poverty_clean).

## Citation

Please cite our paper as

> Markus B. Pettersson, Mohammad Kakooei, Julia Ortheden, Fredrik D. Johansson, & Adel Daoud (2023). Time Series of Satellite Imagery Improve Deep Learning Estimates of Neighborhood-Level Poverty in Africa. *In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, ĲCAI-23* (pp. 6165–6173).

Or use the follwoing BibTex entry

```
@inproceedings{pettersson2023time,
  author       = {Markus B. Pettersson and
                  Mohammad Kakooei and
                  Julia Ortheden and
                  Fredrik D. Johansson and
                  Adel Daoud},
  title        = {Time Series of Satellite Imagery Improve Deep Learning Estimates of
                  Neighborhood-Level Poverty in Africa},
  booktitle    = {Proceedings of the Thirty-Second International Joint Conference on
                  Artificial Intelligence, {IJCAI-23}},
  pages        = {6165--6173},
  publisher    = {International Joint Conferences on Artificial Intelligence Organization},
  year         = {2023},
  month        = {8}
  url          = {https://doi.org/10.24963/ijcai.2023/684},
  doi          = {10.24963/ijcai.2023/684}
}
```
