
# TL;DR
- Create your environment using section 1.
- Download our dataset or format your data in a similar fashion
- Export the data path as your _REFINER_DATA_DIR_
- (Optional) Configure the registration hyperparameters using section 3.
- Run 
```
$ python main.py \
  --set <set> \
  --scene <scene_name> \
  --img_name <img_name>
```
e.g
```
$ python main.py \
  --set TanksAndTemples \
  --scene Church \
  --img_name 21
```

# 1. ReFineR Environment
We tested our code on NVIDIA GeForce RTX 4070 with CUDA 12.8 installed and python 3.10
### 1. create environment
```
conda create --name refiner python=3.10
```
```
conda activate refiner
```

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```
pip install -r requirements.txt
```

```
pip install -e . --no-build-isolation
```

### 2. Install GSSLAM renderer In MonoGS folder
```
pip install third_party/FORK_MonoGS/submodules/simple-knn --no-build-isolation
```
```
pip install third_party/FORK_MonoGS/submodules/diff-gaussian-rasterization --no-build-isolation
```
if you have issues try in both submodules folders:
```
python setup.py install
```

6. Simlinks
install einops h5py loguru yacs pytorch_lightning
inside third_party
ln -s <path to matchAnything> MatchAnything (the one inside the imcui folder)
change the header when you have the error


# 2. Data
1. Download our dataset or create data with our common dataset format 
<!-- 
ReFineRData
  set0
    scene0
      images
        img0.png
        ...
      cameras
        img0.yaml
        ...
      pc.ply
    scene1
        images
          img0.png
          ...
        cameras
        pcs
          img0.ply
          ...
    ...
  set1
  ... -->
```
.
└── ReFineRData/
    ├── set0/
    │   ├── scene0/
    │   │   ├── images/
    │   │   │   ├── img0.png
    │   │   │   └── ...
    │   │   ├── cameras/
    │   │   │   ├── img0.yaml
    │   │   │   └── ...
    │   │   └── pc.ply
    │   ├── scene1/
    │   │   ├── images/
    │   │   │   ├── img0.png
    │   │   │   └── ...
    │   │   ├── cameras
    │   │   └── pcs/
    │   │       ├── img0.ply
    │   │       └── ...
    │   └── ...
    ├── set1
    └── ...
```
2. Export the REFINER_DATA_DIR variable to the dataset
# 3. Configuration
0. Download MatchAnything weights

1. Configure main.yaml in the config folder by choosing a list of components in the config/components folder. Configurations will be merged by using the last configuration values in the list  

2. (Optional) Change the configuration values in the config/components folder. In particular:
    - If you have restrained GPU memory available and large point clouds, decrease the _data.pc_subsample_factor_ value or add _data_subs_01_ to the main config list (Required for our 
    TanksAndTemples, Livox and Carla sets)

    - Change the loss function comparison by selecting a different loss configuration. From our experiments normals correspond to the geometric registration (for all our sets), reflectivity for the intensity registration for nuScenes and Livox and gray for TanksAndTemples and RGBDScenes. From the _loss.hogs.channels.comp_ list the first argument is the point cloud channel and the second the image channel. 
    - /!\ If you use more than 6 different point cloud channels, you need to change the NUM_CHANNELS field from the gaussian splatting rasterizer (_submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h_) and rebuild it.

    - Choose a saver configuration and modules: _saver_file_ for file saving and _saver_rerun_ for better visualization using Rerun (you can add both)

    - Add a random perturbation with the _data.pert_init_ field or the _data_pert_10_ file. Perturbation is defined in terms of pixels (_data.pert_init.p_) with the method defined in the article or rotations and translation (_data.pert_init.r_, _data.pert_init.t_)
    
    - Please do not touch the fields with predefined variable interpolations "${}"

# 4. Running the main script
- You can simply run the main.py script using
```
$ python main.py \
  --cfg_file config/main.yaml \
  --data_path <data_path> \
  --set TanksAndTemples \
  --scene Church \
  --img_idx 1
```
Changing _TanksAndTemples_, _Church_ and _img_idx_ to your data tree structure. 

- The _img_idx_ field is an integer between 0 and the number of images in the scene "image" folder, sorted in alphabetical order. You can also use the _img_name_ option to select the image by its name (without extension) instead.

- If no data path is specified the script will try to use the _REFINER_DATA_DIR_ environment variable as the data folder

- If no config file is specified, config/main.yaml is used

- If no scene is specified, the script will run over all images of all scenes from the set

- The _precomp_dir_ option specifies the precomputation directory used for Marigold and MatchAnything (see sections 5 and 6). Default is output/precomp

- You can add the --repeat \<int> options to repeat the registration a given number of times for each image. This is useful when running experiments.

# 4b. Running the benchmark experiments
You can use the defined configuration files in "config/experiments" to run the experiment setup on our data, for instance using 
```
python main.py \
  --set nuScenes \
  --cfg_file config/experiments/reflectivity.yaml \
  --out_dir output/experiments/intensity \
  --repeat 5 
```
Commands are also given in the scripts/experiments.sh file. After each experiment on several sets you should run the compile_results.py script with 
```
python compile_results.py \
  --out_dir output/experiments/intensity
```

# 5. (Optional) Registration with generated Marigold channels
You can generate synthetic channels using the marigold.py script.
1. Download the Marigold weights and put it in the weights/Marigold folder (check [the Marigold repository](https://github.com/prs-eth/Marigold?tab=readme-ov-file#-checkpoint-cache))
2. Build the docker image from docker/Marigold with
```
$ docker image build -t marigold .
```
3. Run the marigold.py script with 
```
python marigold.py \
--data_path <data_path> \
--set TanksAndTemples \
--scene Church \
--models normals iid 
```
- This will create an output/precomp/\<set>/\<scene> folder with Marigold outputs. You can specify another folder using the _out_dir_ option
- You can specify which model(s) to use (depth, normal, iid)

4. Select synthetic channels in the loss configuration component on the image side of the _loss.hogs.channels.comp_ options (nx, ny, nz for normals, dn for depth, mat and alb for material and albedo)

5. Run the main script
