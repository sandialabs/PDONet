# PDONet: a graph convolutional layer based on differential operators implemented in PyTorch
Implementation and code to test the parameterized differential operator layer
for "Parameterized Pseudo-Differential Operators for Graph Convolutional Neural Networks" GSP-CV 2021.

## Installation

Some modules are dependent on python3 and code is written to run on python3 (given the long install times of some of the modules, taking the time to make sure you're using python is worthwhile if installing in a virtual environment).

Torch cluster/scatter expect certain packages to be available during install.

    pip3 install -r prereqs.txt

Note: Builds for torch-scatter and torch-sparse can be extremely slow.

    pip3 install -r requirements.txt

## Running

Datasets should be download as needed for runs.  MNIST hierarchical, CIFAR10, and CIFAR100 will have an extra delay as superpixel datasets are built.  Unfortunately, using a pruned dataset will result in an additional download despite being the same raw data as the superpixel version (this can be avoided by manually copying the raw folder to the *-SP-Pruned directory).

Each will train and validate against the test set and report the test accuracy for the best training epoch.

Running the superpixel classifier with the default configuration options:

    python3 run_classifier.py

Running the FAUST node correspondence task with default configuration:

    python3 run_faust.py

To override config options and/or load in additional configuration files:

    python3 run_classifier.py with depth=3             # set the model to use 3 layers
    python3 run_classifier.py with configs/cifar10.yml # override settings from a yml file

To run any of the classifiers with the hierarchical datasets, run a session such as:

    python3 run_classifier.py with hierarchical=True                      # hierarchical with default dataset (MNIST)
    python3 run_classifier.py with configs/cifar100.yml hierarchical=True # hierarchical with CIFAR100

In addition to the text output, full runs are logged to
`output/file\_observer/<run id>.`

## Datasets
Additional override configuration files are provided for MNIST hierarchical, CIFAR10, and CIFAR100 under `configs/`.  All configuration files correspond to the optimal values used in the paper.

## Configuration options

##### note
Blank by default.
Use for experiment notes at run time.

##### dataset
MNIST by default.  Classifier only.
Other options are CIFAR10 and CIFAR100.

##### batch\_size
Default 128.

##### learning\_rate
Default 0.0002.

##### weight\_decay
Default 0.0001.

##### device
Default cuda.
Use to specify a particular GPU to use on multigpu systems.

##### epochs
Default 100.

##### prune\_edges
Default False.  Classifier only.
Whether to reduce edges by replacing edge connections via Delauney triangulation.

##### hierarchical
Default False.  Classifier only.
Whether to include hierarchical graphs as described in Knyazev et al. 2019 https://arxiv.org/pdf/1907.09000.pdf.  Note: generating a CIFAR hierarchical graph is an order of magnitude slower than its superpixel counterpart.

##### edge\_dropout
Default 0.45.
Probability of dropping edges on input graph.

Following options are nested under model\_options:

##### depth
Default 7.
Number of convolutional/downsampling layers in model.

##### dropout\_rate
Default 0.5.
Dropout rate prior to fully connected layers.

##### grid\_scale
Default [0, 28].
Describes the relative scale of the node positions (assumes
equal scale for x and y)

##### initial\_features
Default 128.
Number of channels to output after first layer.  Doubled every
subsequent layer up to...

##### max\_features
Default 512.
Highest number of channels allowed.

##### voxels\_at\_depth
Default 3.  Classifier only.
After the downsampling layers, voxel grid should have an n x n
grid.  Voxels are halved in size each layer.

## Citation
Potter, Kevin et al. _Parameterized Pseudo-Differential Operators for Graph Convolutional Neural Networks_ GSP-CV 2021, Montreal, Canada, October 2021.

## License
Revised BSD.  See the LICENSE.txt file.
