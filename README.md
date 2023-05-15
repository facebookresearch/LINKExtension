## LINKExtension
This repository provides a reference implementation for "Node Attribute Prediction on Multilayer Networks with Weighted and Directed Edges" on the protein-protein network from <a href="https://ogb.stanford.edu/">Open Graph Benchmark</a>.

### Documentation 
This repository contains code to replicate the results in the paper on the protein-protein network from the Open Graph Benchmark repository. All code was run in March/April 2023. We do not include data for replicating internal FB results but note that the code architecture is the same as the simulation.

### Directions

This repository illustrates how to run the LINK extensions on a publicly available dataset. All code was run with Python 3.8.6. Since we report ROC-AUC metrics, we do not normalize degree counts as illustrated in the paper. If you report other predictive performance metrics, you will want to add this normalization back-in. For this paper, we also enfoce bins with no edges to be 0.

#### Single-Layer
* All notebooks are run from the /single_layer/ folder
* Update all file paths for saving output
* Code is run in parts which correspond to the different outcomes, so comment out related section 

#### Multi-Layer
* All notebooks are run from the /multi_layer/ folder
* Update all file paths for saving output
* Code is run in parts which correspond to the different outcomes, so comment out related section 

### LICENSE
LINKExtension is MIT Licensed, as found in the LICENSE file.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

The majority of LINKExtension is licensed under the MIT License, however portions of the project are available under separate license terms: <a href="https://ogb.stanford.edu/">Open Graph Benchmark</a> and <a href="https://github.com/scikit-learn/sklearn-docbuilder">scikit-learn</a> is licensed under the MIT license; <a href="https://numpy.org/doc/stable/license.html">numpy</a>, <a href="https://networkx.org/documentation/networkx-2.1/index.html#">networkx</a>, <a href="https://github.com/pandas-dev/pandas">pandas</a>, and <a href="https://scipy.org">scipy</a> is licensed under the BSD 3-Clause license; sys is licensed under the following license <a href="https://docs.python.org/3/license.html">Python3</a>; and matplotlib is licensed under the following <a href="https://matplotlib.org/stable/users/project/license.html">license</a>. We did not make any changes to existing Python libraries and refer users to the original license agreements for all imported libraries.


### Code Authors
* Yiguang Zhang, yiguang.zhang@columbia.edu
* Kristen M. Altenburger, kristenaltenburger@gmail.com


Copyright (c) Meta Platforms, Inc. and affiliates.
