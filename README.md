# LSH-community-detection

This is the code for our paper 'Real-Time Community Detection in Large Social Networks on a Laptop' https://arxiv.org/abs/1601.03958 Community detection for large networks on a single laptop. We use minhash signatures to encode the Jaccard similarity between neighbourhood graphs of vertices in social networks. A Locality Sensitive Hash table is built on top of the minhashes to perform extremely fast nearest neighbour search. The results of the nearest neighbour search are ranked and structured using the WALKTRAP community detection algorithm.

### Prerequisites

The code uses the numpy, pandas and scikit-learn python packages. We recommend installing these through Anaconda. Generating minhashes requires the mmh3 package. 

pip install mmh3

We provide binaries of the cython code. If you wish to alter the cython code you will need to install cython

pip install cython


## Replicating the experiments with Twitter data

Download the minhash data available at:

https://www.dropbox.com/s/sce6qcmbkpjpeuh/hashes.csv?dl=0

Assuming you are in the directory of the source code and have cloned this repository.

To build the LSH table

python LSH.py minhash_data_path LSH_path

To generate metrics for the ground truth communities

python assess_community_quality.py minhash_data_path outpath

To run experimentation

python run_experimentation.py minhash_data_path LSH_path outpath


## Replicating the end-to-end process with the public email data set from SNAP https://snap.stanford.edu/data/email-EuAll.html

python run_email_data.py

This will generate minhashes from the raw data and use them to build an LSH table. From the LSH table all of the results shown in the paper are generated. 

The LSH table and the minhashes are written to the resources folder. The plots are written to the results folder.


## Authors

**Ben Chamberlain**

###  Citation

If you make use of this code please cite

Chamberlain BP, Levy-Kramer J, Humby C, Deisenroth MP. Real-Time Community Detection in Large Social Networks on a Laptop. arXiv preprint arXiv:1601.03958. 2016 Jan 15.
