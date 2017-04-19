# LSH-community-detection

community detection for the whole Twitter graph on a single laptop. We use minhash signatures to encode the Jaccard similarity between neighbourhood graphs of vertices in social networks. A Locality Sensitive Hash table is built on top of the minhashes to perform extremely fast nearest neighbour search. The results of the nearest neighbour search are ranked and structured using the WALKTRAP community detection algorithm

## Getting Started

Download the minhash data available at:

https://www.dropbox.com/s/sce6qcmbkpjpeuh/hashes.csv?dl=0

Assuming you are in the directory of the source code and have cloned the rep.

To build the LSH table

python LSH.py minhash_data_path LSH_path

To generate metrics for the ground truth communities

python assess_community_quality.py minhash_data_path outpath

To run experimentation

python run_experimentation.py minhash_data_path LSH_path outpath

### Prerequisites

The code uses the numpy, pandas and scikit-learn python packages. We recommend installing these through Anaconda. It also requires the mmh3 package

pip install mmh3

## Authors

**Ben Chamberlain**
