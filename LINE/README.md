In this folder are the functions to use the LINE algorithm. 
Note that we use an implementation of LINE using previous versions of packages. Therefore, to use the LINE algorithm you need to save in the adjacency folder the adjacency matrix of the graph you want to process. 
If you want to use the ptb dataset, you need to put the csv in the folder.

To do this, just enter the command line:

python save_adjacency.py --dataset Beef --eps 0.1 

--dataset: precise the dataset
--eps: precise the value of eps 


Then, on another environnement, you need to enter the command line: (Replace Beef and 0.1 by your value of dataset and eps)

python line_main.py --input adjacency/.Beef;eps=0.1.npz --output embeddings/.Beef;eps=0.1.line.embeddings --iter 500 --proximity second-order

python evaluate_tencent.py --emb embeddings/.Coffee;eps=0.478297.line.embeddings --net adjacency/.Coffee;eps=0.478297.npz --testdir output

Now, the embeddings of the graph will be saved in the output file, and you can load it using functions in clustering_algos.py or in the notebook. 

To install the second environnment, you need to do the following steps:

First:
    conda create --name tf1env python=3.7 #create an environnment with python 3.7
    conda activate tf1env 
    pip install tensorflow==1.15.5

In case of protobuf error:
    pip uninstall protobuf
    conda install protobuf=3.20

Finally:
    pip install requirement.txt
