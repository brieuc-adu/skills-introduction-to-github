<header>

# Neural Incomplete Factorization

_A method to precondition sparse symmetric linear systems using Graph Neural Networks._

</header>

## Prerequisites

**Required python packages** : torch, torch_geometric, ilupp, networkx, pyvis.

**Hardware** : 40Go of CPU RAM to generate the 1000 graphs of the train dataset and a minimum of 12Go of GPU RAM to train a model with cuda for the Hutchinson loss and a very small batch size.

## Usage of the '.py' files

1. feature.py contains the diagonal_dominance and diagonal_decay functions as well as other functions taken from the networkx package to compute the feaures of a graph.
2. **data_gen.py generates the train, validation and test datasets in ".pth" formats**. Parsing options allow to change the degree of sparsity of the underlying matrices with '-p', their order with '-n_min', and '-n_max', and the number of graph per dataset with '-n_train', 'n_valid' and 'n_test'. Defaults values replicate the datasets of the Neural-IF paper, i.e : p = 0.999 , n_min = n_miax = 10000, n_train = 1000, n_valid = 10, n_test = 100.
3. utils.py contains the PCG algotithm and the pcg_model_vs_ic function to asses the different preconditionning method with the PCG on the test dataset. It also contains the (Hutchinson and Frobenius) losses functions.
4. gnn.py contains the architecture of the Neural-IF model
5. **train.py trains the model definde in gnn.py over the train datasets and stores it in the /models folder**. Parsing options allow to choose the device on which the model is trained with -device followed by 'cpu' or 'cuda' (default is 'cuda'), the number of epoch with -epoch (default is 50), the batch_size with -batch_size (default is 1) and the output model's name with -output_name (default is 'model'). It also returns the plot of the losses per epoch for the training and validation data compared to the losses obtained with IC(0).
6. **test.py applies a trained model on the PCG with the test dataset and compare its performances with IC(0) and the absence of preconditioners. It returns the plot of the number of iteration of the PCG for each linear systems of the test dataset obtained each method.** Parsing option allow to choose the name of the applied model with -model (default : 'model5.pt'), to choose the device on which the model is tested with -device followed by 'cpu' or 'cuda' (default is 'cuda'), and to compare or not the model to PCG with IC(0) and CG with the boolean option -only_neural_IF (default is False). 
7. visualization.py contains the function creating the plots of the train.py and test.py files.
