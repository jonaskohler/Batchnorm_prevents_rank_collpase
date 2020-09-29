# Code for Batch normalization provably avoids ranks collapse for randomly initialised deep networks
NeurIPS2020


In this work we prove that BN layers effectively prevent the input output mapping of deep neural networks to deteriorate to a rank one mapping.


![Theorem 2](/theorem2.png)




The code supplied alongside our submission can reproduce all figures from the main text. 

Package requirements: numpy, torch, torchvision, python 3, ipython, jupyter, pickle,seaborn, panda, sys, getopt, CUDA. Furthermore, access to GPUs is needed for running the code. 

Running the code: 
1. To produce the result of Fig.1, one needs to run code_fig_1.py first and then plot the results by running plotting_fig_1.ipynp using jupyter. 
2. Fig. 2, 3 and 6 are straightforward to run directly within the Jupyter notebook.
3. For Fig. 4, code_fig_4_bn.py runs SGD on a BN-net and code_fig_4_pre_training.py runs our pertaining method to produce the result of Fig. 4. To run the code you need to type "python code_fig_4_bn.py #num_layers" in the terminal where #num_layers is the number of hidden layers in the network. Similarly, you can run pre-training method by "python code_fig_4_bn.py #num_layers". The outputs of these codes are lists containing training loss in different epochs for 5 independent runs. The output is saved in result_bn_(#num_layer).npy, result_our_(#num_layer)_loss.npy, result_our_(#num_layer)_acc.np, and result_our_(#num_layer)_test_loss.npy
4. Instructions for running Figure 5 are provided in code_fig_5/readme

![Simulations](/intro.png)
