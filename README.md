# Automating Stepsizes for Decentralized Optimization and Learning with Geometric Convergence
Decentralized optimization is a promising paradigm for addressing fundamental challenges in machine learning. However, despite the unprecedented success of existing decentralized optimization and learning methods, the selection of effective stepsizes is still elusive.

We propose an approach that allows individual agents to autonomously adapt their individual stepsizes. The effectiveness of the proposed approach is confirmed using three typical machine learning applications on benchmark datasets, including logistic regression, matrix factorization, and image classification.

![Introduction](./figures/introduction.png)

## Outlines
- Installation Tutorial and Preliminaries
- Logistic Regression
- Matrix Factorization
- Training of Convolutional Neural Networks
- Discussions
- License

## Installation Tutorial and Preliminaries
### Install Setup
1. Clone this [repository](https://github.com/cziqin/Automating-Stepsizes/tree/main)
2. Download and install [Anaconda](https://www.anaconda.com) (if you don't have it already)
3. Create a new virtual environment with python 3.12, take conda as an example:
   ```shell
   conda create -n autostep python=3.12
   conda activate autostep
   ```
4. Install any additional packages you need in this environment using conda or pip. Recommend installing necessary packages by using the following command:
   ```shell
   pip install -r requirements.txt
   ```

### Hardware/computing resources
The experiments were conducted using the Windows 11 OS equipped with a 32-core CPU, 32GB RAM, and one NVIDIA GeForce RTX 4090 GPU with 24GB VRAM.

### Repository Structure
```
├── Logistic_regression                 # directory for logistic regression classification
│   ├── loss_function.py                # loss function 
│   ├── main.py                         # entry point
│   ├── matrix.py                       # communication operations
│   ├── mushrooms                       # the mushrooms datasets
│   ├── optimizer.py                    # optimization algorithms
│   └── train.py                        # model training
├── Matirx_factorization                # directory for matrix factorization
│   ├── main.py                         # entry point
│   ├── matrix.py                       # communication operations
│   ├── optimizer.py                    # optimization algorithms
│   └── u.data                          # the MovieLens 100k dataset
├── Neural_networks                         
│   ├── datadeal.py                     # split the ImageNet dataset into training and test sets
│   ├── decompress.sh                   # decompress the ImageNet dataset in Linux OS
│   ├── main.py                         # entry point
│   ├── matrix.py                       # communication operations
│   ├── models.py                       # model for the CIFAR-10 experiment
│   ├── ops.py                          # optimization algorithms
│   ├── resnet.py                       # ResNet 18
│   └── train.py                        # model training
├── figures                             # figures in the README document
├── LICENSE                             # License file
├── README.md                                             
└── requirements.txt                    # list of dependencies required to run the project             
```

### Datasets
| Datasets       | Download link                                            | Storage Location                   |
|----------------|----------------------------------------------------------|------------------------------------|
| Mushrooms      | https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ | `./Logistic_regression/`           |
| MovieLens 100k | https://grouplens.org/datasets/movielens/                | `./matrix_factorization/`          |
| CIFAR-10       | https://www.cs.toronto.edu/~kriz/cifar.html              | `./Neural_networks/data/`          |
| ImageNet       | https://image-net.org                                    | `./Neural_networks/data/imagenet/` |

Ensure that each dataset is downloaded and placed in its corresponding directory before running the experiments.
>Note: For the ImageNet dataset, first ensure that the [dataset](https://image-net.org) has been downloaded. Then, run [datadeal.py](./Neural_networks/datadeal.py) to split it into the training set and the test set, respectively. Finally, place them in the `./Neural_networks/data/imagenet/train` directory and the `./Neural_networks/data/imagenet/val` directory, respectively. 

## Logistic regression
1. You can use the following command to execute Algorithm 1 for the logistic regression task:
    ```shell
    python main.py --test_num 0 --iterations 1000
    ```
   > Note: Please change the directory to [`./Logistic_regression`](./Logistic_regression) before running the above command.

   ![Mushroom](./figures/mushrooms_execution.gif)
   - `--test_num`: specifies which optimization algorithm to train: `0`:Algorithm 1; `1`: Algorithm 2; `2`: DGM-BB-C [R1]; `3`: DGD [R2].
   - `--iterations`: sets the number of total training iterations.
2. To execute Algorithm 2 with a desired number of inner-consensus-loop iterations K (e.g., K=10), you can run the following command:
   ```shell
   python main.py --test_num 1 --iterations 1000 --k_loop 10
   ```
3. To execute Algorithm 3 with a desired number of local updates Q (e.g., Q=10), you can run the following command:
   ```shell
   python main.py --test_num 0 --iterations 1000 --const_q 10
   ```
   > Note: The parameter `K` represents the number of inner-consensus-loop iterations in Algorithm 2 and DGM-BB-C. The parameter `Q` represents the number of local updates in Algorithm 3.
4. In this experiment, we set the stepsize $\eta=1/L_{\max}=0.03511$ for DGD, which follows the default parameter suggested by [R2]. Following Lemma 2 in [R1], the stepsize upper bound for DGM-BB-C was set to $10/L_{\max}=28.4819$. The stepsizes of Algorithm 1 and Algorithm 2 are automatically adjusted without requiring any manual tuning.
5. The update code for our tuning-free automatic stepsize in the [`optimizer.py`](./Logistic_regression/optimizers.py) file is given as follows:
   ```python
   if iteration == 0:
       self.eta[i].append(self.eta[i][0])
   else:
       a = np.sqrt(1 + self.eta[i][iteration] / self.eta[i][iteration - 1]) * self.eta[i][iteration]
       b = la.norm(self.agent_parameters[i][iteration + 1] - self.agent_parameters[i][iteration]) / (2 * la.norm(self.agent_y[i][iteration+ 1] - self.agent_y[i][iteration]))
       self.eta[i].append(min(a, b))
   ```
### Experimental results
<div style="text-align:center">
  <img src="./figures/mushrooms.png" alt="Fig3" width="900">
</div>

- <b>Fig. a</b> The loss evolution of Algorithm 1, Algorithm 2 (with K=1 and K=10, respectively), Algorithm 3 (with Q=10), DGM-BB-C [R1], and DGD [R2], respectively. 
- <b>Fig. b</b> and <b>Fig. c</b> The scatter and box plots of the average stepsize of five agents across the six algorithms. Box plots show the median, 1st and 3rd quartiles, and 5th to 95th percentiles. Pvalues
were calculated via one-way analysis of variance (ANOVA) with Tukey’s multiple comparison test.
- <b>Fig. d</b> The loss evolution of Algorithm 3 under different numbers of local iterations Q with respect to communication rounds.
- <b>Fig. e</b> The loss evolution of Algorithm 1, Algorithm 2 (with K=1 and K=10, respectively), Algorithm 3 (with Q=10), DGM-BB-C, and DGD, respectively, with respect to communication rounds. 
- <b>Fig. f</b> The used wallclock time and associated loss of Algorithm 1, Algorithm 2 (with K=1 and K=10, respectively), Algorithm 3 (with Q=10), DGM-BB-C, and DGD, respectively.

<b>Result discussion:</b> Fig. a demonstrates that Algorithm 1 has a better convergence accuracy compared with its single-loop variant (Algorithm 2 with K=1), its asynchronous-parallelupdate variant (Algorithm 3 with Q=10), the decentralized adaptive algorithm with Barzilai-Borwein stepsizes (DGM-BB-C) [R1], and DGD [R2]. Fig. e and Fig. f demonstrate that our Algorithm 1 requires fewer communication rounds and a shorter wallclock time to reach a given convergence accuracy compared with DGM-BB-C and DGD. Moreover, removing the inner-consensus loop (i.e., Algorithm 2 with K=1) or using asynchronous parallel updates (i.e., Algorithm 3 with Q=10) can reduce both communication complexity and the used wallclock time. Among all comparison algorithms, Algorithm 3 with Q=10 local updates in each communication round achieves the lowest communication overhead and shortest wallclock time. Nevertheless, a too large number of local updates in each communication round leads to degraded convergence accuracy, especially in the later stages of convergence, as shown in Fig. d.

> Note: All experimental results (e.g., loss, wallclock time, and the average stepsize) will be autonomously saved as `.csv` files in the `./Logistic_regression/results` directory.

## Matrix factorization
1. You can use the following command to execute Algorithm 1 for the matrix factorization task:
   ```shell
   python main.py --test_num 0 --iterations 1000
   ```
   > Note: Please change the directory to [`./Matrix_factorization`](./Matrix_factorization) before running the above command.
   
   ![Matrix](./figures/Matrix_factorization_execution.gif)
   - `--test_num`: specifies which optimization algorithm to train: `0`:Algorithm 1; `1`: Algorithm 2; `2`: DGM-BB-C; `3`: DGD.
   - `--iterations`: sets the number of training iterations.
2. To execute Algorithm 2 with a desired number of inner-consensus-loop iterations K (e.g., K=10), you can run the following command:
   ```shell
   python main.py --test_num 1 --iterations 1000 --k_loop 10
   ```
3. To execute Algorithm 3 with a desired number of local updates Q (e.g., Q=10), you can run the following command:
   ```shell
   python main.py --test_num 0 --iterations 1000 --const_q 10
   ```
4. In this experiment, we used the stepsize $\eta=10^{-4}$ for DGD [R2] that gave us the best performance in the stepsize tuning process. The losses of DGD under different stepsize settings are summarized in Table S1, with
the best value highlighted in bold. The stepsize upper bound for DGM-BB-C [R1] was set to 10.
   <table>
      <caption><b>Table S1</b> Loss of DGD [R2] under different stepsize settings after 200 iterations in the matrix factorization experiment using the MovieLens 100k dataset.</caption>
    <tr>   <th rowspan="2">Stepsize</th>    <th colspan="2">Loss</th>   </tr>
    <tr>   <th>DGD</th>   <th>Algorithm 1 (with no tuning)</th>   </tr>
    <tr>   <td>10<sup>-8</sup></td> <td>5.673</td>   <td rowspan="9"><b>5.095</b></td>   </tr>
    <tr>   <td>10<sup>-7</sup></td> <td>5.667</td>   </tr>
    <tr>   <td>10<sup>-6</sup></td> <td>5.614</td>   </tr>
    <tr>   <td>10<sup>-5</sup></td> <td>5.330</td>   </tr>
    <tr>   <td>10<sup>-4</sup></td> <td><b>5.135</b></td>   </tr>
    <tr>   <td>10<sup>-3</sup></td> <td>nan</td>   </tr>
    <tr>   <td>10<sup>-2</sup></td> <td>nan</td>   </tr>
    <tr>   <td>10<sup>-1</sup></td> <td>nan</td>   </tr>
    <tr>   <td>10<sup>0</sup></td> <td>nan</td>   </tr>
   </table>
   
   > Since the standard deviations of DGD's and Algorithm's losses are both less than 0.001, they are omitted in this table.

### Experimental results
<div style="text-align:center">
  <img src="./figures/matrix_factorization.png" alt="Fig4" width="900">
</div>

- <b>Fig. a</b> The loss evolution of Algorithm 1, Algorithm 2 (with K=1 and K=10, respectively), Algorithm 3 (with Q=5), DGM-BB-C [R1], and DGD [R2], respectively. 
- <b>Fig. b</b> and <b>Fig. c</b> The scatter and box plots of the average stepsize of five agents across the six algorithms. Box plots show the median, 1st and 3rd quartiles, and 5th to 95th percentiles. Pvalues
were calculated via one-way analysis of variance (ANOVA) with Tukey’s multiple comparison test.
- <b>Fig. d</b> The loss evolution of Algorithm 3 under different numbers of local iterations Q with respect to communication rounds.
- <b>Fig. e</b> The loss evolution of Algorithm 1, Algorithm 2 (with K=1 and K=10, respectively), Algorithm 3 (with Q=5), DGM-BB-C, and DGD, respectively, with respect to communication rounds. 
- <b>Fig. f</b> The used wallclock time and associated loss of Algorithm 1, Algorithm 2 (with K=1 and K=10, respectively), Algorithm 3 (with Q=5), DGM-BB-C, and DGD, respectively.

<b>Result discussion:</b> Fig. a shows that even under nonconvex and nonsmooth objective functions, Algorithm 1 still outperforms DGD and DGM-BB-C. This better performance is attributed to a larger stepsize produced by our Algorithm 1, as illustrated in Fig. b and Fig. c. Fig. a also shows that Algorithm 2 with K=1 (i.e., Algorithm 2 without inner-consensus loops) leads to inferior performance compared with Algorithm 1. Although increasing K to 10 improves the convergence performance, it also results in an increase in communication complexity and the used wallclock time, as shown in Fig. e and Fig. f. In addition, Fig. d, Fig. e, and Fig. f demonstrate that Algorithm 3 with Q=5 achieves the lowest communication complexity and shortest wallclock time among all comparison algorithms. However, the reduced communication complexity in Algorithm 3 comes at the price of reduced convergence accuracy compared with algorithms that communicate every time when a gradient descent is conducted (for example, as shown in Fig. a, Algorithm 1, Algorithm 2 with K=1 and K=10, respectively, and DGM-BB-C that execute synchronous communication and computation have a higher convergence accuracy than Algorithm 3 with Q=5).

>Note: The loss value is presented using its base-10 logarithm for better visualization. In addition, all experimental results (e.g., loss, wallclock time, and the average stepsize) will be autonomously saved as `.csv` files in the `./Matrix_factorization/results` directory.

## Neural network training
> Note: Please change the directory to [`./Neural_networks`](./Neural_networks) before running the following commands.
### CIFAR-10
1. You can use the following command to execute Algorithm 1 for the conventional neural network training task on the CIFAR-10 dataset:
   ```shell
   python main.py --test_num 0 --epochs 70 --batch_size 128 --dataset 'cifar10'
   ```
   
   - `--test_num`: specifies the optimization algorithm to be trained: `0`:Algorithm 3; `1`: DADAM [R3]; `2`: DAMSGrad [R4]; `3`: DSGD-N [R5]; `4`: ATC-DIGing [R6]; `5`: DSGD [R2].
   - `--epochs`: sets the number of training epochs.
   - `--batch_size`: sets the batch size for training.
   - `--dataset`: specifies the dataset to be used for training. The default option is 'cifar10'.

   >Note: Before running the script, please ensure that the [`CIFAR-10`](https://www.cs.toronto.edu/~kriz/cifar.html) dataset has been downloaded and placed in the `./Neural_networks/data` directory.
   
2. To execute Algorithm 3 with a desired number of local updates Q (e.g., Q=15), you can run the following command:
   ```shell
   python main.py --test_num 0 --epochs 70 --batch_size 128 --dataset 'cifar10' --const_q 15
   ```
3. To specify the print interval (e.g., printing the training loss, test loss, and test accuracy every 10 iterations), you can run the following command:
   ```shell
   python main.py --test_num 0 --epochs 70 --batch_size 128 --dataset 'cifar10' --interval 10
   ```
4. To specify the random seed used in training (e.g., setting seed=42), you can run the following command:
   ```shell
   python main.py --test_num 0 --epochs 70 --batch_size 128 --dataset 'cifar10' --seed 42
   ```
5. In this experiment, we employed a four-layer CNN architecture, where the LeakyReLU activation function and batch normalization are applied to every layer. This architecture comprises two convolutional blocks: the first block consists of a 32-filter convolution layer with dropout (rate 0.3), followed by a 64-filter convolution layer with dropout (rate 0.1) and a max pooling layer; the second block consists of two 128-filter convolution layers, with the latter followed by a max pooling layer, a global average pooling, and a fully connected layer.

6. The learning-rate parameters for DADAM [R3], DAMSGrad [R4], and DSGD with Nesterov’s momentum [R5] (called DSGD-N) were set to $\eta=0.005$, $\eta=0.1$, and $\eta=0.5$, respectively, after systematic tuning. We provide the performance of these algorithms under the tried parameters in the tuning process in the following Table S2, with the best value highlighted in bold:
   
<table>
   <caption><b>Table S2</b> Test accuracies of Algorithm 3, DADAM [R3], DAMSGrad [R4], and DSGD-N [R5] under different learning-rate parameters after 70 epochs in the CNN experiment using the CIFAR-10 dataset.</caption>
    <tr> 
        <th rowspan="2">Stepsize</th>    
        <th colspan="4">Test accuracy</th>
    </tr>
    <tr> 
        <th>DADAM</th>   
        <th>DAMSGrad</th>  
        <th>DSGD-N</th>   
        <th>Algorithm 3 (Q=1; with no tuning)</th>
    </tr>
    <tr>
        <td>10</td> <td>0.309 &pm; 0.096</td> <td>0.164 &pm; 0.040</td> <td>0.100 &pm; 0.000</td> <td rowspan="13"><b>0.804 &pm; 0.011</b></td>
    </tr>
    <tr>
        <td>5</td> <td>0.329 &pm; 0.124</td> <td>0.169 &pm; 0.031</td> <td>0.664 &pm; 0.013</td>
    </tr>
    <tr>
        <td>1</td> <td>0.511 &pm; 0.083</td> <td>0.652 &pm; 0.025</td> <td>0.736 &pm; 0.007</td>
    </tr>
    <tr>
        <td>0.5</td> <td>0.634 &pm; 0.027</td> <td>0.668 &pm; 0.008</td> <td><b>0.764 &pm; 0.008</b></td>
    </tr>
    <tr>
        <td>0.1</td> <td>0.646 &pm; 0.026</td> <td><b>0.762 &pm; 0.008</b></td> <td>0.757 &pm; 0.012</td>
    </tr>
    <tr>
        <td>0.05</td> <td>0.671 &pm; 0.009</td> <td>0.741 &pm; 0.009</td> <td>0.697 &pm; 0.013</td>
    </tr>
    <tr>
        <td>0.01</td> <td>0.754 &pm; 0.023</td> <td>0.718 &pm; 0.022</td> <td>0.619 &pm; 0.026</td>
    </tr>
    <tr>
        <td>0.005</td> <td><b>0.767 &pm; 0.009</b></td> <td>0.713 &pm; 0.018</td> <td>0.571 &pm; 0.007</td>
    </tr>
    <tr>
        <td>0.001</td> <td>0.709 &pm; 0.009</td> <td>0.654 &pm; 0.019</td> <td>0.323 &pm; 0.002</td>
    </tr>
    <tr>
        <td>0.0005</td> <td>0.685 &pm; 0.006</td> <td>0.597 &pm; 0.009</td> <td>0.261 &pm; 0.002</td>
    </tr>
    <tr>
        <td>0.0001</td> <td>0.579 &pm; 0.010</td> <td>0.368 &pm; 0.005</td> <td>0.220 &pm; 0.013</td>
    </tr>
    <tr>
        <td>0.00005</td> <td>0.507 &pm; 0.006</td> <td>0.295 &pm; 0.004</td> <td>0.208 &pm; 0.014</td>
    </tr>
    <tr>
        <td>0.00001</td> <td>0.304 &pm; 0.012</td> <td>0.223 &pm; 0.012</td> <td>0.156 &pm; 0.022</td>
    </tr>
</table>

### Experimental results
<figure style="text-align: center">
  <img src="./figures/cifar10.png" alt="Fig5" width="900">
</figure>

- <b>Fig. a</b> and <b>Fig. b</b> The evolution of the training loss and (top-1) test accuracy, respectively. The shaded area represents 95% confidence intervals.
- <b>Fig. c</b> and <b>Fig. d</b> The scatter and box plots of the average stepsize of five agents in the five algorithms. The error bar in the scatter plot represents standard deviation. Box plots show the median, 1st and 3rd quartiles, and 5th to 95th percentiles. P-values were calculated via one-way analysis of variance (ANOVA) with Tukey’s multiple comparison test.
- <b>Fig. e</b> The test-accuracy evolution of Algorithm 3 (with Q=1 and Q=15, respectively), ATC-DIGing [R6], and DSGD [R2], respectively, with respect to communication rounds.
- <b>Fig. f</b> The (top-1) test-accuracy evolution of Algorithm 3 under different numbers of local iterations Q with respect to communication rounds.

<b>Result discussion:</b> Fig. a and Fig. b demonstrate that Algorithm 3 with Q=1 (i.e., Algorithm 1 using noisy rather than exact gradients) achieves a lower training loss and higher test accuracy compared with DADAM, DAMSGrad, and DSGD with Nesterov’s momentum. While DADAM is slightly faster than our algorithm in the first 20 epochs, its performance is highly dependent on careful tuning, which is not a trivial task. If we do not tune DADAM, it almost always underperforms our algorithm, as evidenced by Table S2. In contrast, our Algorithm 3 is tuning free. Furthermore, we plotted the test-accuracy-vs-communication-rounds trajectory in Fig. e, which confirms that our tuning-free algorithm has lower communication complexity compared with DSGD [R2] and ATC-DIGing [R6] under a circle graph. In addition, Fig. f shows that Algorithm 3 with a moderate value of Q in the CNN training (e.g., Q=15 or Q=25) can dramatically reduce communication overhead and speed up convergence (e.g., as shown in Fig. f, Algorithm 3 with Q=15 reaches a 0.7 test accuracy using only 200 communication rounds, whereas Algorithm 3 with Q=1 requires over 1000 communication rounds to achieve the same level of test accuracy). However, an excessively large value of Q (e.g., Q=50) will compromise convergence accuracy. Q should be chosen based on application scenarios in practice.

> Note: Training parameters include a batch size of 128 and training durations of 70 epochs. We evaluated the stability and effectiveness of our algorithm and comparison algorithms using 95% confidence intervals derived from 3 runs under different random seeds: 42, 2024, and 1010. All experimental results (e.g., training loss, test loss, and test accuracy) will be automatically saved as `.csv` files in the `./Neural_networks/results` directory.

### ImageNet
1. You can use the following command to execute Algorithm 3 for the conventional neural network (CNN) training task on the ImageNet dataset:
   ```shell
   python main.py --test_num 0 --epochs 10 --batch_size 128 --dataset 'imagenet'
   ```
   - `--test_num`: specifies the optimization algorithm to be trained: `0`:Algorithm 3; `1`: DADAM [R3]; `2`: DAMSGrad [R4]; `3`: DSGD-N [R5]; `4`: ATC-DIGing [R6]; `5`: DSGD [R2].
   - `--epochs`: sets the number of training epochs.
   - `--batch_size`: sets the batch size for training.
   - `--dataset`: specifies the dataset to be used for training. The default option is 'cifar10'.
  
   >Note: Before running the script, please ensure that the [`ImageNet`](https://image-net.org) dataset has been downloaded and splited. Moreover, please ensure that the training and test sets are placed in    
    the `./Neural_networks/data/imagenet/train` and `./Neural_networks/data/imagenet/sort_val` directories, respectively.

2. To execute Algorithm 3 with a desired number of asynchronous-parallel-update iterations Q (e.g., Q=25), you can run the following command:
   ```shell
   python main.py --test_num 0 --epochs 10 --batch_size 128 --dataset 'imagenet' --const_q 25
   ```
3. To specify the print interval (e.g., printing the training loss, test accuracy, and average stepsize every 10 iterations), you can run the following command:
   ```shell
   python main.py --test_num 0 --epochs 10 --batch_size 128 --dataset 'imagenet' --interval 10
   ```
4. To specify the random seed used in training (e.g., setting seed=42), you can run the following command:
   ```shell
   python main.py --test_num 0 --epochs 10 --batch_size 128 --dataset 'imagenet' --seed 42
   ```
5. In this experiment, we used the standard ResNet-18 architecture [R7] with the ReLU activation function applied to all layers. 

6. We used the same learning-rate parameters for DADAM, DAMSGrad, and DSGD with Nesterov's momentum as those used in the CIFAR-10 experiment. This is because the large size of the ImageNet dataset (which includes 1.28 million training images, 50,000 validation images, and 100,000 test images across 1,000 categories) makes learning-rate tuning extremely time-consuming. This also highlights the advantage of our tuning-free algorithm, which does not need any manual tuning.

### Experimental results
<div style="text-align: center">
  <img src="./figures/imagenet.png" alt="Fig5" width="900">
</div>

- <b>Fig. a</b> and <b>Fig. b</b> The evolution of the top-1 test accuracy and top-5 test accuracy, respectively. The shaded area represents 95% confidence intervals.
- <b>Fig. c</b> and <b>Fig. d</b> The scatter and box plots of the average stepsize of five agents in the five algorithms. The error bar in the scatter plot represents standard deviation. Box plots show the median, 1st and 3rd quartiles, and 5th to 95th percentiles. P-values were calculated via one-way analysis of variance (ANOVA) with Tukey’s multiple comparison test.
- <b>Fig. e</b> The top-5 test accuracy evolution of Algorithm 3, ATC-DIGing [R6], and DSGD [R2], respectively, with respect to communication rounds.
- <b>Fig. f</b> The top-5 test accuracy evolution of Algorithm 3 under different numbers of local iterations Q with respect to communication rounds.

<b>Result discussion:</b> Fig. a and Fig. b confirm that our Algorithm 3 with Q=1 outperforms existing counterpart algorithms. In addition, Fig. e and Fig. f once again confirm the advantage of our Algorithm 3 with asynchronous parallel updates in reducing communication complexity and the used wallclock time.

> Note: Training parameters include a batch size of 128 and training durations of 10 epochs. We evaluated the stability and effectiveness of our algorithm and comparison algorithms using 95% confidence intervals derived from 3 runs under different random seeds: 42, 2024, and 1010. All experimental results (e.g., training loss, top-1 test accuracy, and top-5 test accuracy) will be automatically saved as `.csv` files in the `./Neural_networks/results` directory.

## Conclusions
This repository provides code for implementing our algorithms and the comparison algorithms in three typical machine learning applications: logistic regression on the mushrooms dataset, matrix factorization on the MovieLens 100k dataset, and image classification on the CIFAR-10 dataset and the ImageNet dataset, respectively. All experimental results demonstrate that Algorithm 1 achieves higher convergence accuracy than existing counterpart algorithms. Moreover, the experimental results for Algorithm 2 confirm the effectiveness of our stepsize-automation approach even under inexact consensus operations. The experimental results for Algorithm 3 demonstrate the effectiveness of our approach in asynchronous-parallel-update settings. To the best of our knowledge, this is the first repository to successfully automate stepsize selection in decentralized optimization and learning without requiring any manual parameter tuning or incurring accuracy degradation.

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## References
[R1] [DGM-BB-C](https://doi.org/10.1007/s11432-020-3256-x)

[R2] [DGD (DSGD)](https://doi.org/10.1137/130943170)

[R3] [DADAM](https://ieeexplore.ieee.org/document/9973382)

[R4] [DAMSGrad](https://openreview.net/pdf?id=CI-xXX9dg9l)

[R5] [DSGD-N](https://proceedings.mlr.press/v97/yu19d/yu19d.pdf)

[R6] [ATC-DIGing](https://ieeexplore.ieee.org/document/7963560)

[R7] [ResNet18](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

## Authors
- [Ziqin Chen](https://scholar.google.com/citations?user=i-IM2rIAAAAJ&hl=zh-CN)
- [Yongqiang Wang](https://scholar.google.com/citations?hl=zh-CN&user=shSZpGUAAAAJ)

