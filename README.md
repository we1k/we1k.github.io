##### Q: 为什么引入非线性激励函数？

A:如果不用激励函数（其实相当于激励函数是f(x) = x），在这种情况下你每一层输出都是上层输入的线性函数，很容易验证，无论你神经网络有多少层，输出都是输入的线性组合，与没有隐藏层效果相当，这种情况就是最原始的感知机（Perceptron）了

##### Q:Why Relu = max(0,x)

A:1.仿生物学原理：在一个使用修正线性单元（即线性整流）的神经网络中大概有50%的神经元处于激活态。

2.更加有效率的梯度下降以及反向传播：避免了梯度爆炸和梯度消失问题

3.简化计算过程：没有了其他复杂激活函数中诸如指数函数的影响；

##### Q: loss 函数中 mse 和 categorical_crossentropy（交叉熵损失函数）区别

交叉熵：实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近。0为最小值。

mse 在梯度下降时![img](https://upload-images.jianshu.io/upload_images/14512145-0689d9d95b5f9df7.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

sigmoid函数导数会在z取大部分值时会很小，这样会使得w和b更新非常慢

##### 动量（momentum）优化？ 学习率衰减（每次参数更新后）？

#### LeakyRelu：

**ReLU**
  ReLU函数代表的的是“修正线性单元”，它是带有卷积图像的输入x的最大函数(x,o)。ReLU函数将矩阵x内所有负值都设为零，其余的值不变。ReLU函数的计算是在卷积之后进行的，因此它与tanh函数和sigmoid函数一样，同属于“非线性激活函数”。这一内容是由Geoff Hinton首次提出的。
**ELUs**
  ELUs是“指数线性单元”，它试图将激活函数的平均值接近零，从而加快学习的速度。同时，它还能通过正值的标识来避免梯度消失的问题。根据一些研究，ELUs分类精确度是高于ReLUs的。下面是关于ELU细节信息的详细介绍：

![img](http://p0.ifengimg.com/pmop/2017/0701/A9B535C61C2D63E152DE2CEECB4531EE83E80208_size26_w740_h230.jpeg)

 

**Leaky ReLUs tanh** 
  ReLU是将所有的负值都设为零，相反，Leaky ReLU是给所有负值赋予一个非零斜率。Leaky ReLU激活函数是在声学模型（2013）中首次提出的。以数学的方式我们可以表示为：

  ![img](http://p0.ifengimg.com/pmop/2017/0701/CFC5A1C95A84A6D8CF3FFC1DD30597782AEEAE57_size20_w740_h231.jpeg)ai是（1，+∞）区间内的固定参数。

**参数化修正线性单元（PReLU）**
  PReLU可以看作是Leaky ReLU的一个变体。在PReLU中，负值部分的斜率是根据数据来定的，而非预先定义的。作者称，在ImageNet分类（2015，Russakovsky等）上，PReLU是超越人类分类水平的关键所在。
**随机纠正线性单元（RReLU）**
  “随机纠正线性单元”RReLU也是Leaky ReLU的一个变体。在RReLU中，负值的斜率在训练中是随机的，在之后的测试中就变成了固定的了。RReLU的亮点在于，在训练环节中，aji是从一个均匀的分布U(I,u)中随机抽取的数值。形式上来说，我们能得到以下结果：*
*

  ![img](http://p0.ifengimg.com/pmop/2017/0701/B3F2F3EA627EBB55D88C8F8FB36942C56B350A4B_size14_w740_h221.jpeg)

**总结**
  下图是ReLU、Leaky ReLU、PReLU和RReLU的比较：

  ![img](http://p0.ifengimg.com/pmop/2017/0701/C56E5C6FCBB36E70BA5EBC90CBD142BA320B3DF6_size19_w740_h217.jpeg)

#### 自己写metrics

from keras import backend as K

#### Upsampling2D() -> (7, 7 , 32) - > (14 ,14 ,32） 改变特征层的长和宽

#### Q:先BN还是先激活？

A: 先BN在激活。https://www.zhihu.com/question/318354788

Batch Norm方法经过规范化和缩放平移，可以使输入数据，重新回到非饱和区，还可以更进一步：控制激活的饱和程度，或是非饱和函数抑制与激活的范围。

从剃度消失的角度来看，比如sigmoid激活函数，两边的剃度很小，容易剃度消失，bn的作用是把输出拉回到非饱和区，就是剃度大的那部分，所以要先bn再激活。

##### Q: tf 中的padding 参数same, vaild

![image-20200905231349413](D:\ad\we1k.github.io\README.assets\image-20200905231349413.png)

outputs_size = (inputs + 2*padding - kernel_size ) // strides + 1

#### BatchNormalization, LayerNormalization作用:

标准正态分布更容易让梯度下降速度提高。

右图为标准化结果

![image-20200920092801730](D:\ad\we1k.github.io\README.assets\image-20200920092801730.png)

#### AUC评判标准：ROC曲线下的面积【0,1】

#### L1,L2正则化L2正则化

在深度学习中，用的比较多的正则化技术是L2正则化，其形式是在原先的损失函数后边再加多一项：$1/2λ\Sigmaθ^2_i$，那加上L2正则项的损失函数就可以表示为：L(θ)=L(θ)+λ∑niθ2iL(θ)=L(θ)+λ∑inθi2，其中θθ就是网络层的待学习的参数，λλ则控制正则项的大小，较大的取值将较大程度约束模型复杂度，反之亦然。

L2约束通常对稀疏的有尖峰的权重向量施加大的惩罚，而偏好于均匀的参数。这样的效果是鼓励神经单元利用上层的所有输入，而不是部分输入。所以L2正则项加入之后，权重的绝对值大小就会整体倾向于减少，尤其不会出现特别大的值（比如噪声），即网络偏向于学习比较小的权重。所以L2正则化在深度学习中还有个名字叫做“权重衰减”（weight decay），也有一种理解这种衰减是对权值的一种惩罚，所以有些书里把L2正则化的这一项叫做惩罚项（penalty）。

我们通过一个例子形象理解一下L2正则化的作用，考虑一个只有两个参数w1w1和w2w2的模型，其损失函数曲面如下图所示。从a可以看出，最小值所在是一条线，整个曲面看起来就像是一个山脊。那么这样的山脊曲面就会对应无数个参数组合，单纯使用梯度下降法难以得到确定解。但是这样的目标函数若加上一项0.1×(w21+w22)0.1×(w1^2+w2^2)，则曲面就会变成b图的曲面，最小值所在的位置就会从一条山岭变成一个山谷了,此时我们搜索该目标函数的最小值就比先前容易了，所以L2正则化在机器学习中也叫做“岭回归”（ridge regression）。

![img](https://images2018.cnblogs.com/blog/1093303/201802/1093303-20180221174027599-1004937268.png)

### Attention：

##### Why not seq2seq model？ 

the encoder and decoder of a seq2seq model. 

- The encoder process the input sequence and compresses the information into a **context vector**. BUT The context vector is a fixed length vector. This process is referred as **Embedding**. which is expected to get a good summary of the whole input senquence.

- A **decoder** is initialized with the context vector to emit the transformed output. The early work only used the last state of the encoder network as the decoder initial state.

  ![encoder-decoder model with additive attention layer](https://lilianweng.github.io/lil-log/assets/images/encoder-decoder-example.png)

The secret sauce of attention is to **create a shortcuts between the context vector and the entire input sequence**.

Essentially the context vector consumes three pieces of information:

- encoder hidden states;
- decoder hidden states;
- alignment between source and target.

![encoder-decoder model with additive attention layer](https://lilianweng.github.io/lil-log/assets/images/encoder-decoder-attention.png)

#### Skip-gram

.More formally, given a sequence of training words w1, w2, w3, . . . , wT , the objective of the Skip-gram model is to maximize the average log probability 

![image-20201109143518576](D:\ad\we1k.github.io\README.assets\image-20201109143518576.png)

 where c is the size of the training context (which can be a function of the center word wt). Larger c results in more training examples and thus can lead to a higher accuracy, at the expense of the 2 training time. The basic Skip-gram formulation defines p(wt+j |wt) using the softmax function:

![image-20201109143602058](D:\ad\we1k.github.io\README.assets\image-20201109143602058.png)

where $v_w$ and $v ′_ w$ are the “input” and “output” vector representations of w, and W is the number of words in the vocabulary. This formulation is impractical because the cost of computing gradient p(wo|wi) is proportional to W.

#### Negative Sampling

While NCE can be shown to approximately maximize the log probability of the softmax, the Skip-gram model is only concerned with learning high-quality vector representations, so we are free to simplify NCE as long as the vector representations retain their quality. We define Negative sampling (NEG) by the objective

MAX : $log σ(v ′ {wO}^ ⊤ v_{wI} ) +\sum^ k _{i=1} Ewi∼Pn(w) [log σ(−v ′ _{wi}^ ⊤ v_{wI} ) ]$

which is used to replace every log P(wO|wI ) term in the Skip-gram objective. Thus the task is to distinguish the target word wO from draws from the noise distribution Pn(w) using logistic regression, where there are k negative samples for each data sample. 





**判别式模型**（Discriminative Model）是直接对条件概率p(y|x;θ)建模。常见的判别式模型有 线性回归模型、线性判别分析、支持向量机SVM、神经网络等。

**生成式模型**（Generative Model）则会对x和y的联合分布p(x,y)建模，然后通过贝叶斯公式来求得p(yi|x)，然后选取使得p(yi|x)最大的yi，即：