# adversarial-ml
Adversarial ML
This repository is based on [yenchenlin](https://github.com/yenchenlin/awesome-adversarial-machine-learning) and [tanjuntao](https://github.com/tanjuntao/Adversarial-Machine-Learning). I will be adding more resources as I come across them. 

## Table of Contents

 - [Blogs](#blogs)
 - [Papers](#papers)
 - [Talks](#talks)

## Blogs
 * [Breaking Linear Classifiers on ImageNet](http://karpathy.github.io/2015/03/30/breaking-convnets/), A. Karpathy et al.
 * [Breaking things is easy](http://www.cleverhans.io/security/privacy/ml/2016/12/16/breaking-things-is-easy.html), N. Papernot & I. Goodfellow et al.
 * [Attacking Machine Learning with Adversarial Examples](https://blog.openai.com/adversarial-example-research/), N. Papernot, I. Goodfellow, S. Huang, Y. Duan, P. Abbeel, J. Clark.
 * [Robust Adversarial Examples](https://blog.openai.com/robust-adversarial-inputs/), Anish Athalye.
 * [A Brief Introduction to Adversarial Examples](http://people.csail.mit.edu/madry/lab/blog/adversarial/2018/07/06/adversarial_intro/), A. Madry et al.
 * [Training Robust Classifiers (Part 1)](http://people.csail.mit.edu/madry/lab/blog/adversarial/2018/07/11/robust_optimization_part1/), A. Madry et al.
 * [Adversarial Machine Learning Reading List](https://nicholas.carlini.com/writing/2018/adversarial-machine-learning-reading-list.html), N. Carlini
 * [Recommendations for Evaluating Adversarial Example Defenses](https://nicholas.carlini.com/writing/2018/evaluating-adversarial-example-defenses.html), N. Carlini

 
## Papers
### General
 * [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199), C. Szegedy et al., arxiv 2014
 * [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), I. Goodfellow et al., ICLR 2015
 * [Motivating the Rules of the Game for Adversarial Example Research](https://arxiv.org/abs/1807.06732), J. Gilmer et al., arxiv 2018
 * [Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning](https://arxiv.org/abs/1712.03141), B. Biggio, Pattern Recognition 2018

### Attack
**Image Classification**

 * [DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/abs/1511.04599), S. Moosavi-Dezfooli et al., CVPR 2016
 * [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/abs/1511.07528), N. Papernot et al., ESSP 2016
 * [Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples](https://arxiv.org/abs/1605.07277), N. Papernot et al., arxiv 2016
 * [Adversarial Examples In The Physical World](https://arxiv.org/pdf/1607.02533v3.pdf), A. Kurakin et al., ICLR workshop 2017 
 * [Delving into Transferable Adversarial Examples and Black-box Attacks](https://arxiv.org/abs/1611.02770) Liu et al., ICLR 2017
 * [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644) N. Carlini et al., SSP 2017
 * [Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples](https://arxiv.org/abs/1602.02697), N. Papernot et al., Asia CCS 2017
 * [Privacy and machine learning: two unexpected allies?](http://www.cleverhans.io/privacy/2018/04/29/privacy-and-machine-learning.html), I. Goodfellow et al.

**Reinforcement Learning**

* [Adversarial attacks on neural network policies](https://arxiv.org/abs/1702.02284), S. Huang et al, ICLR workshop 2017
* [Tactics of Adversarial Attacks on Deep Reinforcement Learning Agents](https://arxiv.org/abs/1703.06748), Y. Lin et al, IJCAI 2017
* [Delving into adversarial attacks on deep policies](https://arxiv.org/abs/1705.06452), J. Kos et al., ICLR workshop 2017

**Segmentation & Object Detection**

* [Adversarial Examples for Semantic Segmentation and Object Detection](https://arxiv.org/pdf/1703.08603.pdf), C. Xie, ICCV 2017

**VAE-GAN**

* [Adversarial examples for generative models](https://arxiv.org/abs/1702.06832), J. Kos et al. arxiv 2017

**Speech Recognition**

* [Audio Adversarial Examples: Targeted Attacks on Speech-to-Text](https://arxiv.org/abs/1801.01944), N. Carlini et al., arxiv 2018

**Questiona Answering System**

* [Adversarial Examples for Evaluating Reading Comprehension Systems](https://arxiv.org/abs/1707.07328), R. Jia et al., EMNLP 2017

### Defence

**Adversarial Training**

* [Adversarial Machine Learning At Scale](https://arxiv.org/pdf/1611.01236.pdf), A. Kurakin et al., ICLR 2017
* [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204), F. Tramèr et al., arxiv 2017

**Defensive Distillation**
* [Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks](https://arxiv.org/pdf/1511.04508.pdf), N. Papernot et al., SSP 2016
* [Extending Defensive Distillation](https://arxiv.org/abs/1705.05264), N. Papernot et al., arxiv 2017

**Generative Model**
* [PixelDefend: Leveraging Generative Models to Understand and Defend against Adversarial Examples](https://arxiv.org/abs/1710.10766), Y. Song et al., ICLR 2018
* [Detecting Adversarial Attacks on Neural Network Policies with Visual Foresight](https://arxiv.org/abs/1710.00814), Y. Lin et al., NIPS workshop 2017

### Regularization
 * [Distributional Smoothing with Virtual Adversarial Training](https://arxiv.org/abs/1507.00677), T. Miyato et al., ICLR 2016
 * [Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725), T. Miyato et al., ICLR 2017

### Others
 * [Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images](https://arxiv.org/abs/1412.1897), A. Nguyen et al., CVPR 2015
 
## Talks
 * [Do Statistical Models Understand the World?](https://www.youtube.com/watch?v=Pq4A2mPCB0Y), I. Goodfellow, 2015
 * [Classifiers under Attack](https://www.usenix.org/conference/enigma2017/conference-program/presentation/evans), David Evans, 2017
  * [Adversarial Examples in Machine Learning](https://www.usenix.org/conference/enigma2017/conference-program/presentation/papernot), Nicolas Papernot, 2017
  * [Poisoning Behavioral Malware Clustering](http://pralab.diee.unica.it/en/node/1121), Biggio. B, Rieck. K, Ariu. D, Wressnegger. C, Corona. I. Giacinto, G. Roli. F, 2014
  * [Is Data Clustering in Adversarial Settings Secure?](http://pralab.diee.unica.it/en/node/955), BBiggio. B, Pillai. I, Rota Bulò. S, Ariu. D, Pelillo. M, Roli. F, 2015
  * [Poisoning complete-linkage hierarchical clustering](http://pralab.diee.unica.it/en/node/1089), Biggio. B, Rota Bulò. S, Pillai. I, Mura. M, Zemene Mequanint. E, Pelillo. M, Roli. F, 2014
  * [Is Feature Selection Secure against Training Data Poisoning?](https://pralab.diee.unica.it/en/node/1191), Xiao. H, Biggio. B, Brown. G, Fumera. G, Eckert. C, Roli. F, 2015
  * [Adversarial Feature Selection Against Evasion Attacks](https://pralab.diee.unica.it/en/node/1188), 	Zhang. F, Chan. PPK, Biggio. B, Yeung. DS, Roli. F, 2016
    
## Books
 * https://www.cambridge.org/core/books/adversarial-machine-learning/C42A9D49CBC626DF7B8E54E72974AA3B
 * https://www.sciencedirect.com/book/9780128240205/adversarial-robustness-for-machine-learning
 * https://christophm.github.io/interpretable-ml-book/adversarial.html


### Useful Links
* https://evademl.org/
* https://secml.github.io/
* http://www.cleverhans.io/
* https://aaai18adversarial.github.io/ 
* https://www.openmined.org/ 
* https://www.pluribus-one.it/
* https://www.ieee-security.org/TC/SPW2018/DLS/
* https://robust.vision/benchmark

## Other resources
Link | Type | Description    
-----|-----|----
[ cleverhans ]( https://github.com/tensorflow/cleverhans) | Attack & Defense | ` AML` field originator repo , developed by *Goodfellow* & * Papernot * , provides attack methods and defense methods .
[ foolbox ]( https://github.com/bethgelab/foolbox/) | Attack | The main function is to generate adversarial samples. It implements about **15 ** attack methods and does not provide defense functions .
[adversarial-robustness- toolbox]( https://github.com/IBM/adversarial-robustness-toolbox) (ART) | Attack & Defense | Provides a large number of attack methods and defense methods, the API is easy to call, and Several methods for detecting adversarial examples are provided .
[machine_learning_adversarial_ examples]( https://github.com/rodgzilla/machine_learning_adversarial_examples) | Attack | Mainly reproduces the ** FGSM algorithm ** from the paper `Explaning and Harnessing Adversarial Examples` . 
[Adversarial_Learning_Paper ]( https://github.com/Guo-Yunzhe/Adversarial_Learning_Paper) | awesome | ` AML` related paper list, including `Survey` , `Attack`, `Defense`
[AdversarialDNN- Playground]( https://github.com/QData/AdversarialDNN-Playground)| Visualization | Visualization of the attack process, and a certain analysis of the attack method at the theoretical level (see the presentation in the warehouse )
[awesome-adversarial-machine- learning]( https://github.com/yenchenlin/awesome-adversarial-machine-learning) | awesome | Summarize `blogs` , `papers`, `talks` related to `AML`
[ AdvBox ]( https://github.com/baidu/AdvBox) | Attack & Defense | Baidu products, provide various attack and defense methods, support command line to directly generate adversarial samples ( zero-coding )
[adversarial- examples]( https://github.com/ifding/adversarial-examples)| Attack | At the theoretical level, several commonly used attack methods are provided; at the practical level, attacks on ** road signs ** .
[adversarial_ examples]( https://github.com/duoergun0729/adversarial_examples) | Attack | Provides several common attack methods and makes a chart analysis .
[Adversarial-Examples-Reading- List]( https://github.com/chawins/Adversarial-Examples-Reading-List) | awesome | ` AML` related papers list, including `attacks` , `defenses` . The author is a PhD student at UC Berkeley .
[ nn_robust_attacks]( https://github.com/carlini/nn_robust_attacks) | Attacks | Paper `Towards Evaluating the Robustness of Neural Networks` code . The author graduated from UC Berkeley with a Ph.D.
[awesome-adversarial-examples- dl]( https://github.com/chbrian/awesome-adversarial-examples-dl) | awesome | ` AML` paper list, including `Attack` , `Defense`, `Application`
[ FeatureSqueezing ]( https://github.com/uvasrg/FeatureSqueezing) | Defense | *Detecting Adversarial Examples in Deep Neural Network* . The project stopped maintenance and moved to [ EvadeML -Zoo](http://evademl.org/zoo/)
[adversarial-example- pytorch]( https://github.com/sarathknv/adversarial-examples-pytorch) | attack | pytorch Realize common attack methods, and provide ** visualization ** function .
[ EvadeML - Zoo]( https://github.com/mzweilin/EvadeML-Zoo) | Attack & Defense | Provides pre-trained models, common data sets, common attack methods; visualized adversarial examples .
[robust-physical- attack]( https://github.com/shantse/robust-physical-attack) | Attack | Attack`Faster in realistic situations The R-CNN` target detection model provides `targeted` and ` untargeted` two attack methods .
[ advertorch ]( https://github.com/BorealisAI/advertorch) | Attack & Defense | [ foolbox ](https://github.com/bethgelab/foolbox/) is a streamlined version , implemented in ` pytorch` , which only provides a part of the attack and defense methods .
[Non-Targeted-Adversarial- Attacks ]( https://github.com/dongyp13/Non-Targeted-Adversarial-Attacks )| Attack | /c/nips-2017-defense-against-adversarial-attack) `Non-targeted attack` first place .
[Targeted-Adversarial- Attack]( https://github.com/dongyp13/Targeted-Adversarial-Attack)| Attack | It is also the first place in `Targeted attack` in the NIPS 2017 Attack and Defense Competition , and the author is Tsinghua University * * Zhu Jun ** team .
[artificial- adversary]( https://github.com/airbnb/artificial-adversary) | Attack | Adversarial examples for text modalities , developed by `airbnb` .

### Related Contests
* [NIPS 2017: Defense Against Adversarial Attack]( https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack)
* [IJCAI-19 Alibaba Artificial Intelligence Contest Algorithm Competition ]( https://tianchi.aliyun.com/competition/entrance/231701/introduction?spm=5176.12281905.5490641.4.358b6bad39hWbP)
