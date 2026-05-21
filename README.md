# Training CNN Rotation Robustness: Is Denser Data Augmentation Worth the Cost?

A common way of training convolutional neural networks to better classify rotated images is through data augmentation. In this process, we typically use one rotated version of every training image to train our model on. What happens when, instead, we train our model on $m>1$ rotated versions?

![Alt Text](figures/cnn_aug_m.jpg)

While a standard non-robust CNN may produce accuracy scores of roughly **42%** on rotated test images, we improve this score to about **90%** through dense data augmentation. 

The paper explores theoretical bounds and rates of convergence with regards to model accuracy gains. For rotation groups $C_k$ and $SO(2)$, we derive model robustness concentration zones for $m$. We then observe and discuss the tradeoff between rotation robustness gains and computational costs.
