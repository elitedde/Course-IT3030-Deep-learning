**Project #1: Implementing Backpropagation**

The heart and soul of deep learning is a nearly-40-year-old algorithm known as backpropagation. Though numerous bells and whistles have supplemented backpropagation over the years, with some of these enhancements proving crucial to the scaling up of neural networks to truly deep learning, the core learning algorithm and framework is still backprop.

Although packages such as **Tensorflow** and **PyTorch** take (almost) all of the drudgery out of deep learning by abstracting away the essence of backpropagation – the calculation of gradients linking weights to loss – an understanding of these details seems quite essential. That understanding stems most readily from low-level implementation, and the skills hereby achieved provide a solid base for deeper investigations and actual AI research. 


My system enables users to build and run networks that have a wide range of sizes and types, from simple input-output networks with no hidden layers to those with up to 5 hidden layers. A limited variety of activation functions and regularization schemes will also be at the user’s disposal.
It also provides a module for generating datasets, and it defines a format and parser for network configuration files. These, along with a few simple visualization tools (for the data images and learning progress), constitute peripheral support for the two core processes of the backpropagation algorithm: the forward and backward passes.

