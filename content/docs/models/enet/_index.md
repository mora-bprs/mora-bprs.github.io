---
date: "2024-04-12T14:46:31+05:30"
title: ENet
next: /docs/models/enet/paper
prev: /docs/models
---

Later in our review of literature published related to segmentation models, we came across ENet (Efficient Neural Network), which is termed as a real-time semantic segmentation neural network designed for efficient processing on embedded systems and mobile devices. It was developed by researchers at Samsung AI Center and first presented in the paper "ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation" published in 2016 [6].

Reviewing the research paper of ENet, it was found that performance wise and efficiency wise it is better than SegNet [5].

{{< cards >}}
{{< card link="https://arxiv.org/pdf/1606.02147.pdf" title="ENet Paper" icon="document-text" subtitle="arxiv pdf link" >}}
{{< /cards >}}

The key advantages of ENet are its high speed, low computational requirements, and smaller memory footprint compared to existing models, while maintaining comparable accuracy. Considering the viability of the model for our project which mainly focuses on deploying a reliable model to identify boxes in an industrial environment, a suggestion was made at a discussion with the supervisor to consider this model, as it would allow us to deploy such a model in edge environments where memory and process is a constrained resource. The suggestion was reviewed and approved by the supervisor later and it lead to the transition from SegNet to ENet.
