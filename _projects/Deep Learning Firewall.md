---
layout: page
title: Deep Learning Firewall
description: A firewall system utilizing deep learning to filter malicious network traffic
img: assets/img/deep_learning_firewall.jpg
importance: 1
category: Machine_Learning
related_publications: true
---
## Deep Learning Firewall

The **Deep Learning Firewall** project focuses on developing a firewall system that uses deep learning algorithms to intelligently filter and block malicious network traffic. This project enhances traditional firewall capabilities by incorporating advanced machine learning techniques to predict and prevent cyber-attacks.

### Backend

- **Language**: Python

- **Frameworks**: TensorFlow, Flask and Keras

### Frontend

- **Framework**: Flask 

### Dataset

- **Name**: CSIC HTTP Attacks Dataset

### Model Architecture

![](https://raw.githubusercontent.com/sachindots/Projects/refs/heads/main/Diagrams/dlf.png)

### Modular Description

#### Data Pre-processing

- **Dataset**: HTTP DATASET CSIC 2010

  - Produced by the Spanish Research National Council (CSIC).

  - Summary: 36,000 normal traffic and over 25,000 malicious traffic.

  - Includes attacks such as SQL injection, buffer overflow, and information gathering queries.

- **Approach**:

  - The raw dataset contains HTTP requests with URL encoding.

  - Challenges include extracting useful features due to a large number of escape characters (`%`).

#### Model Training

- **Embedding Layer**:

  - Converts each letter of the input HTTP request sentence into a 128-dimensional vector.

- **Architecture**:

  - Characterized by significant dimensional reduction at the second max-pooling layer (498:1 at K=2).

  - Greatly reduces the number of fully connected (FC) elements in subsequent layers.

  - Reduces the size of GPU memory required.

#### Connection of WAF

- **Integration**:

  - The created model is connected with a sample web application using the Flask framework in Python.

  - A proxy is created between the model and the sample application to evaluate incoming requests.

#### Malicious Query Detection

- **Process**:

  - Malicious requests executed to the sample application are detected and blocked.

  - User IP addresses of malicious requests are logged and displayed in the WAF panel.

### Illustrations

Figure 1.1 Sample Web Application Output when applied normal request
![](https://raw.githubusercontent.com/sachindots/Projects/refs/heads/main/Diagrams/1.1.png)

Figure 1.2 Sample Web Application Output when applied malicious request
![](https://raw.githubusercontent.com/sachindots/Projects/refs/heads/main/Diagrams/1.2.png)

Figure 2.1 WAF Dashboard for the sample web application
![](https://raw.githubusercontent.com/sachindots/Projects/refs/heads/main/Diagrams/2.1.png)



### Future Extensions

**Intrusion Detection Systems (IDS)**:

  -  The WAF can be used as an IDS in a large-scale network to protect systems from major cyber attacks.

**Blockchain Protection**:

  - The firewall can be modified to act as an enhanced protection system for bitcoin mines and decentralized applications (DApps).

**Antivirus**:

  - Antivirus systems can use this model to classify different types of HTTP attacks that pose a threat to a computer and block the malicious actors.


---

Please note that the source code for this project is confidential and not publicly available, as it is part of my academic research. Feel free to explore the project description and also reach out if you have any questions or need further assistance.
