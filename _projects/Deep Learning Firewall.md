---
layout: page
title: Deep Learning Firewall
description: A firewall system utilizing deep learning to filter malicious network traffic
img: assets/img/deep_learning_firewall.jpg
importance: 1
category: work
related_publications: true
---

The **Deep Learning Firewall** project aims to enhance traditional firewall systems by employing deep learning algorithms to detect and block malicious network traffic. This system predicts and prevents cyber-attacks using machine learning techniques.

### Backend
- **Language**: Python
- **Frameworks**: TensorFlow, Flask, Keras

### Frontend
- **Framework**: Flask

### Dataset
- **Name**: CSIC HTTP Attacks Dataset
  - Contains 36,000 normal traffic samples and over 25,000 malicious traffic samples.

### Modular Description
1. **Data Pre-processing**: Processed HTTP requests to extract useful features.
2. **Model Training**: Trained a model to classify requests as normal or malicious.
3. **Integration**: Connected the model to a sample web application.
4. **Detection**: Detected and blocked malicious requests, logging IPs.

### Model Architecture
The model architecture consists of embedding layers, convolutional layers, and pooling layers to efficiently process and classify HTTP requests.

### Illustrations
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/firewall_output_normal.jpg" title="Normal Request Output" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/firewall_output_malicious.jpg" title="Malicious Request Output" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Future Extensions
- **Intrusion Detection Systems (IDS)**: Use as an IDS for network protection.
- **Blockchain Protection**: Adapt the model for protecting blockchain applications.
- **Antivirus Integration**: Classify different HTTP attacks for antivirus use.
