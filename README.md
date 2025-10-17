# Shared Representation Learning for Generalizable SOH Estimation Across Multiple Battery Configurations 

## Authors:

Shunyu Wu*, Zhuomin Chen*, Bingxin Lin, Haozheng Ye, Jiahui Zhou, Dan Li†, Jian Lou
School of Software Engineering, Sun Yat-sen University, Zhuhai, China
*Equal contribution †Corresponding author

📧 {wushy88, chenzhm39, linbx25, yehzh8, zhoujh99}@mail2.sysu.edu.cn, {lidan263, louj5}@mail.sysu.edu.cn

## 🧩 Overview

Battery health monitoring is crucial for applications such as electric vehicles (EVs) and energy storage systems (ESS), where accurate State of Health (SOH) estimation ensures both safety and cost efficiency.
However, traditional models are often limited to single battery types or conditions, resulting in poor generalization across different configurations.

To overcome this limitation, we propose SRSE (Shared Representation learning for SOH Estimation) — a novel framework that jointly learns shared representations across multiple battery configurations to achieve robust, generalizable, and accurate SOH estimation.

## 🚀 Key Features

Shared Representation Learning:
Learns invariant knowledge across battery configurations to capture cross-domain correlations.

Adversarial Domain Alignment:
Utilizes adversarial training to remove task-specific bias in the shared feature space.

Feature- and Logit-Level Knowledge Sharing:
Transfers knowledge from shared layers to task-specific branches for enhanced adaptation.

Generalization Across Configurations:
Achieves consistent SOH estimation across diverse battery types and operational conditions.
