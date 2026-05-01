# COSC-6375.001_Distributed_System_Final-Project

# Project: Ensuring Grid Resilience through Responsible Federated Learning: A Multi-Pillar Audit of Distributed Short-Term Load Forecasting

This repository contains the implementation of a privacy-preserving framework designed for Short-Term Load Forecasting (STLF) in electrical grids. The project addresses the critical challenge of balancing forecasting accuracy with data privacy and model trustworthiness in distributed energy systems.

# Core Objectives
The primary goal of this research is to enable multiple utility regions to collaboratively train deep learning models without sharing sensitive, raw operational data. By leveraging Federated Learning (FL), the framework maintains regional data sovereignty while benefiting from the collective patterns found across diverse geographical areas.


# Technical Architecture
- Dataset: Utilizes ten years of hourly energy demand data from the ERCOT (Texas) market (2015–2025), specifically focusing on the North Central, Coast, and Far West regions.  
- Algorithms: Evaluates a variety of architectures, including Linear Regressors and Long Short-Term Memory (LSTM) networks, to determine the optimal balance between complexity and performance.  
- Optimization: Implements the FedProx algorithm via the Flower framework to handle the Non-IID nature of regional power consumption patterns.

# Technical Stack
- Orchestration: Flower (flwr)
- Simulation Engine: Ray
- Deep Learning: PyTorch
- Data Processing: Pandas, Scikit-learn (Min-Max Scaling)

# The Five Pillars of Responsible AI (RAI)
Unlike standard forecasting tools, this project incorporates a comprehensive Multi-Pillar RAI Audit to ensure the model is ready for critical infrastructure deployment:
1. Explainability: Uses SHAP feature attribution to visualize how temporal and weather variables drive specific load predictions.  
2. Fairness: Evaluates performance across different regions to ensure the model does not disproportionately favor one utility zone over another.  
3. Robustness: Assesses model stability and error margins (NMBE) to provide operators with a reliable "safety buffer".  
4. Privacy: Ensures that only model gradients—never raw data—are exchanged during the training process.  
5. Accountability: Generates automated Model Cards (JSON) to document performance metrics and intended use cases for regulatory compliance

# Key Results
- Collaborative Gains: The Federated BiLSTM model showed a 15.4% improvement in forecasting accuracy over local models trained in isolation.  
- Minimal Privacy Tax: The gap between centralized (data-sharing) and federated (privacy-preserving) models was negligible, proving that privacy does not have to come at the cost of performance.

# Overleaf Document
Overleaf Link: https://www.overleaf.com/8757582595wrnjqxcnwkcg#f9c4df
