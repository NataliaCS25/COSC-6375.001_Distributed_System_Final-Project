# COSC-6375.001_Distributed_System_Final-Project

Federated Short-Term Load Forecasting for ERCOT

Project Overview
This project implements a privacy-preserving Federated Learning (FL) framework to forecast short-term electricity demand across three ERCOT zones: North Central (NCENT), Coast (COAST), and Far West (FWEST). By utilizing a BiLSTM with a Self-Attention mechanism, the model captures complex temporal dependencies and regional weather correlations without requiring raw data to leave the local utility zones.

Technical Stack
Orchestration: Flower (flwr)

Simulation Engine: Ray

Deep Learning: PyTorch

Data Processing: Pandas, Scikit-learn (Min-Max Scaling)

Architecture: Bidirectional LSTM + Self-Attention

Overleaf Document: https://www.overleaf.com/1518385616qwmgpcwwfjzv#58bc45
