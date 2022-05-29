The task in this project is to train a deep neural network to forecast the need for tertiary reserves, i.e. to forecast
the remaining imbalance after the day-ahead and Intra-day markets are cleared. 

A similar system is currently under development for use in the upcoming **Automated Nordic mFRR energy activation market**.
The data available for training and running your model includes production plans and historical imbalance data
for area NO1. 

I use this data to construct datasets with appropriate features for your model, typically including estimated imbalances from the immediate past, 
as well as historical and future grid plans to forecast future imbalance. 
