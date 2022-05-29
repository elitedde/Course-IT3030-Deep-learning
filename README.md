# Time series forecasting
Individual projects

The electric power grid is a complex network of power lines, substations and transformers that connects producers
(wind farms, hydropower plants, nuclear power plants, etc.) to consumers (end-users such as residential homes,
industry, etc.). The grid does not store energy, so in order to keep the power system balanced, the demand from
consumers must be met continuously and instantaneously by the producers. Failure to balance the power system leads
to frequency deviations and, in extreme cases, outages. It is the responsibility of the Transmission System Operator
(TSO) to ensure that the system is balanced at all times. In Norway, Statnett is the TSO and is responsible for
keeping the frequency between 49.9 and 50.1 Hz.
The power system is balanced through a number of steps involving different markets. In the Nordic countries,
producers and consumers settle prices and volumes for the next day on the Nord Pool day-ahead market, and may
subsequently trade on the Intra-day market to restore balance whenever their plans or expectations change As we move closer to real time, a series of reserve markets ensure that balance is
maintained at all time and in the presence of unexpected fluctuations in production, consumption or transmission.
The Nordic Synchronous Area consists of Norway, Sweden, Finland and parts of Denmark. The frequency (which
should always be 50 Hz) represents the overall balance or imbalance for this grid as a whole. However, in the markets
and during grid operation, the grid is divided into smaller areas, as shown in Figure 1.
Your task in this project is to train a deep neural network to forecast the need for tertiary reserves, i.e. to forecast
the remaining imbalance after the day-ahead and Intra-day markets are cleared. A similar system is currently under
development for use in the upcoming Automated Nordic mFRR energy activation market.
The data available for training and running your model includes production plans and historical imbalance data
for area NO1. You should use this data to construct datasets with appropriate features for your model, typically
including estimated imbalances from the immediate past, as well as historical and future grid plans to forecast future
imbalance. The following sections will flesh out more practical information on the data, as well as some general tips
for sequence modelling and the different sub-tasks.
