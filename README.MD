# Algorithmic Trading With Python using Zipline

## Setup
Follow the instructions on the zipline tutorial for installation.  You
need python 3.5 or 3.6 for it to work.  Installing with conda seemed
to be most seamless.

Make sure that you set up a new python project with the conda
environment as the project interpreter, or else you won't have
access to zipline.

Once it's installed check out the algo.py file and copy/paste it 
into your local directory.

## The Basics

Now is where our algorithm starts.  The ```initialize``` method 
is run once per backtest, so when we run our algorithm 
we basically tell it what stocks we want to lookup and 
the symbol ids.  Symbol ids are important for the complicated reason
that sometimes ticker symbols change, but symbol ids are unique 
to each asset.  
```python
STOCKS = ['AMD', 'CERN', 'COST', 'DELL', 'GPS', 'INTC', 'MMM']

def initialize(algo):
    algo.stocks = STOCKS
    algo.symbol_ids = [algo.symbol(asset) for asset in STOCKS]
```
Here we tell it that we want our algorithm to run on these stocks.  
Also, the algo keyword might be a bit confusing, but it basically
acts as a ```self``` parameter for the algorithm.

Now to buy and sell stocks we need to tell the algorithm how much of 
each stock to buy and sell when we execute an order.  
The ```make_weights``` function is an easy way for us to tell the 
algorithm that we want to order a percent of the stock based on
how much the short moving average is above the long moving average.

```python
def make_weights(algo, data):
    recent = data.history(assets=algo.symbol_ids, 
                          fields='close', 
                          bar_count=60, 
                          frequency='1d')
    short_mean = recent[-10:].mean()
    long_mean = recent[-50:].mean()
    weights = (short_mean - long_mean) / long_mean
    norm_weights = weights / weights.abs().sum()
    return norm_weights
```

The history method tells the algorithm that we want to consider the 
last 60 days of price.  And the short/long mean variables are the 
mean price of the last 10 and 50 days respectively.  We calculate the
normalized percent change between these values to determine how
large of a percent of our portfolio's value to invest in each asset
each day.

Now the handle data method just executes the trades that we told 
the algorithm to make in the ```make_weights``` method.  
```python
def handle_data(algo, data):
    weights = make_weights(algo, data)
    for security in algo.symbol_ids:
        order_target_percent(security, weights[security])
```

To run the algorithm we simply execute the ```run_algorithm``` method.

```python

start = pd.Timestamp('2016-1-1', tz='utc')
end = pd.Timestamp('2017-1-1', tz='utc')
results = run_algorithm(start=start, end=end, capital_base=1000, 
                        initialize=initialize, handle_data=handle_data)
```

Note that the ```initialize``` and ```handle_data``` methods are
required as it definines our algorithm's behavior.  
Any other function we create doesn't need to be included in 
```run_algorithm```.

To get usable results from this backtest we run the following pyfolio
function to unpack results.
```python
returns, positions, orders = pf.utils.extract_rets_pos_txn_from_zipline(results)
```

Two simple functions that help us visualize our model's performance
display the returns.  The plots are too big to be effectively
displayed on the screen so I usually just save them to a png.  
```python
def plot_returns():
    pf.plot_rolling_returns(returns)
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.savefig("returns")
    plt.show()
```