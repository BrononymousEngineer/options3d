# options3d
Visualize basic options contract parameters

## Project Structure

option.py defines black scholes functions as well as an option class to hold the results
chain.py defines functions that get data from yahoo and make calculations with the option object
plot.py defines plotting functions that create graph objects
app.py ties it all together in a dash webapp

so...

option -> chain -> plot -> app