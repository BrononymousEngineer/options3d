# options3d
Visualize basic options contract parameters

## Project Structure

* option.py defines black scholes functions as well as an option class to hold the results
* chain.py defines functions that get data from yahoo and make calculations with the option object
* plot.py defines plotting functions that create graph objects
* app.py ties it all together in a dash webapp

so...

option -> chain -> plot -> app
---------------------------------------------------------------------------------------------------
The app is hosted at https://options3d.herokuapp.com/ on a free account

To start the webapp locally, clone the repo and run app.py

Note: this hasn't been extensively tested so there is a good chance bugs exist.
