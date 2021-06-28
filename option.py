"""Calculates implied volatility/the greeks for a call or a put, according to the Black-Scholes-Merton model.

Contents
--------
JIT compiled functions (thanks numba) that compute BSM formulas, as well as an OptionContract class that acts as a
container for the results.

Sources
-------
Implementation:
    https://en.wikipedia.org/wiki/Black–Scholes_model
    https://en.wikipedia.org/wiki/Greeks_(finance)
    https://en.wikipedia.org/wiki/Bisection_method
    https://www.amazon.com/Options-as-Strategic-Investment-Fifth/dp/0735204659
    https://www.amazon.com/Options-Futures-Other-Derivatives-10th-dp-013447208X/dp/013447208X/ref=dp_ob_title_bk
Intuition:
    https://quant.stackexchange.com/questions/12901/does-implied-volatility-always-exist
    https://www.youtube.com/watch?v=spkim5Ns304
    https://www.youtube.com/watch?v=lD8nzTETqcs
    https://www.reddit.com/r/options/comments/hgg1c1/clear_explanation_of_d1_in_black_scholes/?sort=confidence
    https://financetrainingcourse.com/education/wp-content/uploads/2011/03/Understanding.pdf

Note
-----
Models are just models. The BSM model famously fails at describing reality, but it can still be useful for thinking
about the market in a different way.

In the end supply and demand are the only things that determine the market price of anything...just because a model says
something should happen doesn't mean it will.
"""

import datetime as dt
import json
from json import dumps
from numba import jit, double
from numpy import sqrt, log, power, exp, pi, inf  # , sin
# from scipy.stats import norm
# from scipy.integrate import quad


'''
General functions
'''


def year_delta(start: str, end: str) -> float:
    """
    Arguments
    ---------
    start : str
        Starting datetime (YYYY-MM-DD HH:MM:SS)
    end   : str
        Ending datetime (YYYY-MM-DD HH:MM:SS)

    Returns
    -------
    float : the time between start/end in years, at a resolution of seconds
    """
    tdelta = dt.datetime.strptime(
        end, '%Y-%m-%d %H:%M:%S') - dt.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    return (tdelta.days / 365) + (tdelta.seconds / (365*24*60*60))


@jit(nopython=True)
def phi(x: float) -> float:
    """Standard normal (0, 1) probability density function.

    Arguments
    ---------
    x : float
        z-score to plug into the PDF

    Returns
    -------
    float
        Probability density at x.
    """
    return exp(-0.5*power(x, 2)) / sqrt(2*pi)


@jit(nopython=True)
def N(x: float) -> double:
    """Standard normal (0, 1) cumulative distribution function.

    Arguments
    ---------
    x : float
        z-score to plug into the CDF

    Returns
    -------
    double
        Probability of some observation being x or less.

    Note
    ----
    Tried many different ways to cut down on calculation time. scipy.stats.norm.cdf is super accurate, but super slow
    because there is a lot of overhead every time it's called. Unfortunately this can't be JITted away, because at its
    core, scipy is calling a FORTRAN routine to make calculation.

    Algebraic approximations are lightning fast, but not accurate enough for extremely ITM or OTM options, causing the
    bisection algorithm to not converge on a price and hit the max_iter limit.

    Fortunately, the approximation from McMillan is very accurate. I want to find out where he got it.
    """
    x = double(x)
    # http://jjmie.hu.edu.jo/vol-13-4/16-19-01.pdf
    # absx = abs(x)
    # return {
    #     True: (1 / (1 + exp(-1.702*x))) - 0.0095*sin(2.5*x)*exp(-0.01*power(x, 2)),
    #     False: (1 / (1 + exp(-1.702*x))) + power(10, -3)*(
    #         0.2656*power(absx, 5) -
    #         5.1970*power(absx, 4) +
    #         39.810*power(absx, 3) -
    #         147.60*power(absx, 2) +
    #         258.64*absx - 163.3
    #     )
    # }[absx <= 2]
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    # return norm.cdf(x)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
    # return quad(PDF, -inf, x)[0]
    # https://www.researchgate.net/publication/337326463_An_accurate_approximation_for_the_standard_normal_distribution_function
    # y = power(
    #     0.8601423*power(x, 2 / 3) +
    #     0.0007497*power(x, 5 /3) -
    #     0.0056102*power(x, 8 / 3) +
    #     0.0008232*power(x, 11 / 3),
    #     3
    # )
    # return 0.5*(1 + sqrt(1 - exp(-y)))
    # https://www.amazon.com/Options-as-Strategic-Investment-Fifth/dp/0735204659
    absx = abs(x)
    y = 1 / (1 + 0.2316419*absx)
    z = 0.3989423*exp(-0.5*power(x, 2))
    xx = 1 - z*(
        1.330274*power(y, 5) -
        1.821256*power(y, 4) +
        1.781478*power(y, 3) -
        0.356538*power(y, 2) +
        0.3193815*y
    )
    return xx if x > 0 else 1 - xx


'''
Pricing functions
'''


@jit(nopython=True)
def d1_func(S: float, K: float, t: float, sigma: float, r: float, q: float) -> float:
    """Z-score of current log return.

    Arguments
    ---------
    S     : float
        Spot price of the underlying
    K     : float
        Strike price
    t     : float
        Time until expiration, in units of years
    sigma : float
        Annualized volatility of log returns
    r     : float
        Annualized risk free rate of return
    q     : float
        Annualized dividend yield of the underlying

    Returns
    -------
    d1 : float

    Note
    ----
    Remember a Z-score is a deviation from the mean, normalized by the standard deviation.

        (x - mu) / sigma

        In the Black-Scholes world:
            x is the current log return we could get by exercising right now:
                ln(S / K)
            mu is the return we can expect on average:
                ((r - q) + 0.5*sigma^2)*t
                 ^ risk free rate
                ((r - q) + 0.5*sigma^2)*t
                     ^ less the amount the stock drops from the dividend
                ((r - q) + 0.5*sigma^2)*t
                               ^ plus some amount due to the normal fluctuation of returns
                ((r - q) + 0.5*sigma^2)*t
                                        ^ scaled by the time left until expiration
            sigma is the annualized volatility scaled by the time left until expiration:
                sigma*sqrt(t)
                      ^ we do a sqrt because sigma is a stdev which is the sqrt of the variance

    Think of d1 as some quantity of standard deviations of log returns. When we plug this into N(), we are calculating
    the probability of this number of standard deviations occurring.
    """
    return (log(S / K) + (r - q + 0.5*power(sigma, 2))*t) / (sigma*sqrt(t))


@jit(nopython=True)
def d2_func(S: float, K: float, t: float, sigma: float, r: float, q: float) -> float:
    """Z-score of current log return.

    Arguments
    ---------
    S     : float
        Spot price of the underlying
    K     : float
        Strike price
    t     : float
        Time until expiration, in units of years
    sigma : float
        Annualized volatility of log returns
    r     : float
        Annualized risk free rate of return
    q     : float
        Annualized dividend yield of the underlying

    Returns
    -------
    d2 : float

    Note
    ----
    Remember a Z-score is a deviation from the mean, normalized by the standard deviation.

        (x - mu) / sigma

        In the Black-Scholes world:
            x is the current log return we could get by exercising right now:
                ln(S / K)
            mu is the return we can expect on average:
                ((r - q) - 0.5*sigma^2)*t
                 ^ risk free rate
                ((r - q) - 0.5*sigma^2)*t
                     ^ less the amount the stock drops from the dividend
                ((r - q) - 0.5*sigma^2)*t
                               ^ plus some amount due to the normal fluctuation of returns
                ((r - q) - 0.5*sigma^2)*t
                                        ^ scaled by the time left until expiration
            sigma is the annualized volatility scaled by the time left until expiration:
                sigma*sqrt(t)
                      ^ we do a sqrt because sigma is a stdev which is the sqrt of the variance

    Think of d2 as some quantity of standard deviations of log returns. When we plug this into N(), we are calculating
    the probability of this number of standard deviations occurring.
    """
    return (log(S / K) + (r - q - 0.5*power(sigma, 2))*t) / (sigma*sqrt(t))


@jit(nopython=True)
def BSMprice(opt_type: int, S: float, K: float, t: float, sigma: float, r: float, q: float) -> float:
    """Black-Scholes-Merton price of an option.

    Arguments
    ---------
    opt_type : int
        1 for a call, 0 for a put
    S        : float
        Spot price of the underlying
    K        : float
        Strike price
    t        : float
        Time until expiration, in units of years
    sigma    : float
        Annualized volatility of log returns
    r        : float
        Annualized risk free rate of return
    q        : float
        Annualized dividend yield of the underlying

    Returns
    -------
    float
        The price of the option with the given parameters, as defined by the Black-Scholes-Merton model.

    Note
    -----
    For a call option, here is the rough meaning for each term:
    S*exp(-q*t)*N(d1) - K*exp(-r*t)*N(d2)
    S*exp(-q*t)                             => present value of the stock price, accounting for the dividend
               *                               aka stock price discounted by the amount it will drop due to the div
                N(d1)                       => probability that takes into account 'underwater outcomes' will be 0
                       -                       that is, the payoff is zero if the stock is not above the strike
                        K*exp(-r*t)         => present value of the strike price, accounting for the risk free rate
                                   *           aka strike discounted by the amount that cash to buy will grow at the rfr
                                    N(d2)   => probability of the option being exercised
                                               that is, the probability of the stock price being above the strike
    """
    d1 = d1_func(S, K, t, sigma, r, q)
    d2 = d2_func(S, K, t, sigma, r, q)
    if opt_type == 1:
        return S*exp(-q*t)*N(d1) - K*exp(-r*t)*N(d2)
    elif opt_type == 0:
        return K*exp(-r*t)*N(-d2) - S*exp(-q*t)*N(-d1)


'''
Greeks functions
'''


@jit(nopython=True)
def BSMIV(opt_type: int, V: float, S: float, K: float, t: float, r: float, q: float) -> float:
    """Calculation of Black-Scholes-Merton implied volatility using the bisection method.

    Arguments
    ---------
    opt_type : int
        1 for a call, 0 for a put
    V        : float
        Spot price of the option
    S        : float
        Spot price of the underlying
    K        : float
        Strike price
    t        : float
        Time until expiration, in units of years
    r        : float
        Annualized risk free rate of return
    q        : float
        Annualized dividend yield of the underlying

    Returns
    -------
    float
        The volatility implied by the price of the option with the given parameters, as defined by the
        Black-Scholes-Merton model.

    Note
    ----
    Implied volatility has nothing to do with the historical volatility of the underlying asset. It is the volatility
    that is expected from now until the option's expiration. More literally (and correctly), it is the volatility that
    makes the current option price 'fair' according to BSM.
    """
    intval = 0.0
    if opt_type == 1:               # Initial check that the OptionContract class already does, but since we are now
        # directly using this function outside the class, we need to perform these checks:
        intval = max([0.0, S - K])
    elif opt_type == 0:             # - the option price cannot be less than intrinsic value, else an arb exists
        # - the option price also cannot be greater than or equal to the stock price,
        intval = max([0.0, K - S])
    if V < intval:  # because that makes no sense
        return 0.0
    if V >= S:
        return inf

    # Flag for failing to find an implied volatility
    failed = False
    # Don't search for a starting lo guess more than 20 times
    max_iter_lo = 20
    # Don't search for a starting hi guess more than 20 times
    max_iter_hi = 20
    # Don't do the bisection loop more than 20 times
    max_iter = 20
    # Initial guess at an IV which ideally gives a price below V
    sigma_lo = 0.1
    # Initial guess at an IV which ideally gives a price above V
    sigma_hi = 0.5
    sigma_mid = 0.5*(sigma_lo + sigma_hi)   # Midpoint of the initial guesses

    counter_lo = 0
    # Calculate the price given by the low IV guess
    V_lo = BSMprice(opt_type, S, K, t, sigma_lo, r, q)
    # If the low price is at or above V, decrease the initial
    while (V_lo >= V) and (not failed):
        # low guess at IV until V_lo is less than V
        sigma_lo /= 1.1
        V_lo = BSMprice(opt_type, S, K, t, sigma_lo, r, q)
        counter_lo += 1
        if counter_lo == max_iter_lo:
            print('COULD NOT FIND INITIAL LO IV')
            failed = True

    counter_hi = 0
    # Calculate the price given by the high IV guess
    V_hi = BSMprice(opt_type, S, K, t, sigma_hi, r, q)
    # If the low price is at or below V, increase the initial
    while (V_hi <= V) and (not failed):
        # high guess at IV until V_hi is greater than V
        sigma_hi *= 1.1
        V_hi = BSMprice(opt_type, S, K, t, sigma_hi, r, q)
        counter_hi += 1
        if counter_hi == max_iter_hi:
            print('COULD NOT FIND INITIAL HI IV')
            failed = True

    # Calculate the price given by the midpoint IV guess
    V_mid = BSMprice(opt_type, S, K, t, sigma_mid, r, q)

    counter = 0
    # Iterate as long as sigma_mid price differs by $0.01
    while (round(V_mid, 2) != round(V, 2)) and (not failed):
        if V_mid < V:                       # If the mid price < V, make the mid IV the new low guess
            sigma_lo = sigma_mid
        if V_mid > V:                       # If the mid price > V, make the mid IV the new high guess
            sigma_hi = sigma_mid
        # Recalculate the middle IV guess
        sigma_mid = 0.5 * (sigma_lo + sigma_hi)
        # Recalculate the price given by the middle IV guess
        V_mid = BSMprice(opt_type, S, K, t, sigma_mid, r, q)
        counter += 1
        if counter >= max_iter:             # Return the middle IV guess if we reach max_iter, and print a message
            print('IV MAX ITER')
            failed = True

    if failed:
        return -1.0

    return sigma_mid


@jit(nopython=True)
def BSMdelta(opt_type: int, t: float, q: float, d1: float):
    """First derivative of V with respect to S (dV/dS).

    Arguments
    ---------
    opt_type : int
        1 for a call, 0 for a put
    t        : float
        Time until expiration, in units of years
    q        : float
        Annualized dividend yield of the underlying
    d1       : float
        z-score of the log return given by exercising the option right now

    Returns
    -------
    float
        delta

    Note
    ----
    There are two main interpretations of delta:
        1) As mentioned above, delta is the first derivative of the option price with respect to the spot price of the
           underlying. This can roughly be translated into the number of dollars the option price will increase for a $1
           increase in the underlying asset.

        2) It also can be interpreted as a rough probability that the option will expire $0.01 in the money. Note that
           this is NOT a probability of the underlying being above the strike at any point before expiration. THAT
           probability is roughly 2 times delta.

    The first interpretation is the most mathematically correct, and is the basis of hedging a portfolio against
    price changes of the underlying asset. For example:

        The delta of the underlying is always 1. Let's say you own 50 shares of some stock. If you buy a 0.5 delta put,
        you are (in theory) momentarily insulated from the stock dropping (and rising).

            1) If the stock price decreases by $1, you lose $50 from that, but gain $50 from the put increasing in value
                (0.5 delta x 100 share contract size = $50)

            2) Conversely, a $1 increase in the stock price will gain you $50, but you lose you $50 from the put
    """
    if opt_type == 1:
        return exp(-q*t)*N(d1)
    elif opt_type == 0:
        return -exp(-q*t)*N(-d1)


@jit(nopython=True)
def BSMvega(S: float, K: float, t: float, r: float, q: float, d1: float, d2: float):
    """First derivative of option price with respect to sigma (dV/dsigma).

    Arguments
    ---------
    S        : float
        Spot price of the underlying
    K        : float
        Strike price
    t        : float
        Time until expiration, in units of years
    r        : float
        Annualized risk free rate of return
    q        : float
        Annualized dividend yield of the underlying
    d1       : float
        z-score of the log return given by exercising the option right now
    d2       : float
        z-score of the log return given by exercising the option right now

    Returns
    -------
    float
        vega

    Note
    ----
    Vega is the sensitivity of the option price with respect to implied volatility. This can seem like circular logic,
    because as shown in the BSMIV function, you have to know the option price to know volatility, and volatility has
    nothing to do directly with the historical volatility of the underlying.

    Also note that both calls and puts will increase in price if volatility increases. This means that if expected
    volatility rises enough, the direction of the underlying asset price might not matter. For example:

        - You have one call option that has a delta of 0.5 and a vega of 0.6. The underlying drops by $1, but implied
          volatility rises by 1%. The option price loses $0.50 of value from the underlying price decrease, but gains
          $0.60 from the increasing expected volatility. The net price change of your call is +$0.10, even though the
          stock price decreased.

    The example above would have the same outcome if the call was instead a put, and the underlying price increased. It
    should also be noted that this means an option price can change even if the underlying price does not move (this
    should hopefully be obvious just from the simple fact that there are multiple parameters which determine the price
    of an option).
    """
    # Use both definitions
    vega1 = S*exp(-q*t)*phi(d1)*sqrt(t)
    vega2 = K*exp(-r*t)*phi(d2)*sqrt(t)
    return 0.5*(vega1 + vega2) / 100


@jit(nopython=True)
def BSMtheta(opt: int, S: float, K: float, t: float, sigma: float, r: float, q: float, d1: float, d2: float) -> float:
    """First derivative of option price with respect to time (dV/dt).

    Arguments
    ---------
    opt   : int
        1 for call, 0 for put
    S     : float
        Spot price of the underlying
    K     : float
        Strike price
    t     : float
        Time until expiration, in units of years
    sigma : float
        Annualized volatility of log returns
    r     : float
        Annualized risk free rate of return
    q     : float
        Annualized dividend yield of the underlying
    d1    : float
        z-score of the log return given by exercising the option right now
    d2    : float
        z-score of the log return given by exercising the option right now

    Returns
    -------
    float
        theta

    Note
    ----
    Theta is expressed in terms of value per year. To get value per calendar day, divide by 365, or by 252 for value per
    trading day.

    Theta is simply the amount that an option will (lose) due to the passage of time. It makes sense if you think about
    less time being less chance for an option to move (further) into the money.
    """
    if opt == 1:
        return (-exp(-q*t)*((S*phi(d1)*sigma) / (2*sqrt(t))) - r*K*exp(-r*t)*N(d2) + q*S*exp(-q*t)*N(d1)) / 365
    elif opt == 0:
        return (-exp(-q*t)*((S*phi(d1)*sigma) / (2*sqrt(t))) + r*K*exp(-r*t)*N(-d2) - q*S*exp(-q*t)*N(-d1)) / 365


@jit(nopython=True)
def BSMrho(opt_type: int, K: float, r: float, t: float, d2: float) -> float:
    """First derivative of option price with respect to the risk free rate (dV/dr).

    Arguments
    ---------
    opt_type : int
        1 for call, 0 for put
    K        : float
        Strike price
    t        : float
        Time until expiration, in units of years
    r        : float
        Annualized risk free rate of return
    d2       : float
        z-score of the log return given by exercising the option right now

    Returns
    -------
    float
        rho

    Note
    ----
    Rho gives the amount the option's value will change by given a 1% increase in the risk free interest rate.
    """
    if opt_type == 1:
        return K*t*exp(-r*t)*N(d2) / 100
    elif opt_type == 0:
        return -K*t*exp(-r*t)*N(-d2) / 100


@jit(nopython=True)
def BSMepsilon(opt_type: int, S: float, t: float, q: float, d1: float) -> float:
    """First derivative of option price with respect to the dividend yield (dV/dq).

    Arguments
    ---------
    opt_type : int
        1 for call, 0 for put
    S        : float
        Spot price of the underlying
    t        : float
        Time until expiration, in units of years
    q        : float
        Annualized dividend yield of the underlying
    d1       : float
        z-score of the log return given by exercising the option right now

    Returns
    -------
    float
        epsilon

    Note
    ----
    As defined here, this is the amount that the option's value will change given a 1% increase in the dividend yield.
    """
    if opt_type == 1:
        return -S*t*exp(-q*t)*N(d1) / 100
    elif opt_type == 0:
        return S*t*exp(-q*t)*N(-d1) / 100


@jit(nopython=True)
def BSMlambda(delta: float, S: float, V: float) -> float:
    """Not really a greek, but rather an expression of leverage.

    Arguments
    ---------
    delta : float
        BSM delta of the option
    V     : float
        Spot price of the option
    S     : float
        Spot price of the underlying

    Returns
    -------
    float
        lambda

    Note
    ----
    Percentage change in the option price per percentage change in the underlying asset's price.
    """
    return delta*(S / V)


@jit(nopython=True)
def BSMgamma(S: float, K: float, t: float, sigma: float, r: float, q: float, d1: float, d2: float) -> float:
    """First derivative of delta with respect to the price underlying (ddelta/dS). Also d2V/dS2.

    Arguments
    ---------
    S     : float
        Spot price of the underlying
    K     : float
        Strike price
    t     : float
        Time until expiration, in units of years
    sigma : float
        Annualized volatility of log returns
    r     : float
        Annualized risk free rate of return
    q     : float
        Annualized dividend yield of the underlying
    d1    : float
        z-score of the log return given by exercising the option right now
    d2    : float
        z-score of the log return given by exercising the option right now

    Returns
    -------
    float
        gamma

    Note
    ----
    Best thought of as: for every dollar the stock moves, delta changes by the amount given by gamma. Gamma is the
    highest at the money, and decreases for strikes farther in or out of the money. This effect becomes more extreme
    as time to expiration decreases.
    """
    # Use both definitions
    gamma1 = (exp(-q*t)*phi(d1)) / (S*sigma*sqrt(t))
    gamma2 = (K*exp(-r*t)*phi(d2)) / (power(S, 2)*sigma*sqrt(t))
    return 0.5*(gamma1 + gamma2)


@jit(nopython=True)
def BSMvanna(S: float, t: float, sigma: float, q: float, d1: float, d2: float, vega: float) -> float:
    """First derivative of delta with respect to volatility (ddelta/dsigma). Also dvega/dS.

    Arguments
    ---------
    S     : float
        Spot price of the underlying
    t     : float
        Time until expiration, in units of years
    sigma : float
        Annualized volatility of log returns
    q     : float
        Annualized dividend yield of the underlying
    d1    : float
        z-score of the log return given by exercising the option right now
    d2    : float
        z-score of the log return given by exercising the option right now
    vega  : float
        First derivative of option price with respect to volatility

    Returns
    -------
    float
        vanna

    Note
    ----
    Can be useful to know for a delta or vega hedged portfolio. Helps the trader determine the usefulness of a delta or
    vega hedge.
    """
    # Use both definitions
    vanna1 = -exp(-q*t)*phi(d1)*(d2 / sigma)
    vanna2 = (vega / S)*(1 - (d1 / (sigma*sqrt(t))))
    return 0.5*(vanna1 + vanna2)


@jit(nopython=True)
def BSMcharm(opt_type: int, t: float, sigma: float, r: float, q: float, d1: float, d2: float) -> float:
    """First derivative of delta with respect to time (ddelta/dt). Also dtheta/dS.

    Arguments
    ---------
    opt_type : int
        1 for call, 0 for put
    t        : float
        Time until expiration, in units of years
    sigma    : float
        Annualized volatility of log returns
    r        : float
        Annualized risk free rate of return
    q        : float
        Annualized dividend yield of the underlying
    d1       : float
        z-score of the log return given by exercising the option right now
    d2       : float
        z-score of the log return given by exercising the option right now

    Returns
    -------
    float
        charm

    Note
    ----
    Expressed in delta per year. For this reason we divide by 365 to get delta per day.

    Mostly accurate when there is a long time until expiration. When time to expiration gets small, charm can change
    often enough that full-day estimates may not be very accurate. Can be useful when delta-hedging a position over a
    weekend.
    """
    if opt_type == 1:
        return -q*exp(-q*t)*N(d1) - exp(-q*t)*phi(d1)*((2*t*(r - q) - d2*sigma*sqrt(t)) / (2*t*sigma*sqrt(t))) / 365
    if opt_type == 0:
        return -q*exp(-q*t)*N(-d1) - exp(-q*t)*phi(d1)*((2*t*(r - q) - d2*sigma*sqrt(t)) / (2*t*sigma*sqrt(t))) / 365


@jit(nopython=True)
def BSMvomma(S: float, t: float, sigma: float, q: float, d1: float, d2: float, vega: float) -> float:
    """First derivative of vega with respect to volatility (dvega/dsigma). Also d2V/dsigma2.

    Arguments
    ---------
    S     : float
        Spot price of the underlying
    t     : float
        Time until expiration, in units of years
    sigma : float
        Annualized volatility of log returns
    q     : float
        Annualized dividend yield of the underlying
    d1    : float
        z-score of the log return given by exercising the option right now
    d2    : float
        z-score of the log return given by exercising the option right now
    vega  : float
        First derivative of option price with respect to volatility

    Returns
    -------
    float
        vomma

    Note
    ----
    Measures second order sensitivity of the option price with respect to volatility, or the rate of change of vega as
    volatility changes.

    See https://en.wikipedia.org/wiki/Greeks_(finance)#Vomma for more info.
    """
    # Use both definitions
    vomma1 = S*exp(-q*t)*phi(d1)*sqrt(t)*((d1*d2) / sigma)
    vomma2 = vega*((d1*d2) / sigma)
    return 0.5*(vomma1 + vomma2)


@jit(nopython=True)
def BSMveta(S: float, t: float, sigma: float, r: float, q: float, d1: float, d2: float) -> float:
    """First derivative of vega with respect to time (dvega/dt).

    Arguments
    ---------
    S     : float
        Spot price of the underlying
    t     : float
        Time until expiration, in units of years
    sigma : float
        Annualized volatility of log returns
    r     : float
        Annualized risk free rate of return
    q     : float
        Annualized dividend yield of the underlying
    d1    : float
        z-score of the log return given by exercising the option right now
    d2    : float
        z-score of the log return given by exercising the option right now

    Returns
    -------
    float
        veta

    Note
    ----
    To get the percentage change in vega per day, divide by 100 times the number of days per calendar year.
    """
    return (-S*exp(-q*t)*phi(d1)*sqrt(t)*(q + ((d1*(r - q)) / (sigma*sqrt(t))) - ((1 + d1*d2) / (2*t)))) / (100*365)


@jit(nopython=True)
def BSMphi(S: float, K: float, t: float, sigma: float, r: float, q: float) -> float:
    """Second derivative of option price with respect to strike (d2V/dK2).

    Arguments
    ---------
    S     : float
        Spot price of the underlying
    K     : float
        Strike price
    t     : float
        Time until expiration, in units of years
    sigma : float
        Annualized volatility of log returns
    r     : float
        Annualized risk free rate of return
    q     : float
        Annualized dividend yield of the underlying

    Returns
    -------
    float
        phi

    Note
    ----
    From Wikipedia: This partial derivative has a fundamental role in the Breeden-Litzenberger formula, which uses
    quoted call option prices to estimate the risk-neutral probabilities implied by such prices.
    """
    return exp(-r*t)*(1 / K)*(1 / sqrt(2*pi*t*power(sigma, 2)))*exp(
        -(1 / (2*t*power(sigma, 2)))*power(
            log(K / S) - t*((r - q) - 0.5*power(sigma, 2)), 2
        )
    )


@jit(nopython=True)
def BSMspeed(S: float, t: float, sigma: float, q: float, d1: float, gamma: float) -> float:
    """First derivative of gamma with respect to option price (dgamma/dS). Also d3V/dS3.

    Arguments
    ---------
    S     : float
        Spot price of the underlying
    t     : float
        Time until expiration, in units of years
    sigma : float
        Annualized volatility of log returns
    q     : float
        Annualized dividend yield of the underlying
    d1    : float
        z-score of the log return given by exercising the option right now
    gamma : float
        Gamma of an option

    Returns
    -------
    float
        gamma

    Note
    ----
    Sometimes referred to as the gamma of gamma. Like other higher order greeks, can be useful when monitoring a delta-
    hedged portfolio.
    """
    # Use both definitions
    speed1 = -exp(-q*t) * (phi(d1) / (power(S, 2) * sigma *
                                      sqrt(t))) * ((d1 / (sigma * sqrt(t))) + 1)
    speed2 = -(gamma / S)*((d1 / (sigma*sqrt(t))) + 1)
    return 0.5*(speed1 + speed2)


@jit(nopython=True)
def BSMzomma(S: float, t: float, sigma: float, q: float, d1: float, d2: float, gamma: float) -> float:
    """First derivative of gamma with respect to volatility (dgamma/dsigma). Also dvanna/dS.

    Arguments
    ---------
    S     : float
        Spot price of the underlying
    t     : float
        Time until expiration, in units of years
    sigma : float
        Annualized volatility of log returns
    q     : float
        Annualized dividend yield of the underlying
    d1    : float
        z-score of the log return given by exercising the option right now
    d2    : float
        z-score of the log return given by exercising the option right now
    gamma : float
        Gamma of an option

    Returns
    -------
    float
        zomma

    Note
    ----
    Can be useful when monitoring a gamma-hedged portfolio.
    """
    # Use both definitions
    zomma1 = exp(-q*t)*((phi(d1) * (d1 * d2 - 1)) /
                        (S * power(sigma, 2) * sqrt(t)))
    zomma2 = gamma*((d1*d2 - 1) / sigma)
    return 0.5*(zomma1 + zomma2)


@jit(nopython=True)
def BSMcolor(S: float, t: float, sigma: float, r: float, q: float, d1: float, d2: float) -> float:
    """First derivative of gamma with respect to volatility (dgamma/dsigma). Also dvanna/dS.

    Arguments
    ---------
    S     : float
        Spot price of the underlying
    t     : float
        Time until expiration, in units of years
    sigma : float
        Annualized volatility of log returns
    r     : float
        Annualized risk free rate of return
    q     : float
        Annualized dividend yield of the underlying
    d1    : float
        z-score of the log return given by exercising the option right now
    d2    : float
        z-score of the log return given by exercising the option right now

    Returns
    -------
    float
        color

    Note
    ----
    Expressed in gamma per year. Divide by 365 to get gamma per day. Can be useful when monitoring a gamma-hedged
    portfolio.
    """
    return (-exp(-q*t) * (phi(d1) / (2*S*t*sigma*sqrt(t))))*(
        2*q*t + 1 + d1*((2*t*(r - q) - d2*sigma*sqrt(t)) / (sigma*sqrt(t)))
    ) / 365


@jit(nopython=True)
def BSMultima(sigma: float, d1: float, d2: float, vega: float) -> float:
    """First derivative of vomma with respect to volatility. (dvomma/dsigma). Also d3V/dS3.

    Arguments
    ---------
    sigma : float
        Annualized volatility of log returns
    d1    : float
        z-score of the log return given by exercising the option right now
    d2    : float
        z-score of the log return given by exercising the option right now
    vega  : float
        First derivative of option price with respect to volatility

    Returns
    -------
    float
        ultima

    Note
    ----
    Third-order sensitivity of the option price with respect to volatility.
    """
    return (-vega / power(sigma, 2))*(d1*d2*(1 - d1*d2) + power(d1, 2) + power(d2, 2))


@jit(nopython=True)
def BSMdualdelta(opt_type: float, t: float, r: float, d2: float) -> float:
    """First derivative of option price with respect to strike (dV/dK).

    Arguments
    ---------
    opt_type : int
        1 for call, 0 for put
    t        : float
        Time until expiration, in units of years
    r        : float
        Annualized risk free rate of return
    d2       : float
        z-score of the log return given by exercising the option right now

    Note
    ----
    The actual probability of an option finishing in the money is its dual delta.
    """
    if opt_type == 1:
        return exp(-r*t) * N(d2)
    elif opt_type == 0:
        return -exp(-r*t) * N(-d2)


@jit(nopython=True)
def BSMdualgamma(K: float, t: float, sigma: float, r: float, d2: float) -> float:
    """First derivative of delta with respect to strike (ddelta/dK).

    Arguments
    ---------
    K     : float
        Strike price
    t        : float
        Time until expiration, in units of years
    sigma : float
        Annualized volatility of log returns
    r        : float
        Annualized risk free rate of return
    d2       : float
        z-score of the log return given by exercising the option right now

    Note
    ----
    The actual gamma?
    """
    return exp(-r*t)*(phi(d2) / (K * sigma * sqrt(t)))


'''
OptionContract class
'''


class OptionContract:
    """Black-Scholes-Merton option contract.

    Required Arguments
    ------------------
    opt_type : str
        'call' or 'put'
    exp_date : str
        Expiration date of the option. 'YYYY-MM-DD'
    V       : float
        Spot price of the option
    S       : float
        Spot price of the underlying
    K       : float
        Strike price

    Optional Arguments
    ------------------
    r                : float
        Annualized risk free rate of return. Default is 0.
    q                : float
        Annualized dividend yield of the underlying. Default is 0.
    exp_time         : str
        Time of day (24h time) that the option expires. Default is 16:00:00.
    current_datetime : str
        Datetime  to use as the current date/time (like 'YYYY-MM-DD HH:MM:SS'). Default is the current date and time.
    IV               : float
        If implied volatility has already been obtained, use it to calculate the greeks. Default is None.
    price_type       : str
        See which price made the IV calc fail, if it did.

    Attributes
    ----------
    current_datetime : str
    expiration         : str
        Datetime like 'YYYY-MM-DD HH:MM:SS'
    opt_type           : float
    V                  : float
    S                  : float
    K                  : float
    r                  : float
    q                  : float
    t                  : float
        Time until expiration in years. Second resolution.
    IV                 : float
    delta              : float
    vega:              : float
    theta              : float
    rho                : float
    epsilon            : float
    Lambda             : float
    gamma              : float
    vanna              : float
    charm              : float
    vomma              : float
    veta               : float
    phi                : float
    speed              : float
    zomma              : float
    color              : float
    ultima             : float
    dualdelta          : float
    dualgamma          : float

    Note
    ----
    Initial versions of this class contained all the calculation functions, but it's just too slow for a whole chain.
    By extracting them out into separate functions, we can use numba to JIT compile them and make them a lot faster.
    """

    def __init__(
            self, opt_type,
            exp_date: str,
            V: float,
            S: float,
            K: float,
            r: float = 0,
            q: float = 0,
            exp_time: str = '16:00:00',
            current_datetime: str = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S'),
            IV: float = None,
            price_type: str = None
    ):
        #
        # Assign inputs
        #
        self.current_datetime = current_datetime
        self.__exp_date = exp_date
        self.__exp_time = exp_time
        self.expiration = f'{exp_date} {exp_time}'
        self.opt_type = opt_type
        self.itm = {'call': S > K, 'put': K >= S}[self.opt_type]
        self.V = max([float(V), 0.01])
        self.S = float(S)
        self.K = float(K)
        self.r = float(r)
        self.q = float(q)
        self.t = float(year_delta(self.current_datetime, self.expiration))
        #
        # Initialize the greeks
        #
        self.IV = 0.0
        self.delta = 0.0
        self.vega = 0.0
        self.theta = 0.0
        self.rho = 0.0
        self.epsilon = 0.0
        self.Lambda = 0.0
        self.gamma = 0.0
        self.vanna = 0.0
        self.charm = 0.0
        self.vomma = 0.0
        self.veta = 0.0
        self.phi = 0.0
        self.speed = 0.0
        self.zomma = 0.0
        self.color = 0.0
        self.ultima = 0.0
        self.dualdelta = 0.0
        self.dualgamma = 0.0
        #
        # IV calc
        #
        opt_type_01 = {'call': 1, 'put': 0}[self.opt_type]
        # If there is no supplied IV, calculate it (if not conditions 1&2)
        if IV is None:
            if self.V <= {'call': max([0.0, S - K]), 'put': max([0.0, K - S])}[self.opt_type]:
                # Condition 1) IV = 0 if V less than intrinsic value
                self.IV = 0
            elif self.V > self.S:
                # Condition 2) IV = inf if V greater than the stock price
                self.IV = inf
            else:
                self.IV = BSMIV(opt_type_01, self.V, self.S,
                                self.K, self.t, self.r, self.q)
                if self.IV == -1.0:
                    self.IV = 0
                    price_type_str = f' ({price_type})' if price_type is not None else ''
                    print(
                        json.dumps(
                            {
                                'exp': self.expiration,
                                'opt_type': self.opt_type,
                                'itm': self.itm,
                                'V': '{:.2f}'.format(self.V) + price_type_str,
                                'S': '{:.2f}'.format(self.S),
                                'K': '{:.2f}'.format(self.K),
                                'r': self.r,
                                'q': self.q,
                                't': self.t
                            }, indent=4
                        )
                    )
        else:
            self.IV = IV                            # If there is a supplied IV, use it
        if self.IV != 0 and self.IV != inf:         # If we have a valid IV, calculate the greeks
            d1 = d1_func(self.S, self.K, self.t, self.IV, self.r, self.q)
            d2 = d2_func(self.S, self.K, self.t, self.IV, self.r, self.q)
            #
            # First order greeks
            #
            self.delta = BSMdelta(opt_type_01, self.t, self.q, d1)
            self.vega = BSMvega(self.S, self.K, self.t, self.r, self.q, d1, d2)
            self.theta = BSMtheta(
                opt_type_01, self.S, self.K, self.t, self.IV, self.r, self.q, d1, d2)
            self.rho = BSMrho(opt_type_01, self.K, self.r, self.t, d2)
            self.epsilon = BSMepsilon(opt_type_01, self.S, self.t, self.q, d1)
            self.Lambda = BSMlambda(self.delta, self.S, self.V)
            #
            # Second order greeks
            #
            self.gamma = BSMgamma(self.S, self.K, self.t,
                                  self.IV, self.r, self.q, d1, d2)
            self.vanna = BSMvanna(self.S, self.t, self.IV,
                                  self.q, d1, d2, self.vega)
            self.charm = BSMcharm(opt_type_01, self.t,
                                  self.IV, self.r, self.q, d1, d2)
            self.vomma = BSMvomma(self.S, self.t, self.IV,
                                  self.q, d1, d2, self.vega)
            self.veta = BSMveta(self.S, self.t, self.IV,
                                self.r, self.q, d1, d2)
            self.phi = BSMphi(self.S, self.K, self.t, self.IV, self.r, self.q)
            #
            # Third order greeks
            #
            self.speed = BSMspeed(self.S, self.t, self.IV,
                                  self.q, d1, self.gamma)
            self.zomma = BSMzomma(self.S, self.t, self.IV,
                                  self.q, d1, d2, self.gamma)
            self.color = BSMcolor(self.S, self.t, self.IV,
                                  self.r, self.q, d1, d2)
            self.color = BSMcolor(self.S, self.t, self.IV,
                                  self.r, self.q, d1, d2)
            self.ultima = BSMultima(self.IV, d1, d2, self.vega)
            #
            # Misc greeks
            #
            self.dualdelta = BSMdualdelta(opt_type_01, self.t, self.r, d2)
            self.dualgamma = BSMdualgamma(self.K, self.t, self.IV, self.r, d2)

    def __repr__(self):
        params = [
            f'"{self.opt_type}"',
            f'"{self.__exp_date}"',
            f'{self.V}', f'{self.S}', f'{self.K}',
            f'r={self.r}', f'q={self.q}', f'exp_time={self.__exp_time}',
            f'current_datetime={self.current_datetime}'
        ]
        return f'{self.__class__.__name__}({", ".join(params)})'

    def __str__(self):
        return dumps(self.__dict__, indent=4)
