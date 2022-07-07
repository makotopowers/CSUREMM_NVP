from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def AR_model(data, p):
    """Autogregressive Model

    This method is suitable for univariate time series without trend and without seasonal components.

    Parameters
    ----------
    data : _type_
        _description_
    p : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return ARIMA(data, order=(p, 0, 0))


def MA_model(data, q):
    """Moving Average

    This method is suitable for univariate time series without trend and without seasonal components.

    Parameters
    ----------
    data : _type_
        _description_
    q : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    "Moving average model"
    return ARIMA(data, order=(0, 0, q))


def ARMA_model(data, p, q):
    """Autoregressive Moving Average

    This method is suitable for univariate time series without trend and without seasonal components.

    Parameters
    ----------
    data : _type_
        _description_
    p : _type_
        _description_
    q : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return ARIMA(data, order=(p, 0, q))


def ARIMA_model(data, p, d, q):
    """Autoregressive Integrated Moving Average

    This method is suitable for univariate time series with trend and without seasonal components.

    Parameters
    ----------
    data : _type_
        _description_
    p : _type_
        _description_
    d : _type_
        _description_
    q : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return ARIMA(data, order=(p, d, q))


def SARIMA_model(data, p, d, q, P, D, Q, m):
    """Seasonal Autoregressive Integrated Moving Average

    This method is suitable for univariate time series with trend and/or seasonal components.

    Parameters
    ----------
    data : _type_
        _description_
    p : _type_
        _description_
    d : _type_
        _description_
    q : _type_
        _description_
    P : _type_
        _description_
    D : _type_
        _description_
    Q : _type_
        _description_
    m : int
        The number of time steps for a single seasonal period. Ex: m=12 when you have
        year long season and your measurements are made each month.
    """
    return SARIMAX(data, order=(d, p, q), seasonal_order=(P, D, Q, m))


def SARIMAX_model(data, data_exog, p, d, q, P, D, Q, m):
    """ Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors.

    The method is suitable for univariate time series with trend and/or seasonal components and exogenous variables.

    Exogenous variables are also called covariates and can be thought of as parallel input sequences 
    that have observations at the same time steps as the original series. 

    Parameters
    ----------
    data : _type_
        _description_
    data_exog : _type_
        _description_
    p : _type_
        _description_
    d : _type_
        _description_
    q : _type_
        _description_
    P : _type_
        _description_
    D : _type_
        _description_
    Q : _type_
        _description_
    m : _type_
        _description_
    """
    return SARIMAX(data, exog=data_exog, order=(d, p, q), seasonal_order=(P, D, Q, m))


def VAR_model(data):
    """Vector Autoregression

    VAR is a multivariate version of the AR method.
    This method is suitable for multivariate time series without trend and without seasonal components.

    Parameters
    ----------
    data : _type_
        _description_
    """
    return VAR(data)


def VARMA_model(data, p, q):
    """Vector Autoregression Moving-Average

    VARMA is a multivariate version of the ARMA method.
    This method is suitable for multivariate time series without trend and without seasonal components.

    Parameters
    ----------
    data : _type_
        _description_
    p : _type_
        _description_
    q : _type_
        _description_
    """
    return VARMAX(data, order=(p, q))


def VARMAX_model(data, data_exog, p, q):
    """Vector Autoregression Moving-Average with Exogenous Regressors

    VARMAX is a multivariate version of the ARMAX method.

    Parameters
    ----------
    data : _type_
        _description_
    data_exog : _type_
        _description_
    p : _type_
        _description_
    q : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return VARMAX(data, exog=data_exog, order=(p, q))


def SES_model(data):
    """Simple Exponential Smoothing

    This method is suitable for univariate time series without trend and without seasonal components.

    Parameters
    ----------
    data : _type_
        _description_
    """
    return SimpleExpSmoothing(data)


def HWES(data):
    """Holt Winter’s Exponential Smoothing

    The method is suitable for univariate time series with trend and/or seasonal components.

    Parameters
    ----------
    data : _type_
        _description_
    """
    return ExponentialSmoothing(data)


def tsf_functions(data, data_exog=None, p=1, d=1, q=1, P=1, D=1, Q=1, m=12):
    """Return 11 time series forcasting functions.

    Parameters
    ----------
    data : _type_
        _description_
    data_exog : _type_, optional
        _description_, by default None
    p : int, optional
        _description_, by default 1
    d : int, optional
        _description_, by default 1
    q : int, optional
        _description_, by default 1
    P : int, optional
        _description_, by default 1
    D : int, optional
        _description_, by default 1
    Q : int, optional
        _description_, by default 1
    m : int, optional
        _description_, by default 12

    Returns
    -------
    _type_
        _description_
    """

    if len(data.shape) == 1 or data.shape[1] == 1:
        univariate_models = {
            'AR': AR_model(data, p),
            'MA': MA_model(data, q),
            'ARMA': ARMA_model(data, p, q),
            'ARIMA': ARIMA_model(data, p, d, q),
            'SARIMA': SARIMA_model(data, p, d, q, P, D, Q, m),
            'SARIMAX': SARIMAX_model(data, data_exog, p, d, q, P, D, Q, m),
            'SES': SES_model(data),
            'HWES': HWES(data)
        }
        return univariate_models
    else:
        multivariate_models = {
            'VAR': VAR_model(data),
            'VARMA': VARMA_model(data, p, q),
            'VARMAX': VARMAX_model(data, data_exog, p, q)
        }
        return multivariate_models
