from typing import Literal, Optional, Tuple, Union

import torch

import tsl
from tsl.utils import ensure_list

__all__ = [
    'mae',
    'nmae',
    'mape',
    'mse',
    'rmse',
    'nrmse',
    'nrmse_2',
    'r2',
    'mre',
    'pinball_loss',
    'multi_quantile_pinball_loss',
    'maape',
    'mase',
]

ReductionType = Literal['mean', 'sum', 'none']
MetricOutputType = Union[float, torch.Tensor]


def _masked_reduce(
    x: torch.Tensor,
    reduction: ReductionType,
    mask: Optional[torch.Tensor] = None,
    nan_to_zero: bool = False,
) -> MetricOutputType:
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    # 'none': return x with x[i] = 0/nan where mask[i] == False
    if reduction == 'none':
        if mask is not None:
            masked_idxs = torch.logical_not(mask)
            x[masked_idxs] = 0 if nan_to_zero else torch.nan
        return x

    # 'mean'/'sum': return mean/sum of x[mask == True]
    if mask is not None:
        x = x[mask]
    if reduction == 'mean':
        return torch.mean(x)
    elif reduction == 'sum':
        return torch.sum(x)
    else:
        raise ValueError(
            f'reduction {reduction} not allowed, must be one of '
            "['mean', 'sum', 'none']."
        )


def mae(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = 'mean',
    nan_to_zero: bool = False,
) -> MetricOutputType:
    r"""Compute the `Mean Absolute Error (MAE)
    <https://en.wikipedia.org/wiki/Mean_absolute_error>`_ between the estimate
    :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        \text{MAE} = \frac{\sum_{i=1}^n |\hat{y}_i - y_i|}{n}

    Args:
        y_hat (torch.Tensor): The estimated variable.
        y (torch.Tensor): The ground-truth variable.
        mask (torch.Tensor, optional): If provided, compute the metric using
            only the values at valid indices (with :attr:`mask` set to
            :obj:`True`). If :attr:`mask` is not :obj:`None` and
            :attr:`reduction` is :obj:`'none'`, masked indices are set to
            :obj:`nan` (see :attr:`nan_to_zero`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the sum of the output will be divided by the
            number of elements in the output, ``'sum'``: the output will be
            summed. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. This has an effect only when :attr:`mask` is
            not :obj:`None` and :attr:`reduction` is :obj:`'none'`.
            (default: :obj:`False`)

    Returns:
        float | torch.Tensor: The Mean Absolute Error.
    """
    err = torch.abs(y_hat - y)
    return _masked_reduce(err, reduction, mask, nan_to_zero)


def nmae(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = 'mean',
    nan_to_zero: bool = False,
) -> MetricOutputType:
    r"""Compute the *Normalized Mean Absolute Error* (NMAE) between the estimate
    :math:`\hat{y}` and the true value :math:`y`. The NMAE is the `Mean Absolute
    Error (MAE) <https://en.wikipedia.org/wiki/Mean_absolute_error>`_ scaled by
    the max-min range of the target data, i.e.

    .. math::

        \text{NMAE} = \frac{\frac{1}{N} \sum_{i=1}^n |\hat{y}_i - y_i|}
        {\max(y) - \min(y)}

    Args:
        y_hat (torch.Tensor): The estimated variable.
        y (torch.Tensor): The ground-truth variable.
        mask (torch.Tensor, optional): If provided, compute the metric using
            only the values at valid indices (with :attr:`mask` set to
            :obj:`True`). If :attr:`mask` is not :obj:`None` and
            :attr:`reduction` is :obj:`'none'`, masked indices are set to
            :obj:`nan` (see :attr:`nan_to_zero`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the sum of the output will be divided by the
            number of elements in the output, ``'sum'``: the output will be
            summed. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. This has an effect only when :attr:`mask` is
            not :obj:`None` and :attr:`reduction` is :obj:`'none'`.
            (default: :obj:`False`)

    Returns:
        float | torch.Tensor: The Normalized Mean Absolute Error
    """
    delta = torch.max(y) - torch.min(y) + tsl.epsilon
    err = torch.abs(y_hat - y) / delta
    return _masked_reduce(err, reduction, mask, nan_to_zero)


def mape(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = 'mean',
    nan_to_zero: bool = False,
) -> MetricOutputType:
    r"""Compute the `Mean Absolute Percentage Error (MAPE).
    <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        \text{MAPE} = \frac{1}{n} \sum_{i=1}^n \frac{|\hat{y}_i - y_i|}
        {y_i}

    Args:
        y_hat (torch.Tensor): The estimated variable.
        y (torch.Tensor): The ground-truth variable.
        mask (torch.Tensor, optional): If provided, compute the metric using
            only the values at valid indices (with :attr:`mask` set to
            :obj:`True`). If :attr:`mask` is not :obj:`None` and
            :attr:`reduction` is :obj:`'none'`, masked indices are set to
            :obj:`nan` (see :attr:`nan_to_zero`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the sum of the output will be divided by the
            number of elements in the output, ``'sum'``: the output will be
            summed. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. This has an effect only when :attr:`mask` is
            not :obj:`None` and :attr:`reduction` is :obj:`'none'`.
            (default: :obj:`False`)

    Returns:
        float | torch.Tensor: The Mean Absolute Percentage Error.
    """
    err = torch.abs((y_hat - y) / (y + tsl.epsilon)) # (y + tsl.epsilon) is not needed since we use mask_reduce
    return _masked_reduce(err, reduction, mask, nan_to_zero)


def smape(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = 'mean',
    nan_to_zero: bool = False,
) -> MetricOutputType:
    r"""Compute the `Symmetric Mean Absolute Percentage Error (SMAPE).
    <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        \text{SMAPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{2 |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}


    Args:
        y_hat (torch.Tensor): The estimated variable.
        y (torch.Tensor): The ground-truth variable.
        mask (torch.Tensor, optional): If provided, compute the metric using
            only the values at valid indices (with :attr:`mask` set to
            :obj:`True`). If :attr:`mask` is not :obj:`None` and
            :attr:`reduction` is :obj:`'none'`, masked indices are set to
            :obj:`nan` (see :attr:`nan_to_zero`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the sum of the output will be divided by the
            number of elements in the output, ``'sum'``: the output will be
            summed. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. This has an effect only when :attr:`mask` is
            not :obj:`None` and :attr:`reduction` is :obj:`'none'`.
            (default: :obj:`False`)

    Returns:
        float | torch.Tensor
    """
    num = 2 * torch.abs((y_hat - y))
    den = torch.abs(y_hat) + torch.abs(y) + tsl.epsilon
    return _masked_reduce(num / den, reduction, mask, nan_to_zero)


def mse(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = 'mean',
    nan_to_zero: bool = False,
) -> MetricOutputType:
    r"""Compute the `Mean Squared Error (MSE)
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        \text{MSE} = \frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2}{n}

    Args:
        y_hat (torch.Tensor): The estimated variable.
        y (torch.Tensor): The ground-truth variable.
        mask (torch.Tensor, optional): If provided, compute the metric using
            only the values at valid indices (with :attr:`mask` set to
            :obj:`True`). If :attr:`mask` is not :obj:`None` and
            :attr:`reduction` is :obj:`'none'`, masked indices are set to
            :obj:`nan` (see :attr:`nan_to_zero`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the sum of the output will be divided by the
            number of elements in the output, ``'sum'``: the output will be
            summed. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. This has an effect only when :attr:`mask` is
            not :obj:`None` and :attr:`reduction` is :obj:`'none'`.
            (default: :obj:`False`)

    Returns:
        float | torch.Tensor: The Mean Squared Error.
    """
    err = torch.square(y_hat - y)
    return _masked_reduce(err, reduction, mask, nan_to_zero)


def rmse(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = 'mean',
) -> MetricOutputType:
    r"""Compute the `Root Mean Squared Error (RMSE)
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        \text{RMSE} = \sqrt{\frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2}{n}}

    Args:
        y_hat (torch.Tensor): The estimated variable.
        y (torch.Tensor): The ground-truth variable.
        mask (torch.Tensor, optional): If provided, compute the metric using
            only the values at valid indices (with :attr:`mask` set to
            :obj:`True`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. ``'mean'``: the sum of the output will be
            divided by the number of elements in the output, ``'sum'``: the
            output will be summed.
            (default: ``'mean'``)

    Returns:
        float: The Root Mean Squared Error.
    """
    err = torch.square(y_hat - y)
    return torch.sqrt(_masked_reduce(err, reduction, mask))


def nrmse(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = 'mean',
) -> MetricOutputType:
    r"""Compute the `Normalized Root Mean Squared Error (NRMSE)
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.
    Normalization is by the max-min range of the data

    .. math::

        \text{NRMSE} = \frac{\sqrt{\frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2}{n}}}
        {\max y - \min y}

    Args:
        y_hat (torch.Tensor): The estimated variable.
        y (torch.Tensor): The ground-truth variable.
        mask (torch.Tensor, optional): If provided, compute the metric using
            only the values at valid indices (with :attr:`mask` set to
            :obj:`True`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. ``'mean'``: the sum of the output will be
            divided by the number of elements in the output, ``'sum'``: the
            output will be summed.
            (default: ``'mean'``)

    Returns:
        float: The range-normalzized NRMSE
    """
    delta = torch.max(y) - torch.min(y) + tsl.epsilon
    return rmse(y_hat, y, mask, reduction) / delta


def nrmse_2(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = 'mean',
) -> MetricOutputType:
    r"""Compute the `Normalized Root Mean Squared Error (NRMSE)
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.
    Normalization is by the power of the true signal :math:`y`

    .. math::

        \text{NRMSE}_2 = \frac{\sqrt{\frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2}
        {n}}}{\sum_{i=1}^n y_i^2}

    Args:
        y_hat (torch.Tensor): The estimated variable.
        y (torch.Tensor): The ground-truth variable.
        mask (torch.Tensor, optional): If provided, compute the metric using
            only the values at valid indices (with :attr:`mask` set to
            :obj:`True`).
        reduction (str): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. ``'mean'``: the sum of the output will be
            divided by the number of elements in the output, ``'sum'``: the
            output will be summed.
            (default: ``'mean'``)

    Returns:
        float: The power-normalzized NRMSE.
    """
    if mask is None:
        power_y = torch.square(y).sum()
    else:
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        power_y = torch.square(y[mask]).sum()
    return rmse(y_hat, y, mask, reduction) / power_y


def r2(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = 'mean',
    nan_to_zero: bool = False,
    mean_axis: Union[int, Tuple] = None,
) -> MetricOutputType:
    r"""Compute the `coefficient of determination
    <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_ :math:`R^2`
    between the estimate :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        R^{2} = 1 - \frac{\sum_{i} (\hat{y}_i - y_i)^2}
        {\sum_{i} (\bar{y} - y_i)^2}

    where :math:`\bar{y}=\frac{1}{n}\sum_{i=1}^n y_i` is the mean of :math:`y`.

    Args:
        y_hat (torch.Tensor): The estimated variable.
        y (torch.Tensor): The ground-truth variable.
        mask (torch.Tensor, optional): If provided, compute the metric using
            only the values at valid indices (with :attr:`mask` set to
            :obj:`True`).
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the sum of the output will be divided by the
            number of elements in the output, ``'sum'``: the output will be
            summed. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. This has an effect only when :attr:`mask` is
            not :obj:`None` and :attr:`reduction` is :obj:`'none'`.
            (default: :obj:`False`)
        mean_axis (int, Tuple, optional): the axis along which the mean of y is
            computed, to compute the variance of y needed in the denominator
            of the R2 formula.
    Returns:
        float | torch.Tensor: The :math:`R^2`.
    """
    mse_ = mse(y_hat, y, mask, reduction, nan_to_zero)

    # compute mean(s) of target data
    if mean_axis is None:
        mean_axis = tuple(range(y.dim()))
    mean_val = torch.mean(y, dim=mean_axis, keepdims=True)

    variance = mse(mean_val, y, mask, reduction, nan_to_zero)
    return 1.0 - (mse_ / variance)


def mre(
    y_hat: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> float:
    r"""Compute the MAE normalized by the L1-norm of the true signal :math:`y`,
    i.e.

    .. math::

        \text{MRE} = \frac{\sum_{i=1}^n |\hat{y}_i - y_i|}{\sum_{i=1}^n |y_i|}

    Args:
        y_hat (torch.Tensor): The estimated variable.
        y (torch.Tensor): The ground-truth variable.
        mask (torch.Tensor, optional): If provided, compute the metric using
            only the values at valid indices (with :attr:`mask` set to
            :obj:`True`).
            (default: :obj:`None`)

    Returns:
        float: The computed MRE value.
    """
    if mask is None:
        den = torch.abs(y).sum() + tsl.epsilon
    else:
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        den = torch.abs(y[mask]).sum() + tsl.epsilon
    err = mae(y_hat, y, mask, reduction='sum')
    return err / den


def pinball_loss(y_hat, y, q):
    err = y - y_hat
    return torch.maximum((q - 1) * err, q * err)


def multi_quantile_pinball_loss(y_hat, y, q):
    q = ensure_list(q)
    assert y_hat.size(0) == len(q)
    loss = torch.zeros_like(y_hat)
    for i, qi in enumerate(q):
        loss += pinball_loss(y_hat[i], y, qi)
    return loss






def maape(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    nan_to_zero: bool = False
) -> torch.Tensor:
    r"""Compute the Mean Arctangent Absolute Percentage Error (MAAPE).
    i.e.

    .. math::

        \text{MAAPE} = \frac{1}{n} \sum_{i=1}^n \arctan\left(\frac{|\hat{y}_i - y_i|}{y_i}\right)

    Args:
        y_hat (torch.Tensor): The estimated variable.
        y (torch.Tensor): The ground-truth variable.
        mask (torch.Tensor, optional): If provided, compute the metric using only
            the values at valid indices (with :attr:`mask` set to :obj:`True`).
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. (default: :obj:`False`)

    Returns:
        torch.Tensor: The Mean Arctangent Absolute Percentage Error
    """
    ratio = torch.abs(y_hat - y) / (torch.abs(y) + tsl.epsilon)  # epsilon for stability
    aape = torch.atan(ratio)
    return _masked_reduce(aape, reduction, mask, nan_to_zero)





def mase(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    nan_to_zero: bool = False
) -> torch.Tensor:
    r"""Compute the `Mean Absolute Scaled Error (MASE)`.

    .. math::

        \text{MASE} = \frac{\text{mean}(|\hat{y} - y|)}
        {\text{mean}(|y_t - y_{t-1}|)}

    Args:
        y_hat (torch.Tensor): The estimated variable.
        y (torch.Tensor): The ground-truth variable.
        mask (torch.Tensor, optional): If provided, compute the metric using only
            the values at valid indices (with :attr:`mask` set to :obj:`True`).
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. (default: :obj:`False`)

    Returns:
        torch.Tensor: The Mean Absolute Scaled Error
    """
    
    mae_err = torch.abs(y_hat - y)
    # mean absolute difference of consecutive observations
    denominator = torch.mean(torch.abs(y[:, 1:] - y[:, :-1]), dim=1)
    denominator = denominator.unsqueeze(1)
    scaled_error = mae_err / denominator
    return _masked_reduce(scaled_error, reduction, mask, nan_to_zero)