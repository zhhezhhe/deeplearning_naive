import numpy as np
import math

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    :param x: Input data of shape (N, C, H, W)
    :param w: Filter weights of shape (F, C, HH, WW)
    :param b: Biases, of shape (F,)
    :param conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    :return:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = floor(1 + (H + 2 * pad - HH) / stride)
    W' = floor(1 + (W + 2 * pad - WW) / stride)
    - cache: (x, w, b, conv_param)
    """

    out = None
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    S = conv_param['stride']
    P = conv_param['pad']
    Ho = math.floor(1 + (H + 2 * P - HH)/ S)
    Wo = math.floor(1 + (W + 2 * P - WW)/ S)
    # x_pad = np.zeros((N, C, H + 2 * P, W + 2 * P))
    # x_pad[:,:,P:P+H,P:P+W] = x
    x_pad = np.pad(x, ((0,), (0,), (P,), (P,)), 'constant')
    out = np.zeros((N, F, Ho, Wo))

    for f in range(F):
        for i in range(Ho):
            for j in range(Wo):
                # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                out[:, f, i, j] = np.sum(x_pad[:, :, i*S : i*S+HH, j*S : j*S+WW] * w[f, :, :, :], axis=(1,2,3))

        out[:, f, :, :]+=b[f]
    cache = (x, w, b, conv_param)

    return out, cache


def conv_bp_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    :param dout: Upstream derivatives.
    :param cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
    :return:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    N, F, Hi, Wi = dout.shape
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    HH = w.shape[2]
    WW = w.shape[3]
    S = conv_param['stride']
    P = conv_param['pad']

    dx, dw, db = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)
    x_pad = np.pad(x, [(0, 0), (0, 0), (P, P), (P, P)], 'constant')
    dx_pad = np.pad(dx, [(0, 0), (0, 0), (P, P), (P, P)], 'constant')
    db = np.sum(dout, axis=(0, 2, 3))

    for n in range(N):
        for i in range(Hi):
            for j in range(Wi):
                # Window we want to apply the respective f th filter over (C, HH, WW)
                x_window = x_pad[n, :, i * S : i * S + HH, j * S : j * S + WW]

                for f in range(F):
                    dw[f] += x_window * dout[n, f, i, j]
                    dx_pad[n, :, i * S : i * S + HH, j * S : j * S + WW] += w[f] * dout[n, f, i, j]

    dx = dx_pad[:, :, P : P + H, P : P + W]

    return dx, dw, db


def main():
    x_shape = (2, 3, 4, 4)  # n,c,h,w
    w_shape = (2, 3, 3, 3)  # f,c,hw,ww
    x = np.ones(x_shape)
    w = np.ones(w_shape)
    b = np.array([1, 2])
    conv_param = {'stride': 1, 'pad': 0}

    Ho = math.floor((x_shape[3] + 2 * conv_param['pad'] - w_shape[3]) / conv_param['stride'] + 1)
    Wo = Ho

    dout = np.ones((x_shape[0], w_shape[0], Ho, Wo))

    out, cache = conv_forward_naive(x, w, b, conv_param)
    dx, dw, db = conv_bp_naive(dout, cache)

    print(out.shape)
    print("dw", '--'*20)
    print(dw)
    print("dx", '--' * 20)
    print(dx)
    print("db", '--' * 20)
    print(db)



if __name__ == '__main__':
    main()