def curveFitter(data, order = 2):
    import numpy as np
    from numpy import linalg
    import scipy.stats as stats

    mi_x = data.x.min()
    mi_y = data.y.min()
    mi_z = data.deviceTime.min()
    ma_x = data.x.max()
    ma_y = data.y.max()
    ma_z = data.deviceTime.max()

    X = np.array((data.x - mi_x) / (ma_x - mi_x))
    X[np.isnan(X)] = 0.
    Y = np.array((data.y - mi_y) / (ma_y - mi_y))
    Y[np.isnan(Y)] = 0.
    Z = np.array((data.deviceTime - mi_z) / (ma_z - mi_z))
    L = np.array([float(x) for x in range(len(Z))]).T

    if order==1:
        Ay = np.array([Y, np.ones(len(Y))]).T
        cly = np.dot(linalg.pinv(Ay), L)
        Ax = np.array([X, np.ones(len(X))]).T
        clx = np.dot(linalg.pinv(Ax), L)
        Az = np.array([Z, np.ones(len(Z))]).T
        clz = np.dot(linalg.pinv(Az), L)

        df = len(Y) - len(cly)
        xhat = np.array([clx[0] * v + clx[1] for v in X])
        yhat = np.array([cly[0] * v + cly[1] for v in Y])
        zhat = np.array([clz[0] * v + clz[1] for v in Z])

        SSEx = np.sum((xhat - L) ** 2)
        SSEy = np.sum((yhat - L) ** 2)
        SSEz = np.sum((zhat - L) ** 2)

        pValY = 1 - stats.chi2.cdf(x=SSEy, df=df)
        pValX = 1 - stats.chi2.cdf(x=SSEx, df=df)
        pValZ = 1 - stats.chi2.cdf(x=SSEz, df=df)
        pv = [pValY, pValX, pValZ]

    if order == 2:
        Ay = np.array([Y ** 2, Y, np.ones(len(Y))]).T
        cly = np.dot(linalg.pinv(Ay), L)
        Ax = np.array([X ** 2, X, np.ones(len(X))]).T
        clx = np.dot(linalg.pinv(Ax), L)
        Az = np.array([Z ** 2, Z, np.ones(len(Z))]).T
        clz = np.dot(linalg.pinv(Az), L)

        df = len(Y) - len(cly)
        xhat = np.array([clx[0] * v ** 2 + clx[1] * v + clx[2] for v in X])
        yhat = np.array([cly[0] * v ** 2 + cly[1] * v + cly[2] for v in Y])
        zhat = np.array([clz[0] * v ** 2 + clz[1] * v + clz[2] for v in Z])

        SSEx = np.sum((xhat - L) ** 2)
        SSEy = np.sum((yhat - L) ** 2)
        SSEz = np.sum((zhat - L) ** 2)

        pValY = 1 - stats.chi2.cdf(x=SSEy, df=df)
        pValX = 1 - stats.chi2.cdf(x=SSEx, df=df)
        pValZ = 1 - stats.chi2.cdf(x=SSEz, df=df)
        pv = [pValY, pValX, pValZ]
    out = {'coefficients':[clx,cly,clz], 'pVals':[pValX, pValY, pValZ],
           'boundaries':[mi_x, ma_x, mi_y, ma_y, mi_z, ma_z]}
    return out

def coordParser(x):
    import re
    co = re.sub(r"[\[]", "", x)
    co = re.sub(r"[\]]", "", co)
    co = re.sub(r" ", "", co)
    co = co.split(',')
    return co

def STprep(data, i=0, j=4):
    import numpy as np
    import pandas as pd
    out = []
    i = 0
    j = 4
    while (i + j) <= data.shape[0]:
        p = True
        while p and (i + j < data.shape[0]):
            d0 = data.ix[i:(i + j), ].copy()
            o = curveFitter(d0, order=2)
            p = np.all([x > 0.8 for x in o['pVals']])
            j += 1
        else:
            if j == 5:
                d0 = data.ix[i:(i + j - 1), ].copy()
            else:
                d0 = data.ix[i:(i + j - 2), ].copy()


            flp = [d0.iloc[0]['x'], d0.iloc[0]['y'], d0.iloc[0]['deviceTime'],
                   d0.iloc[-1]['x'], d0.iloc[-1]['y'], d0.iloc[-1]['deviceTime']]
            o = curveFitter(d0, order=2)
            p = np.all([x > 0.8 for x in o['pVals']])
            if p:
                if j == 5:
                    oo = {'coef': o['coefficients'], 'pval': o['pVals'],
                          'boundary': o['boundaries'],'i': i, 'ij': i + j - 1, 'flp': flp}
                    i = i + (j - 1)
                else:
                    oo = {'coef': o['coefficients'], 'pval': o['pVals'],
                          'boundary': o['boundaries'], 'i': i, 'ij': i + j - 2, 'flp': flp}
                    i = i + (j - 2)
                out.append(oo)

                j = 4
            else:
                i += 1
                j = 4
    helmets = pd.DataFrame(out)
    return(helmets)

def quadratic(a,b,c):
    import numpy as np
    d = (b**2) - (4*a*c)
    # find two solutions
    sol1 = (-b-np.sqrt(d))/(2*a)
    sol2 = (-b+np.sqrt(d))/(2*a)
    out = np.array((sol1, sol2))
    return(out)

def bezierSolver(t):
    import numpy as np
    z1 = 1./3
    z2 = 2./3

    l1 = t['coef'][2][0] * z1 ** 2 + t['coef'][2][1] * z1 + t['coef'][2][2]
    l2 = t['coef'][2][0] * z2 ** 2 + t['coef'][2][1] * z2 + t['coef'][2][2]

    # Find intersect between each parametric curve and L at points l1 and l2
    # Unsure why it is the smallest solution, but seems to work
    q = quadratic(t['coef'][0][0], t['coef'][0][1], t['coef'][0][2] - l1)
    x1 = np.abs(q).min()

    q = quadratic(t['coef'][0][0], t['coef'][0][1], t['coef'][0][2] - l2)
    x2 = np.abs(q).min()

    q = quadratic(t['coef'][1][0], t['coef'][1][1], t['coef'][1][2] - l1)
    y1 = np.abs(q).min()

    q = quadratic(t['coef'][1][0], t['coef'][1][1], t['coef'][1][2] - l2)
    y2 = np.abs(q).min()

    u = 1. / 3
    v = 2. / 3

    a = 3 * (1 - u) * (1 - u) * u
    b = 3 * (1 - u) * u * u
    c = 3 * (1 - v) * (1 - v) * v
    d = 3 * (1 - v) * v * v

    det = a * d - b * c

    p0x = (t['flp'][0] - t['boundary'][0]) / (t['boundary'][1] - t['boundary'][0])
    p0y = (t['flp'][1] - t['boundary'][2]) / (t['boundary'][3] - t['boundary'][2])
    p0z = (t['flp'][2] - t['boundary'][4]) / (t['boundary'][5] - t['boundary'][4])
    p3x = (t['flp'][3] - t['boundary'][0]) / (t['boundary'][1] - t['boundary'][0])
    p3y = (t['flp'][4] - t['boundary'][2]) / (t['boundary'][3] - t['boundary'][2])
    p3z = (t['flp'][5] - t['boundary'][4]) / (t['boundary'][5] - t['boundary'][4])

    q1x = x1 - ((1 - u) * (1 - u) * (1 - u) * p0x + u * u * u * p3x)
    q1y = y1 - ((1 - u) * (1 - u) * (1 - u) * p0y + u * u * u * p3y)
    q1z = z1 - ((1 - u) * (1 - u) * (1 - u) * p0z + u * u * u * p3z)

    q2x = x2 - ((1 - v) * (1 - v) * (1 - v) * p0x + v * v * v * p3x)
    q2y = y2 - ((1 - v) * (1 - v) * (1 - v) * p0y + v * v * v * p3y)
    q2z = z2 - ((1 - v) * (1 - v) * (1 - v) * p0z + v * v * v * p3z)

    p1x = d * q1x - b * q2x
    p1y = d * q1y - b * q2y
    p1z = d * q1z - b * q2z
    p1x /= det
    p1y /= det
    p1z /= det

    p2x = (-c) * q1x + a * q2x
    p2y = (-c) * q1y + a * q2y
    p2z = (-c) * q1z + a * q2z
    p2x /= det
    p2y /= det
    p2z /= det

    x_ = np.array([p0x, p1x, p2x, p3x]) * (t['boundary'][1] - t['boundary'][0]) + t['boundary'][0]
    y_ = np.array([p0y, p1y, p2y, p3y]) * (t['boundary'][3] - t['boundary'][2]) + t['boundary'][2]
    z_ = np.array([p0z, p1z, p2z, p3z]) * (t['boundary'][5] - t['boundary'][4]) + t['boundary'][4]

    control = [[x_[0], y_[0], z_[0]], [x_[1], y_[1], z_[1]], [x_[2], y_[2], z_[2]], [x_[3], y_[3], z_[3]]]
    return(control)
