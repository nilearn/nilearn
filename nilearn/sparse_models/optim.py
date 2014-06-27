"""
    1st Order Optmization schemes
    Used to minimize f1 + f2
    where f1 is differentiable and f2 is non differentiable

    Algorithms implemented are:
        - Forward-Backward
        - FISTA (with or without local lipschitz constant)
        - Nesterov (with or without local lipschitz constant)
"""
import time

import numpy as np
import scipy.linalg as linalg


def wrap_function(function, *args):
    ncalls = [0]

    def function_wrapper(x, *args):
        ncalls[0] += 1
        return function(x, *args)
    return ncalls, function_wrapper


def fmin_prox(x0, f1, f1_grad, f2, f2_prox, maxit=100, lipschitz_constant=1,
        verbose=False, mode="nl", dual_gap=None, tol=1e-4,
        dual_gap_rate=1, track_convergence=False):
    """
        Solve:

        x^* = argmin f1(x) + f2(x)
               x

        Parameters:
            f1
            f2
            f1_grad : the gradient of f1
            f2_prox : the proximity operator of f2
            lipschitz_constant : the lipschitz_constant of f1_grad
            verbose : show energy decrease
            mode : can be:
                    - "fb" for Forward-Backward
                    - "f" for FISTA
                    - "fl" for FISTA with line search on lipschitz constant
                    - "n" for Nesterov
                    - "nl" for Nesterov with line search on lipschitz constant
                    - "fe": FISTA with early-stop early-stop heurostic: if
                      a local minimum of the energy is reached after n
                      iterations, and at 2*n iterations the energy still
                      hasn't gone below this value, stop and keep the
                      local minimum. This option is very useful if
                      f1_grad or f2_prox have errors
            dual_gap_rate : (default = 1), rate of computation of dual gap
            track_convergence: boolean, defaults to False
                If True, track the energy as a function of time

        Return:
            x^: the optimal point
            E: the energy at the optimal point
            if track_convergence is true: a list of time
    """
    x = x0.copy()
    if verbose or track_convergence or mode == 'fn':
        E = []

    f1_calls, f1 = wrap_function(f1)
    f2_calls, f2 = wrap_function(f2)
    f1_grad_calls, f1_grad = wrap_function(f1_grad)
    f2_prox_calls, f2_prox = wrap_function(f2_prox)
    dobj = -np.inf

    # Init auxiliary variables
    if mode == "n" or mode == "nl":
        t = 2./lipschitz_constant
        g = np.zeros(x0.shape)
        A = 0
        if mode == "nl":
            # Line search parameters
            ls_gamma_d = 1.3
            ls_gamma_u = 1.5
            max_LC_search_iter = 20
            ls_mu = 0
            v = np.zeros(x0.shape)
    elif mode in ("f", "fe"):
        t = 1
        y = x.copy()
        A = 0
        if mode == "fe":
            best_energy = np.inf
            best_i = 0
    elif mode == "fl":
        t = 1
        y = x.copy()
        A = 0
        eta = 1.5
        max_LC_search_iter = 1000

    times = list()

    for i in xrange(maxit):
        if mode == "fb": # ISTA
            x = f2_prox(x - (1/lipschitz_constant) * f1_grad(x),
                        1/lipschitz_constant)
        elif mode == "n": # Nesterov
            y = f2_prox(x0 - g, A)
            a = (t + np.sqrt(t**2 + 4*t*A)) / 2
            y = (A * x + a * y) / (A + a)
            x = f2_prox(y - f1_grad(y) / lipschitz_constant,
                        1 / lipschitz_constant)
            g = g + a*f1_grad(x)
            A += a
        elif mode == "nl": # Nesterov with local lipschitz constant
            line_search_iter = 0
            good_lipschitz_constant = True
            lipschitz_constant = lipschitz_constant / ls_gamma_u
            while good_lipschitz_constant \
                  and (line_search_iter < max_LC_search_iter):
                lipschitz_constant = lipschitz_constant * ls_gamma_u
                t = 2*(1+ls_mu*A) / lipschitz_constant
                a = (t + np.sqrt(t**2 + 4*t*A)) / 2
                y = (A*x + a*v) / (A+a)
                f1_grad_y = f1_grad(y)
                z = f2_prox(y - f1_grad_y / lipschitz_constant,
                            1 / lipschitz_constant)
                f1_grad_z = f1_grad(z)
                diffgrad = f1_grad_y - f1_grad_z
                good_lipschitz_constant = (diffgrad.ravel()
                                        * (y.ravel() - z.ravel())).sum() \
                        < linalg.norm(diffgrad.ravel())**2 / lipschitz_constant
                line_search_iter += 1

            # print lipschitz_constant
            lipschitz_constant = lipschitz_constant/ls_gamma_d
            x = z
            f1_grad_x = f1_grad_z
            x = f2_prox(y-f1_grad(y)/lipschitz_constant, 1/lipschitz_constant)
            g = g + a*f1_grad_x
            A += a
            v = f2_prox(x0 - g, A)
        elif mode in ("f", "fe"): # Fista
            x0 = x
            x = f2_prox(y-f1_grad(y)/lipschitz_constant, 1/lipschitz_constant)
            t0 = t
            t = (1.0 + np.sqrt(1 + 4*t**2)) * 0.5
            y = x + ((t0-1.0)/t) * (x-x0)
            if mode == "fe" and i > 0:
                energy = f1(x) + f2(x)
                if energy < best_energy:
                    # Energy went up, store the best x:
                    best_energy = energy
                    best_x = x
                    best_i = i
                elif i > 2 * best_i:
                    # We still are not beating the best energy
                    x = best_x
                    break
        elif mode == "fl": # Fista local
            x0 = x
            lipschitz_constant /= eta
            f1_grad_y = f1_grad(y)
            for line_search_iter in range(max_LC_search_iter):
                lipschitz_constant *= eta
                x = f2_prox(y - f1_grad_y/lipschitz_constant,
                            1.0/lipschitz_constant)
                f2_x = f2(x)
                q = f1(y) + np.sum((x-y) * f1_grad_y) + \
                    0.5 * lipschitz_constant * np.linalg.norm(x - y)**2 + f2_x
                if f1(x) + f2_x <= q:
                    break
            t0 = t
            t = (1.0 + np.sqrt(1 + 4*t**2)) * 0.5
            y = x + ((t0-1.0)/t) * (x-x0)

        # Log
        if verbose or track_convergence or mode == 'fn':
            # XXX: if mode == "fn", we are computing the energy twice
            energy = f1(x) + f2(x)
            if track_convergence:
                times.append(time.time())
            if verbose:
                print "Iteration: % 02i, energy %10.5e" % (i, energy)
            E.append(energy)

        if dual_gap and ((i+1) % dual_gap_rate) == 0:
            dobj_old = dobj
            gap, pobj, dobj = dual_gap(x)
            # dobj = max(dobj,dobj_old) # FIXME : pb with dual old unfeasible
            # gap = pobj - dobj
            if abs(gap) < tol:
                if verbose:
                    print "Optimization converged after %d iterations."%(i)
                break

    else:
        # FIXME : add proper warning message
        if dual_gap:
            gap, pobj, dobj = dual_gap(x)
            print "Optimization DID NOT CONVERGE " \
                  "after %d iterations (%e > %e)." % (i, gap, tol)

    if verbose:
        print "Optimization terminated."
        print "         Current function value: %e" % (f1(x)+f2(x))
        if dual_gap:
            print "         Final duality gap: %e" % gap
        print "         Iterations: %d" % i
        print "         F1 function evaluations: %d" % f1_calls[0]
        print "         F2 function evaluations: %d" % f2_calls[0]
        print "         F1 Gradient evaluations: %d" % f1_grad_calls[0]
        print "         F2 Proximal evaluations: %d" % f2_prox_calls[0]
        E = np.array(E)
    else:
        E = f1(x) + f2(x)
        if track_convergence:
            E = np.array(E)

    if track_convergence:
        return x, E, np.array(times)
    else:
        return x, E

if __name__ == '__main__':
    """Demo code to solve

    x^* = argmin 0.5 * ||m - G*x||_2^2 + alpha*||x||_1
             x

    It is done by setting:

        f1(x) = 0.5 * ||m - G*x||_2^2
        grad(f1)(x) = - G.T * (m - G*x)
        f2(x) = alpha * ||x||_1
        prox(f2)(y,beta) = argmin 0.5 * ||y - x||_2^2 + beta * ||x||_1
                              x
    """

    from operators import prox_l1

    n_samples = 10
    n_features = 100
    # N = 100; P = 500;
    np.random.seed(0)
    m = np.random.randn(n_samples, 1)
    G = np.random.randn(n_samples, n_features)

    alpha = 1

    lipschitz_constant = 1.05*linalg.norm(G)**2
    f1 = lambda x: 0.5 * linalg.norm(m - np.dot(G, x))**2
    f2 = lambda x: alpha * linalg.norm(x, ord=1)
    f1_grad = lambda x: - np.dot(G.T, m - np.dot(G, x))
    f2_prox = lambda x, l: prox_l1(x, l*alpha)

    def l1_dual_gap(x):
        GX = np.dot(G, x)
        A = m - GX
        dn = max(np.abs(np.dot(G.T, A)))
        if (alpha / dn) < 1:
            A = alpha * A / dn
        pobj = f1(x) + f2(x)
        dobj = 0.5 * (linalg.norm(A + GX)**2 - linalg.norm(GX, 'fro')**2)
        gap = pobj - dobj
        return gap, pobj, dobj

    x0 = np.zeros((n_features, 1))
    t0 = time.time()
    x, E = fmin_prox(x0, f1, f1_grad, f2, f2_prox, maxit=3000,
                    lipschitz_constant=lipschitz_constant,
                    verbose=True, mode="f",
                    dual_gap=l1_dual_gap)
    print time.time() - t0

    t0 = time.time()
    x, E = fmin_prox(x0, f1, f1_grad, f2, f2_prox, maxit=3000,
                    lipschitz_constant=lipschitz_constant/10,
                    verbose=True, mode="fl",
                    dual_gap=l1_dual_gap)
    print time.time() - t0

    import pylab as pl
    pl.close()
    pl.loglog(E)
    pl.show()
