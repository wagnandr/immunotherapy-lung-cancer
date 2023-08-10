import numpy as np
import multiprocessing
from run_spherical_patient_1 import run_patient_1
from tumor_io.experimental_data import load_experimental_data


def get_eigenpairs(C):
    """Computes eigenpairs of a matrix in decreasing order of eigenvalues."""
    eigVals, eigVecs = np.linalg.eigh(C)

    idx = eigVals.argsort()[::-1]
    eigVals = eigVals[idx]
    eigVecs = eigVecs[:, idx]

    return (eigVals, eigVecs)


def approximate_gradient_at(f, vec, h):
    """
    Approximates gradient by central finite differences at a specified location.

    :param x: Location
    :param f: Function
    :param float h: Discretization parameter # mostly 1e-3
    """
    f_diffs = np.array([f(vec+h_)-f(vec-h_) for h_ in h*np.eye(len(vec))])
    derivs  = f_diffs / (2 * h)
    return derivs if np.isscalar(derivs[0]) else derivs.T


def get_active_subspaces_par(
    reference_data,
    gamma_inv,
    h,
    f,
    min_values,
    max_values,
    num_samples,
    verbose
):
    num_jobs = multiprocessing.cpu_count() 

    import joblib
    parallel = joblib.Parallel(n_jobs=num_jobs, return_as="generator")

    def create_new_sample_misfit(vector):
        qoi_forward = f(vector)
        qoi_grad = approximate_gradient_at(f, vector, h)
        misfit_grad = np.dot(np.dot(qoi_grad.T, gamma_inv), (reference_data - qoi_forward))
        return np.outer(misfit_grad.T, misfit_grad)
    
    list_vector_samples = []
    for i in range(num_samples):
        list_vector_samples.append(np.random.uniform(min_values, max_values))

    list_mat = []

    output_generator = parallel(joblib.delayed(create_new_sample_misfit)(vector) for vector in list_vector_samples)

    for i,mat in enumerate(output_generator): 
        list_mat.append(mat)

        C = 1./len(list_mat) * sum(list_mat)
        eigVals, eigVecs = get_eigenpairs(C)

        if verbose:
            print('i', i)
            print('len', len(list_mat))
            print('C', C)
            print('vals')
            print(eigVals)
            print('vecs')
            print(eigVecs)

    return eigVals, eigVecs


def get_active_subspaces_seq(
    reference_data,
    gamma_inv,
    h,
    f,
    min_values,
    max_values,
    num_samples,
    verbose
):
    list_mat = []
    for i in range(num_samples):
        vector = np.random.uniform(min_values, max_values)

        if verbose:
            print(f'vector = {vector}')

        qoi_forward = f(vector)
        qoi_grad = approximate_gradient_at(f, vector, h)
        misfit_grad = np.dot(np.dot(qoi_grad.T, gamma_inv), (reference_data - qoi_forward))
        list_mat.append(np.outer(misfit_grad.T, misfit_grad))

        C = 1./len(list_mat) * sum(list_mat)
        eigVals, eigVecs = get_eigenpairs(C)

        if verbose:
            print('i', i)
            print('C', C)
            print('vals')
            print(eigVals)
            print('vecs')
            print(eigVecs)

    return eigVals, eigVecs


def create_inverse_covariance_matrix(data, noise_relative_std_deviation):
    return np.diag(1/(data * noise_relative_std_deviation)**2)


def demo():
    dt = 1.
    # final_time = 400
    final_time = 1470
    num_samples = 100

    vector_mean = np.array([52.5, 15 * 0.7, 2.])
    vector_min = np.array([0.5, 0.5, 1]) 
    vector_max = np.array([100, 20, 2]) 
    dosage = 1.


    exp_data = load_experimental_data(t_start=0)
    time_offset=176
    exp_data.t -= time_offset
    exp_mask = np.logical_and(exp_data.t >= 0., exp_data.t <= final_time)
    reference_data = exp_data.volumes[exp_mask]
    gamma_inv = create_inverse_covariance_matrix(reference_data, 0.05) 

    h = 1e-4
    #key = 'tumor_visible'
    key = 'tumor_sigmoid'

    # dry run for extracting the keys
    data_point = run_patient_1(dt=dt, final_time=final_time, dosage=dosage, vec=vector_mean)
    data_keys = [idx for idx, t in enumerate(data_point['time'].tolist()) if t.is_integer() and t in exp_data.t.astype(int)]

    run = lambda vector: run_patient_1(dt=dt, final_time=final_time, dosage=dosage, vec=vector, verbose=False, output_enabled=False)[key][data_keys]

    get_active_subspaces_par(
        reference_data=reference_data,
        gamma_inv=gamma_inv,
        h=h,
        f=run,
        min_values=vector_min,
        max_values=vector_max,
        num_samples=num_samples,
        verbose=True
    )


if __name__ == '__main__':
    demo()