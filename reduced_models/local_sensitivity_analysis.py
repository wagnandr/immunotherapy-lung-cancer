import numpy as np
import multiprocessing
from run_spherical_patient_1 import run_patient_1_parametrized
from run_spherical_patient_2 import run_patient_2_parametrized
from parameters import TumorModelParameters


def integrate(t, f):
    ''' Integration with trapezoidal rule. '''
    return np.sum((t[1:] - t[:-1]) * (f[1:] + f[:-1]) / 2)


def default_parameters_patient1():
    return TumorModelParameters(
        pro_P=52.5 * 6.87e-3,
        MED_eff=15 * 0.7 * 0.499,
        landa=2. * 1,
        landa_HN=0.1,
        MED_dos=0.240 * 1. 
    )


def default_parameters_patient2():
    return TumorModelParameters(
        pro_P=0.55 * 6.87e-3,
        MED_eff=1.1 * 0.499,
        landa=1.*1,
        landa_HN=20.,
        MED_mas=143600,
        MED_dos=0.2 * 1.,
        MED_tim=22.,
    )


def vector_to_parameters(vec):
    return TumorModelParameters(
        pro_P=vec[0],
        MED_eff=vec[1],
        landa=vec[2],
        landa_HN=vec[3],
        sigma_HN=vec[4],
        mob_P=vec[5],
        eps_P=vec[6],
        pot_P=vec[7],
        kappa=vec[8],
        alpha_healthy=vec[9],
        alpha_tumor=vec[10],
    )


def parameters_to_vector(params):
    vec = np.zeros(11)
    vec[0] = params.pro_P
    vec[1] = params.MED_eff
    vec[2] = params.landa
    vec[3] = params.landa_HN
    vec[4] = params.sigma_HN
    vec[5] = params.mob_P
    vec[6] = params.eps_P
    vec[7] = params.pot_P
    vec[8] = params.kappa
    vec[9] = params.alpha_healthy
    vec[10] = params.alpha_tumor
    return vec


def run_local_sensitivity_analysis(dt, final_time, h, name, default_params, runner):
    default_params_vec = parameters_to_vector(default_params)

    results = runner(dt=dt, final_time=final_time, model_params=default_params, verbose=False, output_enabled=False)
    time = results['time']
    proliferative = integrate(time, results['proliferative'])
    necrotic = integrate(time, results['necrotic'])
    tumor = integrate(time, results['tumor'])
    vtumor = integrate(time, results['tumor_visible'])

    g_p_list = []
    g_n_list = []
    g_t_list = []
    g_vt_list = []

    def run(component):
        h_vec = np.eye(len(default_params_vec))[component] * h

        print(f'{component} {default_params_vec}, {default_params_vec + default_params_vec * h_vec}')

        model_params = vector_to_parameters(default_params_vec + default_params_vec * h_vec)
        results_plus = runner(dt=dt, final_time=final_time, model_params=model_params, verbose=False, output_enabled=False)
        proliferative_plus = integrate(time, results_plus['proliferative'])
        necrotic_plus = integrate(time, results_plus['necrotic'])
        tumor_plus = integrate(time, results_plus['tumor'])
        vtumor_plus = integrate(time, results_plus['tumor_visible'])

        model_params = vector_to_parameters(default_params_vec - default_params_vec * h_vec)
        results_neg = runner(dt=dt, final_time=final_time, model_params=model_params, verbose=False, output_enabled=False)
        proliferative_neg = integrate(time, results_neg['proliferative'])
        necrotic_neg = integrate(time, results_neg['necrotic'])
        tumor_neg = integrate(time, results_neg['tumor'])
        vtumor_neg = integrate(time, results_neg['tumor_visible'])

        g_p = (proliferative_plus-proliferative_neg) / (2 * h * default_params_vec[component] * proliferative)
        g_n = (necrotic_plus-necrotic_neg) / (2 * h * default_params_vec[component] * necrotic)
        g_t = (tumor_plus-tumor_neg) / (2 * h * default_params_vec[component] * tumor)
        g_vt = (vtumor_plus-vtumor_neg) / (2 * h * default_params_vec[component] * vtumor)

        return g_t, g_p, g_n, g_vt


    num_jobs = multiprocessing.cpu_count() 

    import joblib
    parallel = joblib.Parallel(n_jobs=num_jobs, return_as="generator")

    output_generator = parallel(joblib.delayed(run)(i) for i in range(len(default_params_vec)))

    for component, (g_t, g_p, g_n, g_vt) in enumerate(output_generator):
        g_t_list.append(g_t)
        g_p_list.append(g_p)
        g_n_list.append(g_n)
        g_vt_list.append(g_vt) 

        print(f'component {component}: g_t = {g_t}, g_p = {g_p}, g_n = {g_n}, g_vt = {g_vt}')

    print(f'g_t_list_{name} = {g_t_list}')
    print(f'g_p_list_{name} = {g_p_list}')
    print(f'g_n_list_{name} = {g_n_list}')
    print(f'g_vt_list_{name} = {g_vt_list}')



if __name__ == '__main__':
    dt = 1/24.
    # dt = 1
    final_time_p1, final_time_p2 = 1470, 1102
    # final_time_p1, final_time_p2 = 100, 100

    h = 1e-3

    print('Patient 1')
    run_local_sensitivity_analysis(
        dt=dt, 
        final_time=final_time_p1, 
        h=h, 
        name='p1',
        default_params=default_parameters_patient1(), 
        runner=run_patient_1_parametrized
    )

    print('Patient 2')
    run_local_sensitivity_analysis(
        dt=dt, 
        final_time=final_time_p2, 
        h=h, 
        name='p2',
        default_params=default_parameters_patient2(), 
        runner=run_patient_2_parametrized
    )
