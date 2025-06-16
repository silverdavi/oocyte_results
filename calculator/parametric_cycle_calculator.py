# Standard
from enum import IntEnum
import json
import traceback as tb
# External
import numpy as np

# Simple replacement for map_hook
def map_hook(response):
    return response

AGE_EXTRA = [ 0, 1, 3, 5, 7 ]

AGE_MAX = 45.0
AGE_MIN = 20.0

AMH_PERCENTILES = {
    20: [ 9.78, 6.72, 5.22, 4.27, 3.60, 3.08, 2.67, 2.33, 2.04, 1.79, 1.58, 1.38, 1.21, 1.05, 0.90, 0.75, 0.62, 0.48, 0.33 ],
    22: [ 8.26, 5.68, 4.41, 3.61, 3.04, 2.60, 2.26, 1.97, 1.73, 1.52, 1.33, 1.17, 1.02, 0.88, 0.76, 0.64, 0.52, 0.40, 0.28 ],
    24: [ 6.85, 4.71, 3.66, 2.99, 2.52, 2.16, 1.87, 1.63, 1.43, 1.26, 1.10, 0.97, 0.85, 0.73, 0.63, 0.53, 0.43, 0.34, 0.23 ],
    26: [ 5.77, 3.96, 3.08, 2.52, 2.12, 1.82, 1.57, 1.37, 1.20, 1.06, 0.93, 0.81, 0.71, 0.62, 0.53, 0.44, 0.36, 0.28, 0.19 ],
    28: [ 4.96, 3.41, 2.65, 2.17, 1.82, 1.56, 1.35, 1.18, 1.04, 0.91, 0.80, 0.70, 0.61, 0.53, 0.45, 0.38, 0.31, 0.24, 0.17 ],
    30: [ 4.38, 3.02, 2.34, 1.92, 1.61, 1.38, 1.20, 1.04, 0.92, 0.80, 0.71, 0.62, 0.54, 0.47, 0.40, 0.34, 0.28, 0.21, 0.15 ],
    32: [ 4.02, 2.76, 2.15, 1.76, 1.48, 1.27, 1.10, 0.96, 0.84, 0.74, 0.65, 0.57, 0.50, 0.43, 0.37, 0.31, 0.25, 0.20, 0.14 ],
    34: [ 3.73, 2.56, 1.99, 1.63, 1.37, 1.18, 1.02, 0.89, 0.78, 0.68, 0.60, 0.53, 0.46, 0.40, 0.34, 0.29, 0.24, 0.18, 0.13 ],
    36: [ 3.28, 2.26, 1.75, 1.43, 1.21, 1.03, 0.90, 0.78, 0.69, 0.60, 0.53, 0.46, 0.41, 0.35, 0.30, 0.25, 0.21, 0.16, 0.11 ],
    38: [ 2.56, 1.76, 1.37, 1.12, 0.94, 0.81, 0.70, 0.61, 0.54, 0.47, 0.41, 0.36, 0.32, 0.27, 0.23, 0.20, 0.16, 0.13, 0.09 ],
    40: [ 1.70, 1.17, 0.91, 0.74, 0.63, 0.54, 0.46, 0.41, 0.36, 0.31, 0.27, 0.24, 0.21, 0.18, 0.16, 0.13, 0.11, 0.08, 0.06 ],
    42: [ 1.00, 0.69, 0.53, 0.44, 0.37, 0.31, 0.27, 0.24, 0.21, 0.18, 0.16, 0.14, 0.12, 0.11, 0.09, 0.08, 0.06, 0.05, 0.03 ],
    44: [ 0.54, 0.37, 0.29, 0.24, 0.20, 0.17, 0.15, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02 ],
}

BMI_MAX = 45.0
BMI_MIN = 15.0

class Condition(IntEnum):
    No = 0
    PCOS = 1
    UF = 2
    UI = 3
    DOR = 4
    ENDO = 5

CONDITION_FACTORS_BIRTH = {
    Condition.ENDO: 0.8,
}

CONDITION_FACTORS_EGGS = {
    Condition.ENDO: 0.9,
    Condition.PCOS: 1.2,
}

CONDITION_NAMES = {
    "Polycystic ovary syndrome (PCOS)": Condition.PCOS,
    "Uterine factor": Condition.UF,
    "Unexplained infertility": Condition.UI,
    "Diminished ovarian reserve": Condition.DOR,
    "Endometriosis": Condition.ENDO,
}

DEFAULT_MODE = 'gauss2'

ETHNICITY_FACTORS = {
    'asian': 0.82,
    'black': 0.8,
    'other': 0.85,
}

PARAMS_BMI = [ -4.439e-06, 0.0005938, -0.02932, 0.6203, -3.744 ]

PARAMS_POLY = [ 0.000835, -0.09641, 3.776, -57.7, 315.1 ]

PARAMS_GAUSS = [ 58.68, 28.56, 10.17 ]

PARAMS_GAUSS1 = [ 41.52, 31.53, 8.163 ]
PARAMS_GAUSS2 = [ 49.22, 15.2, 13.91 ]

ROUNDS_MAX = 3

def handler(event, context):

    try:
        request = json.loads(event['body'])
        print('request:', json.dumps(request))

        age = float(request['age'])
        weight = float(request['weight'])
        height = float(request['height'] / 100.0)
        bmi = float(request.get('bmi', weight / (height ** 2)))
        ethnicity: list = request['ethnicity']
        amh = request.get('amh')
        no_amh = amh is None
        if no_amh:
            amh = normal_amh(age)
        amh = float(amh)

        condition: list = request['condition']
        conditions = set(CONDITION_NAMES.get(c, Condition.No) for c in condition if c in CONDITION_NAMES)
        
        response = {
            "input": request,
            "results": [ compute_results(age + y, normal_amh(age + y) if no_amh else fix_amh_diff(amh, age, age + y), bmi, ethnicity, conditions) for y in AGE_EXTRA ],
        }

    except Exception as e:
        tb.print_exc()
        response = { 'error': e }

    return map_hook(response)

def babies_cycles(p1: float, eggs: int):
    probs = []
    if eggs == 0:
        p2 = 0
    else:
        a = (1 - p1) ** (1  / eggs)
        p2 = 1 - (1 - p1) * (1 + eggs * (1 - a) / a)
    for n in range(1, 1 + ROUNDS_MAX):
        p_n_1 = 1 - (1 - p1) ** n
        p_n_2 = 1 - (1 - p1) ** n - n * (1 - p1) ** (n - 1) * (p1 - p2)
        probs.append([ prettify(p_n_1), prettify(p_n_2) ])
    return probs

def bmi_factor(bmi: float):
    return np.polyval(PARAMS_BMI, bmi)

def clbr_by_age(age):
    return 1 - (1 - lbr_by_age(age)) ** oocytes_by_age_old(age)

def compute_results(age: float, amh: float, bmi: float, ethnicity: list, conditions: list):
    # normalize
    age0 = age
    age = float(np.clip(age, AGE_MIN, AGE_MAX))
    bmi = np.clip(bmi, BMI_MIN, BMI_MAX)
    # eggs
    health_factor_eggs = condition_factor(conditions, CONDITION_FACTORS_EGGS)
    print('health_factor_eggs:', health_factor_eggs)
    eggs_normal = int(np.floor(oocytes_by_age_old(age) * health_factor_eggs))
    eggs = []
    eggs_tot = 0
    for i in range(ROUNDS_MAX):
        age_i = np.clip(age + i, AGE_MIN, AGE_MAX)
        eggs_i = oocytes_by_age_new(age_i)
        eggs_i *= health_factor_eggs
        norm_amh = normal_amh(age_i)
        fixed_amh = fix_amh_diff(amh, age, age_i)
        gomp = gompertz(fixed_amh / norm_amh)
        eggs_i *= gomp
        # if i == 0:
        # print(f'age: {age} \tage_i: {age_i} \tamh: {amh}\tfixed: {fixed_amh} \tnorm: {norm_amh}\tgomp: {gomp} \teggs: {eggs_i}')
        # eggs_i = min(eggs_i, 1.3 * oocytes_by_age_new(age_i))
        eggs_i = int(np.floor(eggs_i))
        eggs_tot += eggs_i
        eggs.append(eggs_tot)
    # births
    clbr = clbr_by_age(age)
    clbr *= bmi_factor(bmi)
    clbr *= ethnicity_factor(ethnicity)
    clbr *= condition_factor(conditions, CONDITION_FACTORS_BIRTH)
    lbr = 1 - (1 - clbr) ** (1 / eggs_normal)
    clbr = 1 - (1 - lbr) ** (eggs[0])
    births = babies_cycles(clbr, eggs[0])
    # done
    return {
        'age': round(age0),
        'births': births,
        'eggs': eggs,
    }

def condition_factor(conditions, factors):
    return np.prod([ factors.get(c, 1.0) for c in conditions ])

def normal_amh(age):
    params = [ 4.6, -0.12, 30.5, 0.17 ]
    return sigmoid(age, *params)

def ethnicity_factor(ethnicity: list):
    if len(ethnicity) == 0:
        return 1.0
    return np.mean([ ETHNICITY_FACTORS.get(eth.lower(), 1.0) for eth in ethnicity ])

def amh_decline(age):
    return -0.02205*np.exp(-((age-30.57)/12.36)**2)
    
def fix_amh(old_amh, old_age, new_age):
    if new_age == old_age:
        return old_amh
    old_age = np.clip(old_age, AGE_MIN, AGE_MAX)
    old_age_bracket = old_age - old_age % 2
    amh_list = AMH_PERCENTILES[old_age_bracket]
    percentile_index = 0
    while percentile_index + 1 < len(amh_list) and old_amh < amh_list[percentile_index]:
        percentile_index += 1
    amh_factor = old_amh / amh_list[percentile_index]
    new_age = np.clip(new_age, AGE_MIN, AGE_MAX)
    new_age_bracket = new_age - new_age % 2
    new_amh = amh_factor * AMH_PERCENTILES[new_age_bracket][percentile_index]
    # print(old_age, new_age, old_amh, percentile_index, new_amh, amh_factor)
    return new_amh

def fix_amh_diff(old_amh, old_age, new_age):
    new_amh = old_amh + amh_decline(0.5*(new_age+old_age))*(new_age-old_age)
    return new_amh

def gompertz(x):
    A = 0.9
    K = 4.52
    T = 0.8
    S = 0.4
    y = A * np.exp(-np.exp(-K * (x - T))) + S
    return y

def lbr_by_age(age):
    # params = [ 11.11, 0.45, 38, 0.44]
    params = [ 13, 1.5, 33, 0.5 ]
    return sigmoid(age, *params) / 100

def oocytes_by_age_old(age):
    params = [ 22, 2, 38, 0.41 ]
    return sigmoid(age, *params)

def oocytes_by_age_new(age):
    params = [-1.4, 22, 37,-0.13]
    return sigmoid(age, *params)
 
def prettify(prob: float):
    return round(prob * 100)
    # return round(prob * 10000) / 100

def sigmoid(x, a, b, c, d):
    y = a + (b - a) / (1 + np.exp(-(x - c) * d))
    return y
