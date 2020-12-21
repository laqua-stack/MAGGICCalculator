# -*- coding: utf-8 -*-
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
This script implements the MAGGIC Calculator for prediction of mortality risk of HF patients.

@author: Fabian Laqua
"""

import numpy as np
import pandas as pd
# TODO: If necessary one can add the SurvivalAnalysisMixin. Then the scikit-survival requirement will become necessary.
# from sksurv.base import SurvivalAnalysisMixin
from sklearn.base import BaseEstimator


class MAGGICCalculator(BaseEstimator):  # , SurvivalAnalysisMixin):
    """
    This class implements a sklearn/sksurv style version of the MAGGIC risk score
    for prediction of mortality for heart failure patients
    https://doi.org/10.1093/eurheartj/ehs337
    https://doi.org/10.1161/JAHA.118.009594

    All the necessary input data should be in the input DataFrame or recarray with shape (n_samples, n_features)
        - 'Age': specify age in years
        - 'Sex': specify sex. Should be either 'male', 'female' or True/1 for male sex.
        - 'Height': specify height in meters.
        - 'Weight': specify weight in kg.
        - 'Diabetes': specify if the patient suffers from diabetes mellitus.
        - 'COPD': specify if the patient suffers from COPD.
        - 'NYHA': give NYHA classification from 1 to 4.
        - 'Currsmoking': specify if the patient is currently a smoker.
        - 'HFonset18m': specify if the onset of HF was within 18m, in that case give True
        - 'BB': specify if the patient currently receives beta blockers.
        - 'ACEiARB': specify if the patient currently receives ACE-inhibitors or Angiotensin recepetor blockers.
        - 'Creatinine': give current serum creatinine in µmol/l.
        - 'EjectionFraction': give current ejection fraction in percent.
        - 'SysBP': give current systolic blood pressure.

    :param:
        - year: calculate risk for 1 or 3 years?


    :example:

    from MAGGICCalculator import MAGGICCalculator
    import pandas as pd
    from matplotlib import pyplot as plt

    X = pd.read_csv(data/example.csv)

    # imply that .predict should return 3 year probability of death
    mc = MAGGICCalculator(year=3)
    prob_3_year = mc.predict(X)
    print(prob_3_year)

    # plot 1 and 3 year survival
    y_surv_fnc = mc.predict_survival_function(X)

    fig, ax = plt.subplots()
    ax.plot(y_surv_fnc['x'], y_surv_fnc['y'], 'bo')




    """
    def __init__(self, year=1):

        if year != 1 or year != 3:
            raise AssertionError("Risk can only be calculated for 1 or 3 years")
        # init X and year
        self.year = year
        self.X = None
        self._event_times = [1, 3]
        super().__init__()
        self.colnames = ['Age',
                         'Sex',
                         'Height',
                         'Weight',
                         'Diabetes',
                         'COPD',
                         'NYHA',
                         'Currsmoking',
                         'HFonset18m',
                         'BB',
                         'ACEiARB',
                         'Creatinine',
                         'EjectionFraction'
                         ]

    def predict(self, X):
        """
        :param X: input recarray-like (pd.DataFrame, np.recarray, np. record), shape (n_samples, n_features)
        :return: y_pred: 1 or 3 year survival probability (specified in __init__())
        """
        self.X = X
        intscore = self._calculateRisk()
        y_pred_1year, y_pred_3year = self._lookup_risk_probability(intscore)
        if self.year == 1:
            return y_pred_1year
        elif self.year == 3:
            return y_pred_3year
        else:
            raise AssertionError('Invalid year.')

    def predict_survival_function(self, X):
        """
        predicts the survival function
        :param X: input recarray-like (pd.DataFrame, np.recarray, np. record), shape (n_samples, n_features)
        :return: survival_fcn: recarray, where 'x' is the time axis ([1,3] in this case) and 'y' is the event
        survival probability at the points in time 'x'
        """
        self.X = X
        survival_fcn = np.zeros((X.shape[0],
                                 2),
                                dtype=[('x', 'float64'), ('y', 'float64')],
                                )

        intscore = self._calculateRisk()
        y_pred_1year, y_pred_3year = self._lookup_risk_probability(intscore)
        survival_fcn['x'] = np.repeat(self._event_times[np.newaxis, :], X.shape[0], axis=0)
        survival_fcn['y'] = np.stack(1 - y_pred_1year, 1 - y_pred_3year, axis=-1)
        return survival_fcn

    def _check_X(self):
        X = self.X
        if isinstance(X, pd.DataFrame):
            try:
                for col in self.colnames:
                    assert col in X.loc(axis=1).columns.tolist()
            except AssertionError as e:
                raise AssertionError('Not all necessary columns are in the DataFrame' + str(e))
            return X

        elif isinstance(X, np.recarray):

            try:
                X = pd.DataFrame.from_records(X)
                for col in self.colnames:
                    assert col in X.loc(axis=1).columns.tolist()
            except AssertionError as e:
                raise AssertionError('Not all necessary columns are in the np.recarray' + str(e))
            except Exception as e:
                print('Something went wrong..')
                raise e
            return X

        elif isinstance(X, np.record):
            try:
                for col in self.colnames:
                    assert col in X.loc(axis=1).columns.tolist()
            except AssertionError as e:
                raise AssertionError('Not all necessary columns are in the np.record' + str(e))
            except Exception as e:
                print('Something went wrong..')
                raise e
            return X

        else:
            raise (AssertionError('%s is not a valid DataFrame, np.recarray or np.record ' % X.__name__))

    def _check_inputs(self, X):
        """
        check for missing or invalid input data
        :return:
        self.X
        """
        # check age
        _age = X['Age'].to_numpy()
        if np.isnan(_age).any() or np.less(_age, 18.0).any() or np.greater(_age, 110.0).any():
            raise ValueError('Height must be between 18 and 110 years.')

        # check sex
        _sex = X['Sex'].to_numpy()
        if np.isnan(_sex).any() or not (np.isin(_sex, ['male', 'female']).all()
                                        or np.isin(_sex, [True, False]).all()):
            raise ValueError('Invalid sex. Male Sex should be True or one')
        if np.isin(_sex, ['male', 'female']).all():
            _new_sex = np.where(_sex == 'male', 1, 0)
            _sex = _new_sex.copy()
        X['_risk_Sex'] = _sex.astype(int)

        # check diabetes
        _diab = X['Diabetes'].to_numpy()
        if np.isnan(_diab).any() or not (np.isin(_diab.astype(int), [0, 1]).all()
                                         or np.isin(_diab, [True, False]).all()):
            raise ValueError('Invalid data on diabetes. Must be non-nan and either bool or 0,1 or 0.0, 1.0')
        X['_risk_Diabetes'] = 3 * _diab.astype(int)

        # check copd
        _copd = X['COPD'].to_numpy()
        if np.isnan(_copd).any() or not (np.isin(_copd.astype(int), [0, 1]).all()
                                         or np.isin(_copd, [True, False]).all()):
            raise ValueError('Invalid data on COPD. Must be non-nan and either bool or 0,1 or 0.0, 1.0')
        X['_risk_COPD'] = 2 * _copd.astype(int)

        # check HF onset
        _HFonset = X['HFonset18m'].to_numpy()
        if np.isnan(_HFonset).any() or not (np.isin(_HFonset.astype(int), [0, 1]).all()
                                            or np.isin(_HFonset, [True, False]).all()):
            raise ValueError('Invalid data on HF onset. You must specify if the patient was diagnosed with HF'
                             'within the last 18 months. '
                             'Must be non-nan and either bool or 0,1 or 0.0, 1.0')
        X['_risk_HFonset18m'] = 2 * _HFonset.astype(int)

        # check for smoking status
        _smok = X['Currsmoking'].to_numpy()
        if np.isnan(_smok).any() or not (np.isin(_smok.astype(int), [0, 1]).all()
                                         or np.isin(_smok, [True, False]).all()):
            raise ValueError('Invalid data on current smoking status. '
                             'Must be non-nan and either bool or 0,1 or 0.0, 1.0')
        X['_risk_Currsmoking'] = _smok.astype(int)

        # check for NYHA status
        _nyha = X['NYHA'].to_numpy()
        if np.isnan(_nyha).any() or not (np.isin(_nyha.astype(int), [1, 2, 3, 4]).all()):
            raise ValueError('Invalid data on current NYHA class. '
                             'Must be non-nan and in 1, 2, 3, 4)')

        # check for Betablockers
        _bb = X['BB'].to_numpy()
        if np.isnan(_bb).any() or not (np.isin(_bb.astype(int), [0, 1]).all()
                                       or np.isin(_bb, [True, False]).all()):
            raise ValueError('You must specifiy if beta blockers are prescribed. '
                             'Must be non-nan and either bool or 0,1 or 0.0, 1.0')
        X['_risk_BB'] = 3 * (1 - _bb.astype(int))

        # check for ARB/ACEi
        _aceiarb = X['ACEiARB'].to_numpy()
        if np.isnan(_aceiarb).any() or not (np.isin(_aceiarb.astype(int), [0, 1]).all()
                                            or np.isin(_aceiarb, [True, False]).all()):
            raise ValueError('You must specifiy if angiotensin recpeter blockers or ACE inhibitors are prescribed. '
                             'Must be non-nan and either bool or 0,1 or 0.0, 1.0')
        X['_risk_ACEiARB'] = 1 - _aceiarb

        # check for BMI / height weight
        _bmi = self._bmicalc_m()
        if np.isnan(_bmi).any() or np.less(_bmi, 10.).any() or np.greater(_bmi, 50.).any():
            raise ValueError('BMI must be between 10 and 50 kg/m^2.')
        X['BMI'] = _bmi

        # check for systolic blood pressure
        _sysbp = X['SysBP'].to_numpy()
        if np.isnan(_sysbp).any() or np.less(_sysbp, 10.).any() or np.greater(_sysbp, 50.).any():
            raise ValueError('SysBP must be between 50 and 250 mmHg.')

        # check for creatinine
        _crea = X['Creatinine'].to_numpy()
        if np.isnan(_crea).any() or np.less(_crea, 20.).any() or np.greater(_crea, 1400.).any():
            raise ValueError('Creatinine must be between 20 and 1400 µmol/l.')

        _ef = X['EjectionFraction'].to_numpy()
        if np.isnan(_ef).any() or np.less(_ef, 1.).any() or np.greater(_ef, 95.).any():
            raise ValueError('Ejection fraction must be between 1 and 95 %%.')

        # map all int scores to int
        for col in ['Sex',
                    'Diabetes',
                    'COPD',
                    'HFonset18m',
                    'Currsmoking',
                    'NYHA',
                    'BB',
                    'ACEiARB']:
            X[col] = X[col].map(int)
        self.X = X
        return X

    def _bmicalc_m(self):
        X = self._check_X()
        _m = X['height'].to_numpy()
        _kg = X['weight'].to_numpy()
        if np.isnan(_m).any() or np.less(_m, 0.5).any() or np.greater(_m, 3.0).any():
            raise ValueError('Height must be between 0.5 and 3.0 meters.')
        if np.isnan(_kg).any() or np.less(_kg, 10.).any() or np.greater(_kg, 300.).any():
            raise ValueError('Height must be between 10 and 300 kg.')

        return _kg / (_m * _m)

    def _calculateRisk(self):
        """
        Main risk calculation.
        :return:
        """
        X = self._check_X()
        X = self._check_inputs(X)
        X['_risk'] = 0
        # add integers to risk score
        for col in ['Sex',
                    'Diabetes',
                    'COPD',
                    'HFonset18m',
                    'Currsmoking',
                    'BB',
                    'ACEiARB']:
            X['_risk'] += X['_risk' + col]

        # lookup integer scores for relative risk score
        # EF + Age
        X['_risk'] += self._risk_ef()
        # SBP + EF interaction
        X['_risk'] += self._calculate_sbpr()
        # Creatinine
        X['_risk'] += self._calculate_crear()
        # BMI
        X['_risk'] += self._calculate_bmir()
        # NYHA
        X['_risk'] += self._calculate_nyhar()
        return X['_risk']

    # noinspection PyMethodMayBeStatic
    def _lookup_risk_probability(self, int_score):
        risk1 = [0.015, 0.016, 0.018, 0.02, 0.022, 0.024, 0.027, 0.029, 0.032, 0.036, 0.039, 0.043, 0.048, 0.052, 0.058,
                 0.063, 0.07, 0.077, 0.084, 0.093, 0.102, 0.111, 0.122, 0.134, 0.147, 0.16, 0.175, 0.191, 0.209, 0.227,
                 0.248, 0.269, 0.292, 0.316, 0.342, 0.369, 0.398, 0.427, 0.458, 0.49, 0.523, 0.557, 0.591, 0.625, 0.659,
                 0.692, 0.725, 0.757, 0.787, 0.816, 0.842]
        risk3 = [0.039, 0.043, 0.048, 0.052, 0.058, 0.063, 0.07, 0.077, 0.084, 0.092, 0.102, 0.111, 0.122, 0.134, 0.146,
                 0.16, 0.175, 0.191, 0.209, 0.227, 0.247, 0.269, 0.292, 0.316, 0.342, 0.369, 0.397, 0.427, 0.458, 0.49,
                 0.523, 0.556, 0.59, 0.625, 0.658, 0.692, 0.725, 0.756, 0.787, 0.815, 0.842, 0.866, 0.889, 0.908, 0.926,
                 0.941, 0.953, 0.964, 0.973, 0.98, 0.985]
        # 1 year risk prob
        prob_1year = np.take(risk1, int_score)
        prob_3year = np.take(risk3, int_score)
        return prob_1year, prob_3year

    def _risk_ef(self):
        """
        calculator integer score for EF
        :return:
        """
        ef_cutoffs = [20., 25., 30., 35., 40., 99.]
        ef_risk_ints = np.array([7, 6, 5,  3, 2, 0])
        _ef = self.X['EjectionFraction'].to_numpy()
        _ef_mask = np.array([(_ef < cut) for cut in ef_cutoffs])

        _ef_risk = np.take(ef_risk_ints, _ef_mask.argmax(axis=0))
        self.X['_ef_risk'] = _ef_risk

        # age EF interaction
        _age = self.X['Age'].to_numpy()
        age_cutoffs = [80, 75, 70, 65, 60, 56, 17]
        age_ef_cutoffs = [30, 40, 99]
        efar_risk_ints = np.array(
            [[10, 13, 15],
             [8, 10, 12],
             [6, 8, 9],
             [4, 6, 7],
             [2, 4, 5],
             [1, 2, 3],
             [0, 0, 0],
             ]
        )
        _efage_mask = np.array([(_ef < cut) for cut in age_ef_cutoffs])
        _age_mask = np.array([(_age > cut) for cut in age_cutoffs])
        # select age axis
        efar_age_risk_ints = np.take(efar_risk_ints, _age_mask.argmax(axis=0), axis=0)
        # select EF axis
        _efar_risk = np.take_along_axis(efar_age_risk_ints, _efage_mask.argmax(axis=0)[:, np.newaxis], axis=1)
        self.X['_efar_risk'] = _efar_risk

        return _ef_risk + _efar_risk

    def _calculate_sbpr(self):
        """
        calculator integer score for sysBP
        taking EF interaction into consideration
        :return:
        _sbp_risk
        """
        if '_efar_risk' not in self.X.columns.tolist():
            raise AssertionError('efar_risk')

        sbp_cutoffs = [110, 120, 130, 140, 150, 250]
        sbp_ef_cutoffs = [30, 40, 99]
        efsbp_risk_ints = np.array(
            [[5, 3, 2],
             [4, 2, 1],
             [3, 1, 1],
             [2, 1, 0],
             [1, 0, 0],
             [0, 0, 0],
             ]
        )
        _sbp = self.X['SysBP'].to_numpy()
        _ef = self.X['EjectionFraction'].to_numpy()

        _efsbp_mask = np.array([(_ef < cut) for cut in sbp_ef_cutoffs])
        _sbp_mask = np.array([(_sbp < cut) for cut in sbp_cutoffs])
        # select sbp axis
        efsbp_sbp_risk_ints = np.take(efsbp_risk_ints, _sbp_mask.argmax(axis=0), axis=0)
        # select EF axis
        _sbp_risk = np.take_along_axis(efsbp_sbp_risk_ints, _sbp_mask.argmax(axis=0)[:, np.newaxis], axis=1)
        self.X['_sbp_risk'] = _sbp_risk
        return _sbp_risk

    def _calculate_bmir(self):
        """
        Calculates the risk contibution for specific BMI
        :return:
        """
        bmi_cutoffs = [15., 20., 25., 30., 99.]
        bmi_risk_ints = np.array([6, 5, 3,  2, 0])
        _bmi = self.X['BMI'].to_numpy()
        _bmi_mask = np.array([(_bmi < cut) for cut in bmi_cutoffs])

        _bmi_risk = np.take(bmi_risk_ints, _bmi_mask.argmax(axis=0))
        self.X['_bmi_risk'] = _bmi_risk
        return _bmi_risk

    def _calculate_crear(self):
        """
        Calculates the risk contibution for Creatinine (µmol/l)
        :return:
        """
        crea_cutoffs = [90., 110., 130., 150., 170., 210., 250., 999.]
        crea_risk_ints = np.array([0, 1, 2, 3, 4, 5, 6, 8])
        _crea = self.X['Creatinine'].to_numpy()
        _crea_mask = np.array([(_crea < cut) for cut in crea_cutoffs])

        _crea_risk = np.take(crea_risk_ints, _crea_mask.argmax(axis=0))
        self.X['_crea_risk'] = _crea_risk
        return _crea_risk

    def _calculate_nyhar(self):
        """
        Calculates the risk contribution for nyha class
        :return:
        """
        nyha_cutoffs = [1, 2, 3, 4]
        nyha_risk_ints = np.array([0, 2, 6, 8])
        _nyha = self.X['NYHA'].to_numpy()
        _nyha_mask = np.array([(_nyha == cut) for cut in nyha_cutoffs])

        _nyha_risk = np.take(nyha_risk_ints, _nyha_mask.argmax(axis=0))
        self.X['_risk_NYHA'] = _nyha_risk
        return _nyha_risk
