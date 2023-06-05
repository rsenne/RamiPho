# %%
import FiberPho as fp
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import use
import scipy.signal
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as sfm
from statsmodels.tsa.stattools import grangercausalitytests

use('Qt5Agg')

# %%
square = lambda x: x ** 2
cube = lambda x: x ** 3
# %%
freeze_splines = fp.bSpline(45, 3, 7)
shock_splines = fp.bSpline(100, 3, 10)

# %%

"""
Fear Conditioning
"""

fc_c1_m1 = fp.fiberPhotometryCurve(r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC"
                                   r"\Test_Pho_BLA_C1_M1_FC.csv", regress=True, **{'task': 'fc', 'treatment': 'shock',
                                                                                   'anymaze_file': r"C:\Users\Ryan "
                                                                                                   r"Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_FC_Freeze - m1.csv",
                                                                                   'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Video_BLA_C1_M1_FCDLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c1_m2 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C1_M2_FC.csv",
    **{'task': 'fc', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_FC_Freeze - m2.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC"
                   r"\Test_Video_BLA_C1_M2_FCDLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c1_m3 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C1_M3_FC.csv",
    **{'task': 'fc', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_FC_Freeze - m3.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC"
                   r"\Test_Video_BLA_C1_M3_FCDLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c1_m5 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C1_M5_FC.csv",
    **{'task': 'fc', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_FC_Freeze - m5.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC"
                   r"\Test_Video_BLA_C1_M5_FCDLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c2_m6 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C2_FC_M6.csv",
    **{'task': 'fc', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_FC_Freeze - m6.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC"
                   r"\Test_Video_BLA_C2_FC_M6DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c2_m8 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C2_FC_M8.csv",
    **{'task': 'fc', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_FC_Freeze - m8.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC"
                   r"\Test_Video_BLA_C2_FC_M8DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c2_m9 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C2_FC_M9.csv",
    **{'task': 'fc', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_FC_Freeze - m9.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC"
                   r"\Test_Video_BLA_C2_FC_M9DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c3_m3 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C3_FC_Shock_M3.csv",
    **{'task': 'fc', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_C3_FC_M3.csv",
       'keystroke_offset': 2541.161408,
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Video_BLA_C3_FC_Shock_M3"
                   r"-007DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c3_m4 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C3_FC_Shock_M4.csv",
    **{'task': 'fc', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_C3_FC_M4.csv",
       'keystroke_offset': 3063.085984,
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Video_BLA_C3_FC_Shock_M4"
                   r"-001DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c3_m7 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C3_FC_Shock_M7.csv",
    **{'task': 'fc', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_C3_FC_M7.csv",
       'keystroke_offset': 3700.139104,
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Video_BLA_C3_FC_Shock_M7"
                   r"-004DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c3_m10 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C3_FC_NoShock_M10.csv",
    **{'task': 'fc', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_C3_FC_M10.csv",
       'keystroke_offset': 5960.602368,
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Video_BLA_C3_FC_NoShock_M10"
                   r"-008DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c3_m12 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C3_FC_NoShock_M12.csv",
    **{'task': 'fc', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_C3_FC_M12.csv",
       'keystroke_offset': 7143.53264,
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Video_BLA_C3_FC_NoShock_M12"
                   r"-006DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

# fc_c3_m5 = fp.fiberPhotometryCurve(r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C3_FC_NoShock_M5.csv", **{'task':'fc', 'treatment':'no-shock', 'keystroke_offset':4273.240096, 'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_C3_FC_M5.csv", 'DLC_File':r'C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test'})

fc_c4_m3 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C4_FC_Shock_M3.csv",
    **{'task': 'fc', 'treatment': 'shock', 'keystroke_offset': 2096.197152,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_C4_M3_FC.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Video_BLA_C4_FC_Shock_M3DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c4_m5 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C4_FC_NoShock_M5.csv",
    **{'task': 'fc', 'treatment': 'no-shock', 'keystroke_offset': 3237.756512,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_C4_M5_FC.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Video_BLA_C4_FC_NoShock_M5DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c4_m8 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C4_FC_NoShock_M8.csv",
    **{'task': 'fc', 'treatment': 'no-shock', 'keystroke_offset': 4854.902912,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_C4_M8_FC.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Video_BLA_C4_FC_NoShock_M8DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c4_m9 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C4_FC_NoShock_M9.csv",
    **{'task': 'fc', 'treatment': 'no-shock', 'keystroke_offset': 5441.1848,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_C4_M9_FC.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Video_BLA_C4_FC_NoShock_M9DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_c4_m11 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Pho_BLA_C4_FC_NoShock_M11.csv",
    **{'task': 'fc', 'treatment': 'no-shock', 'keystroke_offset': 6609.466528,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\BLA_C4_M11_FC.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\FC\Test_Video_BLA_C4_FC_NoShock_M11DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

fc_cohort = fp.fiberPhotometryExperiment(fc_c1_m1, fc_c1_m2, fc_c1_m3, fc_c1_m5, fc_c2_m6, fc_c2_m8, fc_c2_m9, fc_c3_m3,
                                         fc_c3_m4, fc_c3_m7, fc_c3_m10, fc_c3_m12, fc_c4_m3, fc_c4_m5, fc_c4_m8,
                                         fc_c4_m9, fc_c4_m11)

# %%
design_mat = pd.DataFrame(data={},
                          columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing', 'Shock_Spline1',
                                   'Shock_Spline2', 'Shock_Spline3'
                              , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7', 'Shock_Spline8', 'Shock_Spline9', 'Shock_Spline10',
                                   'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                              , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7', 'end_Spline1',
                                   'end_Spline2', 'end_Spline3'
                              , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7', 'Animal'],
                          dtype="float64")
i = 1
for trace in list(getattr(fc_cohort, 'fc-shock').values())[0]:
    shock_map, shock_dict = shock_splines.create_spline_map([1150, 1710, 2270, 2830], len(trace.DF_F_Signals['GCaMP']))
    freeze_map, freeze_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['start_freezing'],
                                                               len(trace.DF_F_Signals['GCaMP']))

    init_map, init_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['end_freezing'],
                                                           len(trace.DF_F_Signals['GCaMP']))
    shock_map = np.vstack((trace.DF_F_Signals['GCaMP'], trace.DF_F_Signals['Isobestic_GCaMP'],
                           scipy.signal.resample(trace.behavioral_data['DLC']['velocity'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           scipy.signal.resample(trace.behavioral_data['DLC']['acceleration'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           trace.behavioral_data['Anymaze']['freeze_vector'][:len(trace.DF_F_Signals['GCaMP'])],
                           shock_map, freeze_map, init_map,
                           np.full(len(trace.DF_F_Signals['GCaMP']), i)))
    design_mat = pd.concat((design_mat, pd.DataFrame(shock_map.transpose(),
                                                     columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration',
                                                              'Freezing', 'Shock_Spline1',
                                                              'Shock_Spline2', 'Shock_Spline3'
                                                         , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7', 'Shock_Spline8', 'Shock_Spline9', 'Shock_Spline10',                                                              'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                                                         , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6',
                                                              'Start_Spline7', 'end_Spline1', 'end_Spline2',
                                                              'end_Spline3'
                                                         , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7',
                                                              'Animal'], dtype="float64")),
                           ignore_index=True, axis=0)
    i += 1

# %%
fc_shock_model = sfm.glm('DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration) '
        '+ cube(Acceleration) + Freezing + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C(Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Shock_Spline8 + C(Animal)*Shock_Spline9 + C(Animal)*Shock_Spline10', data=design_mat,
                         family=sm.families.Gaussian()).fit()

# alternate_shock_model1 = sfm.glm(
#     'DF_F ~   Isobestic + Freezing + Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
#         'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
#         'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
#         'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
#         'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
#         'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
#         'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
#         'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
#         'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
#         'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
#         'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat,
#     family=sm.families.Gaussian()).fit()
#
# alternate_shock_model2 = sfm.glm(
#     'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
#         '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
#         'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + '
#         'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
#         'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
#         'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
#         'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7  +  C(Animal)*end_Spline1 + C('
#         'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
#         'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat,
#     family=sm.families.Gaussian()).fit()
#
# alternate_shock_model3 = sfm.glm(
#     'DF_F ~   Isobestic +Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
#         '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
#         'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
#         'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
#         ' C(Animal) + C(Animal)*Shock_Spline1 + C('
#         'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
#         'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
#         'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
#         'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7', data=design_mat,
#     family=sm.families.Gaussian()).fit()
#
# alternate_shock_model4 = sfm.glm(
#     'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
#         '+ cube(Acceleration) + Freezing + Start_Spline1 + Start_Spline2 + '
#         'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
#         'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
#         'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
#         'Animal)*Shock_Spline2 + C(Animal)*Start_Spline1 + C('
#         'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
#         'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
#         'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
#         'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat,
#     family=sm.families.Gaussian()).fit()
#
# alternate_shock_model5 = sfm.glm(
#     'DF_F ~  + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration) '
#         '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
#         'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
#         'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
#         'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
#         'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
#         'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
#         'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
#         'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
#         'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
#         'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
#         'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat,
#     family=sm.families.Gaussian()).fit()



model = scipy.stats.chi2(84)
# %%
design_mat_null = pd.DataFrame(data={},
                          columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing', 'Shock_Spline1',
                                   'Shock_Spline2', 'Shock_Spline3'
                              , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7',
                                   'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                              , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7', 'end_Spline1',
                                   'end_Spline2', 'end_Spline3'
                              , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7', 'Animal'],
                          dtype="float64")
i = 1
for trace in list(getattr(fc_cohort, 'fc-no-shock').values())[0]:
    shock_map, shock_dict = shock_splines.create_spline_map([1120, 1680, 2240, 2800], len(trace.DF_F_Signals['GCaMP']))
    freeze_map, freeze_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['start_freezing'],
                                                               len(trace.DF_F_Signals['GCaMP']))

    init_map, init_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['end_freezing'],
                                                           len(trace.DF_F_Signals['GCaMP']))
    shock_map = np.vstack((trace.DF_F_Signals['GCaMP'], trace.DF_F_Signals['Isobestic_GCaMP'],
                           scipy.signal.resample(trace.behavioral_data['DLC']['velocity'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           scipy.signal.resample(trace.behavioral_data['DLC']['acceleration'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           trace.behavioral_data['Anymaze']['freeze_vector'][:len(trace.DF_F_Signals['GCaMP'])],
                           shock_map, freeze_map, init_map,
                           np.full(len(trace.DF_F_Signals['GCaMP']), i)))
    design_mat_null = pd.concat((design_mat_null, pd.DataFrame(shock_map.transpose(),
                                                     columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing', 'Shock_Spline1',
                                   'Shock_Spline2', 'Shock_Spline3'
                              , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7',
                                   'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                              , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7', 'end_Spline1',
                                   'end_Spline2', 'end_Spline3'
                              , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7', 'Animal'], dtype="float64")),
                           ignore_index=True, axis=0)
    i += 1
#%%
time_lag_plot, time_ax = plt.subplots(1, 2)
time_ax[0].scatter(design_mat.DF_F[1:], design_mat.DF_F[:-1])
time_ax[1].stem(sm.tsa.acf(design_mat.DF_F))
time_ax[0].set_xlabel('k=1')
time_ax[0].set_ylabel('Un-lagged Data')
time_ax[1].set_xlabel('k-th Time Lag')
time_ax[1].set_ylabel("ACF")

#%%
compare_plot , compare_ax= plt.subplots()
compare_ax.plot(range(len(fc_shock_model.predict(design_mat[design_mat['Animal']==np.float64(8)]))), fc_shock_model.predict(design_mat[design_mat['Animal']==np.float64(8)]))
compare_ax.plot(range(len(design_mat[design_mat['Animal']==8].DF_F)), design_mat[design_mat['Animal']==8].DF_F)
compare_ax.set_xlabel(r'$y_n$')
compare_ax.set_ylabel(r'$\frac{dF}{F}$')
compare_ax.legend(['Predicted', 'Observed'])

#%%
df_hist, df_ax = plt.subplots(2)
df_ax[0].hist(design_mat.DF_F, bins=50, density=True)
df_ax[1].hist(design_mat_null.DF_F, bins=50, density=True)
df_ax[0].set_ylabel('Pr. Density')
df_ax[1].set_ylabel('Pr. Density')
df_ax[1].set_xlabel(r'$\frac{dF}{F}$')

qq_plots, qq_ax = plt.subplots(1, 2)
sm.qqplot(design_mat_null.DF_F, scipy.stats.norm(np.mean(design_mat_null.DF_F), np.std(design_mat_null.DF_F)), line='45', ax=qq_ax[0])
sm.qqplot(design_mat.DF_F, scipy.stats.norm(np.mean(design_mat.DF_F), np.std(design_mat.DF_F)), line='45', ax=qq_ax[1])


test_figure, test_ax= plt.subplots(1,2)
test_ax[0].hist(design_mat[design_mat['DF_F'] < 1*np.std(design_mat.DF_F)].DF_F, bins=50, density=True)
sm.qqplot(design_mat[design_mat['DF_F'] < 1*np.std(design_mat.DF_F)].DF_F,
          scipy.stats.norm(np.mean(design_mat[design_mat['DF_F'] < 1*np.std(design_mat.DF_F)].DF_F),
                           np.std(design_mat[design_mat['DF_F'] < 1*np.std(design_mat.DF_F)].DF_F)), line='45', ax=test_ax[1])
test_ax[0].set_xlabel(r'$\frac{dF}{F}$')
test_ax[0].set_ylabel(r'Pr. Density')
# %%
fc_null_model = sfm.glm('DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration) '
        '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_null,
                         family=sm.families.Gaussian()).fit()

alternate_null_model1 = sfm.glm(
    'DF_F ~   Isobestic + Freezing + Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_null,
    family=sm.families.Gaussian()).fit()

alternate_null_model2 = sfm.glm(
    'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7  +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_null,
    family=sm.families.Gaussian()).fit()

alternate_null_model3 = sfm.glm(
    'DF_F ~   Isobestic +Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        ' C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7', data=design_mat_null,
    family=sm.families.Gaussian()).fit()

alternate_null_model4 = sfm.glm(
    'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_null,
    family=sm.families.Gaussian()).fit()

alternate_null_model5 = sfm.glm(
    'DF_F ~  Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_null,
    family=sm.families.Gaussian()).fit()

model = scipy.stats.chi2(1)
#%%
compare_plot , compare_ax= plt.subplots()
compare_ax.plot(range(len(fc_null_model.predict(design_mat_null[design_mat_null['Animal']==np.float64(3)]))), fc_null_model.predict(design_mat_null[design_mat_null['Animal']==np.float64(3)]))
compare_ax.plot(range(len(design_mat_null[design_mat_null['Animal']==3].DF_F)), design_mat_null[design_mat_null['Animal']==3].DF_F)
compare_ax.set_xlabel(r'$y_n$')
compare_ax.set_ylabel(r'$\frac{dF}{F}$')
compare_ax.legend(['Predicted', 'Observed'])
# %%

"""
Recall
"""

recall_c1_m1 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C1_M1_Recall.csv",
    **{'task': 'recall', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_Recall_Freeze - m1.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C1_M1_RecallDLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c1_m2 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C1_M2_Recall.csv",
    **{'task': 'recall', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_Recall_Freeze - m2.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C1_M2_RecallDLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c1_m3 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C1_M3_Recall.csv",
    **{'task': 'recall', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_Recall_Freeze - m3.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C1_M3_RecallDLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c1_m5 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C1_M5_Recall.csv",
    **{'task': 'recall', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_Recall_Freeze - m5.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C1_M5_RecallDLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c2_m6 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C2_RECALL_M6.csv",
    **{'task': 'recall', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_Recall_Freeze - m6.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall"
                   r"\Test_Video_BLA_C2_RECALL_M6DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c2_m8 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C2_RECALL_M8.csv",
    **{'task': 'recall', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_Recall_Freeze - m8.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall"
                   r"\Test_Video_BLA_C2_RECALL_M8DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c2_m9 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C2_RECALL_M9.csv",
    **{'task': 'recall', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_Recall_Freeze - m9.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall"
                   r"\Test_Video_BLA_C2_RECALL_M9DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c3_m3 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C3_Recall_Shock_M3.csv",
    **{'task': 'recall', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_C3_Recall_M3.csv",
       'keystroke_offset': 605.571008,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C3_Recall_Shock_M3-003DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c3_m4 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C3_Recall_Shock_M4.csv",
    **{'task': 'recall', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_C3_Recall_M4.csv",
       'keystroke_offset': 1282.591936,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C3_Recall_Shock_M4-006DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c3_m7 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C3_Recall_Shock_M7.csv",
    **{'task': 'recall', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_C3_Recall_M7.csv",
       'keystroke_offset': 3114.81968,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C3_Recall_Shock_M7-002DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

# recall_c3_m5 = fp.fiberPhotometryCurve(
#     r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C3_Recall_NoShock_M5.csv",
#     **{'task': 'recall', 'treatment': 'no-shock', 'keystroke_offset': 2004.30272,
#        'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_C3_Recall_M5.csv", "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C3_Recall_NoShock_M5-004DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c3_m10 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C3_Recall_NoShock_M10.csv",
    **{'task': 'recall', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_C3_Recall_M10.csv",
       'keystroke_offset': 4320.436832,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C3_Recall_NoShock_M10-005DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

# recall_c3_m12 = fp.fiberPhotometryCurve(r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C3_Recall_NoShock_M12.csv", **{'task':'recall', 'treatment':'no-shock', 'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_C3_Recall_M12.csv", 'keystroke_offset':5459.102592})

recall_c4_m5 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C4_Recall_NoShock_M5.csv",
    **{'task': 'recall', 'treatment': 'no-shock', 'keystroke_offset': 1245.03664,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_C4_M5_Recall.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C4_Recall_NoShock_M5DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c4_m8 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C4_Recall_NoShock_M8.csv",
    **{'task': 'recall', 'treatment': 'no-shock', 'keystroke_offset': 3018.708128,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_C4_M8_Recall.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C4_Recall_NoShock_M8DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c4_m9 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C4_Recall_NoShock_M9.csv",
    **{'task': 'recall', 'treatment': 'no-shock', 'keystroke_offset': 3609.86528,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_C4_M9_Recall.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C4_Recall_NoShock_M9DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c4_m11 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C4_Recall_NoShock_M11.csv",
    **{'task': 'recall', 'treatment': 'no-shock', 'keystroke_offset': 4694.270016,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_C4_M11_Recall.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C4_Recall_NoShock_M9DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_c4_m3 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Pho_BLA_C4_Recall_Shock_M3.csv",
    **{'task': 'recall', 'treatment': 'shock', 'keystroke_offset': 656.99232,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\BLA_C4_M11_Recall.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Recall\Test_Video_BLA_C4_Recall_Shock_M3DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

recall_cohort = fp.fiberPhotometryExperiment(recall_c1_m1, recall_c1_m2, recall_c1_m3, recall_c1_m5, recall_c2_m6,
                                             recall_c2_m8, recall_c2_m9, recall_c3_m3, recall_c3_m4, recall_c3_m7,
                                             recall_c3_m10, recall_c4_m3, recall_c4_m11, recall_c4_m9,
                                             recall_c4_m8, recall_c4_m5)
# %%
i = 1
design_mat_recall = pd.DataFrame(data={},
                                 columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing', 'Shock_Spline1',
                                          'Shock_Spline2', 'Shock_Spline3'
                                     , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7', 'Shock_Spline8', 'Shock_Spline9', 'Shock_Spline10',
                                          'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                                     , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7', 'end_Spline1', 'end_Spline2', 'end_Spline3'
                                     , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7', 'Animal'],
                                 dtype="float64")
for trace in list(getattr(recall_cohort, 'recall-shock').values())[0]:
    shock_map, shock_dict = shock_splines.create_spline_map([1120, 1680, 2240, 2800], len(trace.DF_F_Signals['GCaMP']))

    freeze_map, freeze_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['start_freezing'],
                                                               len(trace.DF_F_Signals['GCaMP']))

    init_map, init_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['end_freezing'],
                                                           len(trace.DF_F_Signals['GCaMP']))

    shock_map = np.vstack((trace.DF_F_Signals['GCaMP'], trace.DF_F_Signals['Isobestic_GCaMP'],
                           scipy.signal.resample(trace.behavioral_data['DLC']['velocity'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           scipy.signal.resample(trace.behavioral_data['DLC']['acceleration'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           trace.behavioral_data['Anymaze']['freeze_vector'][:len(trace.DF_F_Signals['GCaMP'])],
                           shock_map, freeze_map, init_map,
                           np.full(len(trace.DF_F_Signals['GCaMP']), i)))
    print(np.shape(shock_map))
    design_mat_recall = pd.concat((design_mat_recall, pd.DataFrame(data=shock_map.transpose(),
                                                                   columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing', 'Shock_Spline1',
                                          'Shock_Spline2', 'Shock_Spline3'
                                     , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7', 'Shock_Spline8', 'Shock_Spline9', 'Shock_Spline10',
                                          'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                                     , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7', 'end_Spline1', 'end_Spline2', 'end_Spline3'
                                     , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7', 'Animal'],
                                                                   dtype="float64")), ignore_index=True, axis=0)
    i += 1

# %%
recall_shock_model = sfm.glm('DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration) '
        '+ cube(Acceleration) + Freezing  + C(Animal) +  C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_recall,
                         family=sm.families.Gaussian()).fit()

alternate_recall_model1 = sfm.glm(
    'DF_F ~   Isobestic + Freezing + Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_recall,
    family=sm.families.Gaussian()).fit()

alternate_recall_model2 = sfm.glm(
    'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7  +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_recall,
    family=sm.families.Gaussian()).fit()

alternate_recall_model3 = sfm.glm(
    'DF_F ~   Isobestic +Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        ' C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7', data=design_mat_recall,
    family=sm.families.Gaussian()).fit()

alternate_recall_model4 = sfm.glm(
    'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_recall,
    family=sm.families.Gaussian()).fit()

alternate_recall_model5 = sfm.glm(
    'DF_F ~  Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_recall,
    family=sm.families.Gaussian()).fit()

model = scipy.stats.chi2(6)

#%%
compare_plot , compare_ax= plt.subplots()
compare_ax.plot(range(len(recall_shock_model.predict(design_mat_recall[design_mat_recall['Animal']==np.float64(2)]))), recall_shock_model.predict(design_mat_recall[design_mat_recall['Animal']==np.float64(2)]))
compare_ax.plot(range(len(design_mat_recall[design_mat_recall['Animal']==2].DF_F)), design_mat_recall[design_mat_recall['Animal']==2].DF_F)
compare_ax.set_xlabel(r'$y_n$')
compare_ax.set_ylabel(r'$\frac{dF}{F}$')
compare_ax.legend(['Predicted', 'Observed'])
# %%
i = 1
design_mat_recall_null = pd.DataFrame(data={},
                                 columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing', 'Shock_Spline1',
                                          'Shock_Spline2', 'Shock_Spline3'
                                     , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7',
                                          'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                                     , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7', 'end_Spline1', 'end_Spline2', 'end_Spline3'
                                     , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7', 'Animal'],
                                 dtype="float64")
for trace in list(getattr(recall_cohort, 'recall-no-shock').values())[0]:
    shock_map, shock_dict = shock_splines.create_spline_map([1120, 1680, 2240, 2800], len(trace.DF_F_Signals['GCaMP']))

    freeze_map, freeze_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['start_freezing'],
                                                               len(trace.DF_F_Signals['GCaMP']))

    init_map, init_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['end_freezing'],
                                                           len(trace.DF_F_Signals['GCaMP']))

    shock_map = np.vstack((trace.DF_F_Signals['GCaMP'], trace.DF_F_Signals['Isobestic_GCaMP'],
                           scipy.signal.resample(trace.behavioral_data['DLC']['velocity'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           scipy.signal.resample(trace.behavioral_data['DLC']['acceleration'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           trace.behavioral_data['Anymaze']['freeze_vector'][:len(trace.DF_F_Signals['GCaMP'])],
                           shock_map, freeze_map, init_map,
                           np.full(len(trace.DF_F_Signals['GCaMP']), i)))
    print(np.shape(shock_map))
    design_mat_recall_null = pd.concat((design_mat_recall_null, pd.DataFrame(data=shock_map.transpose(),
                                                                   columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing', 'Shock_Spline1',
                                          'Shock_Spline2', 'Shock_Spline3'
                                     , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7',
                                          'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                                     , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7', 'end_Spline1', 'end_Spline2', 'end_Spline3'
                                     , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7', 'Animal'],
                                                                   dtype="float64")), ignore_index=True, axis=0)
    i += 1
# %%
recall_shock_model_null = sfm.glm('DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration) '
        '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_recall_null,
                         family=sm.families.Gaussian()).fit()

alternate_recall_model1_null = sfm.glm(
    'DF_F ~   Isobestic + Freezing + Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_recall_null,
    family=sm.families.Gaussian()).fit()

alternate_recall_model2_null = sfm.glm(
    'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7  +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_recall_null,
    family=sm.families.Gaussian()).fit()

alternate_recall_model3_null = sfm.glm(
    'DF_F ~   Isobestic +Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        ' C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7', data=design_mat_recall_null,
    family=sm.families.Gaussian()).fit()

alternate_recall_model4_null = sfm.glm(
    'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_recall_null,
    family=sm.families.Gaussian()).fit()

alternate_recall_model5_null = sfm.glm(
    'DF_F ~  Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_recall_null,
    family=sm.families.Gaussian()).fit()

model = scipy.stats.chi2(6)

#%%
compare_plot, compare_ax = plt.subplots()
compare_ax.plot(range(len(recall_shock_model_null.predict(design_mat_recall_null[design_mat_recall_null['Animal']==np.float64(2)]))), recall_shock_model_null.predict(design_mat_recall_null[design_mat_recall_null['Animal']==np.float64(2)]))
compare_ax.plot(range(len(design_mat_recall_null[design_mat_recall_null['Animal']==2].DF_F)), design_mat_recall_null[design_mat_recall_null['Animal']==2].DF_F)
compare_ax.set_xlabel(r'$y_n$')
compare_ax.set_ylabel(r'$\frac{dF}{F}$')
compare_ax.legend(['Predicted', 'Observed'])
#%%
df_hist, df_ax = plt.subplots(2)
df_ax[0].hist(design_mat_recall.DF_F, bins=50, density=True)
df_ax[1].hist(design_mat_recall_null.DF_F, bins=50, density=True)
df_ax[0].set_ylabel('Pr. Density')
df_ax[1].set_ylabel('Pr. Density')
df_ax[1].set_xlabel(r'$\frac{dF}{F}$')

qq_plots, qq_ax = plt.subplots(1, 2)
sm.qqplot(design_mat_recall_null.DF_F, scipy.stats.norm(np.mean(design_mat_recall_null.DF_F), np.std(design_mat_recall_null.DF_F)), line='45', ax=qq_ax[0])
sm.qqplot(design_mat_recall.DF_F, scipy.stats.norm(np.mean(design_mat_recall.DF_F), np.std(design_mat_recall.DF_F)), line='45', ax=qq_ax[1])
# %%

"""
Extinction 1
"""

ex_c1_m1 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C1_EXT1_M1.csv",
    **{'task': 'ext1', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\BLA_EXT1_Freeze - m1.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C1_EXT1_M1DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c1_m2 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C1_EXT1_M2.csv",
    **{'task': 'ext1', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\BLA_EXT1_Freeze - m2.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C1_EXT1_M2DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c1_m3 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C1_EXT1_M3.csv",
    **{'task': 'ext1', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\BLA_EXT1_Freeze - m3.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C1_EXT1_M3DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c1_m5 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C1_EXT1_M5.csv",
    **{'task': 'ext1', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\BLA_EXT1_Freeze - m5.csv",
       'DLC_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C1_EXT1_M5DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c2_m6 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C2_EXT1_M6.csv",
    **{'task': 'ext1', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\BLA_EXT1_Freeze - m6.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C2_EXT1_M6DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c2_m8 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C2_EXT1_M8.csv",
    **{'task': 'ext1', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\BLA_EXT1_Freeze - m8.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C2_EXT1_M8DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c2_m9 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C2_EXT1_M9.csv",
    **{'task': 'ext1', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\BLA_EXT1_Freeze - m9.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C2_EXT1_M9DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c3_m3 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C3_EXT1_Shock_M3.csv",
    **{'task': 'ext1', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\BLA_C3_EXT1_M3.csv",
       'keystroke_offset': 680.33152,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C3_EXT1_Shock_M3-002DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c3_m4 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C3_EXT1_Shock_M4.csv",
    **{'task': 'ext1', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\BLA_C3_EXT1_M4.csv",
       'keystroke_offset': 1859.806464,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C3_EXT1_Shock_M4-005DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c3_m7 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C3_EXT1_Shock_M7.csv",
    **{'task': 'ext1', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\BLA_C3_EXT1_M7.csv",
       'keystroke_offset': 5393.57184,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C3_EXT1_Shock_M7-006DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c3_m5 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C3_EXT1_NoShock_M5.csv",
    **{'task': 'ext1', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\BLA_C3_EXT1_M5.csv",
       'keystroke_offset': 3025.582464,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C3_EXT1_NoShock_M5-009DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c3_m10 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C3_EXT1_NoShock_M10.csv",
    **{'task': 'ext1', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\BLA_C3_EXT1_M10.csv",
       'keystroke_offset': 7801.399136,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C3_EXT1_NoShock_M10-003DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c4_m5 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C4_EXT1_NoShock_M5.csv",
    **{'task': 'ext1', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Unnamed experiment - BLA_C4_M5_EXT1.csv",
       'keystroke_offset': 1467.031552,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C4_Ext1_NoShock_M5DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c4_m8 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C4_EXT1_NoShock_M8.csv",
    **{'task': 'ext1', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Unnamed experiment - BLA_C4_M8_EXT1.csv",
       'keystroke_offset': 4851.319008,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C4_Ext1_NoShock_M8DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c4_m9 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C4_EXT1_NoShock_M9.csv",
    **{'task': 'ext1', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Unnamed experiment - BLA_C4_M9_EXT1.csv",
       'keystroke_offset': 6026.581472,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C4_Ext1_NoShock_M9DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex_c4_m11 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C4_EXT1_NoShock_M11.csv",
    **{'task': 'ext1', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Unnamed experiment - BLA_C4_M11_EXT1.csv",
       'keystroke_offset': 8426.802752,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Video_BLA_C4_Ext1_NoShock_M11DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

# ex_c4_m3 = fp.fiberPhotometryCurve(r"C:\Users\Ryan
# Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Test_Pho_BLA_C4_EXT1_Shock_M3.csv", **{'task':'ext1',
# 'treatment':'shock','anymaze_file':r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex1\Unnamed
# experiment - BLA_C4_M3_EXT1.csv", 'keystroke_offset':344.918176,"DLC_file":})

ex_cohort = fp.fiberPhotometryExperiment(ex_c1_m1, ex_c1_m2, ex_c1_m3, ex_c1_m5, ex_c2_m6, ex_c2_m8, ex_c2_m9, ex_c3_m3,
                                         ex_c3_m4, ex_c3_m7, ex_c3_m5, ex_c3_m10, ex_c4_m5, ex_c4_m8, ex_c4_m9,
                                         ex_c4_m11)

# %%
i = 1
design_mat_ext1 = pd.DataFrame(data={},
                                 columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing', 'Shock_Spline1',
                                          'Shock_Spline2', 'Shock_Spline3'
                                     , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7', 'Shock_Spline8', 'Shock_Spline9', 'Shock_Spline10',
                                          'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                                     , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7', 'end_Spline1', 'end_Spline2', 'end_Spline3'
                                     , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7', 'Animal'],
                                 dtype="float64")
for trace in list(getattr(ex_cohort, 'ext1-shock').values())[0]:
    shock_map, shock_dict = shock_splines.create_spline_map([1120, 1680, 2240, 2800], len(trace.DF_F_Signals['GCaMP']))

    freeze_map, freeze_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['start_freezing'],
                                                               len(trace.DF_F_Signals['GCaMP']))

    init_map, init_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['end_freezing'],
                                                           len(trace.DF_F_Signals['GCaMP']))

    shock_map = np.vstack((trace.DF_F_Signals['GCaMP'], trace.DF_F_Signals['Isobestic_GCaMP'],
                           scipy.signal.resample(trace.behavioral_data['DLC']['velocity'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           scipy.signal.resample(trace.behavioral_data['DLC']['acceleration'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           trace.behavioral_data['Anymaze']['freeze_vector'][:len(trace.DF_F_Signals['GCaMP'])],
                           shock_map, freeze_map, init_map,
                           np.full(len(trace.DF_F_Signals['GCaMP']), i)))
    print(np.shape(shock_map))
    design_mat_ext1 = pd.concat((design_mat_ext1, pd.DataFrame(data=shock_map.transpose(),
                                                                   columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing', 'Shock_Spline1',
                                          'Shock_Spline2', 'Shock_Spline3'
                                     , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7', 'Shock_Spline8', 'Shock_Spline9', 'Shock_Spline10',
                                          'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                                     , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7', 'end_Spline1', 'end_Spline2', 'end_Spline3'
                                     , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7', 'Animal'],
                                                                   dtype="float64")), ignore_index=True, axis=0)
    i += 1
#%%
ext1_shock_model = sfm.glm(
        'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration) '
        '+ cube(Acceleration) + Freezing + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Shock_Spline8+ + C(Animal)*Shock_Spline9+  C(Animal)*Shock_Spline10 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_ext1,
        family=sm.families.Gaussian()).fit()

alternate_ext1_model1 = sfm.glm(
        'DF_F ~   Isobestic + Freezing + Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_ext1,
        family=sm.families.Gaussian()).fit()

alternate_ext1_model2 = sfm.glm(
        'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7  +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_ext1,
        family=sm.families.Gaussian()).fit()

alternate_ext1_model3 = sfm.glm(
        'DF_F ~   Isobestic +Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        ' C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7', data=design_mat_ext1,
        family=sm.families.Gaussian()).fit()

alternate_ext1_model4 = sfm.glm(
        'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_ext1,
        family=sm.families.Gaussian()).fit()

alternate_ext1_model5 = sfm.glm(
        'DF_F ~  Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
        '+ cube(Acceleration) + Freezing + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_ext1,
        family=sm.families.Gaussian()).fit()

model = scipy.stats.chi2(6)
#%%
compare_plot, compare_ax = plt.subplots()
compare_ax.plot(range(len(ext1_shock_model.predict(
    design_mat_ext1[design_mat_ext1['Animal'] == np.float64(1)]))), ext1_shock_model.predict(
    design_mat_ext1[design_mat_ext1['Animal'] == np.float64(1)]))
compare_ax.plot(range(len(design_mat_ext1[design_mat_ext1['Animal'] == 1].DF_F)),
                    design_mat_ext1[design_mat_ext1['Animal'] == 1].DF_F)
compare_ax.set_xlabel(r'$y_n$')
compare_ax.set_ylabel(r'$\frac{dF}{F}$')
compare_ax.legend(['Predicted', 'Observed'])
#%%
i = 1
design_mat_ext1_null = pd.DataFrame(data={},
                                   columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing',
                                            'Shock_Spline1',
                                            'Shock_Spline2', 'Shock_Spline3'
                                       , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7',
                                            'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                                       , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7', 'end_Spline1', 'end_Spline2',
                                            'end_Spline3'
                                       , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7', 'Animal'],
                                   dtype="float64")
for trace in list(getattr(ex_cohort, 'ext1-no-shock').values())[0]:
        shock_map, shock_dict = shock_splines.create_spline_map([1120, 1680, 2240, 2800],
                                                                len(trace.DF_F_Signals['GCaMP']))

        freeze_map, freeze_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['start_freezing'],
                                                                   len(trace.DF_F_Signals['GCaMP']))

        init_map, init_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['end_freezing'],
                                                               len(trace.DF_F_Signals['GCaMP']))

        shock_map = np.vstack((trace.DF_F_Signals['GCaMP'], trace.DF_F_Signals['Isobestic_GCaMP'],
                               scipy.signal.resample(trace.behavioral_data['DLC']['velocity'],
                                                     len(trace.DF_F_Signals['GCaMP'])),
                               scipy.signal.resample(trace.behavioral_data['DLC']['acceleration'],
                                                     len(trace.DF_F_Signals['GCaMP'])),
                               trace.behavioral_data['Anymaze']['freeze_vector'][:len(trace.DF_F_Signals['GCaMP'])],
                               shock_map, freeze_map, init_map,
                               np.full(len(trace.DF_F_Signals['GCaMP']), i)))
        print(np.shape(shock_map))
        design_mat_ext1_null = pd.concat((design_mat_ext1_null, pd.DataFrame(data=shock_map.transpose(),
                                                                   columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing',
                                            'Shock_Spline1',
                                            'Shock_Spline2', 'Shock_Spline3'
                                       , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7',
                                            'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                                       , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7', 'end_Spline1', 'end_Spline2',
                                            'end_Spline3'
                                       , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7', 'Animal'],
                                                                   dtype="float64")), ignore_index=True, axis=0)
        i += 1
        # %%
        ext_null_shock_model = sfm.glm(
            'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration) '
            '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
            'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
            'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
            'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
            'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
            'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
            'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
            'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
            'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
            'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
            'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_ext1_null,
            family=sm.families.Gaussian()).fit()

        alternate_ext_null_model1 = sfm.glm(
            'DF_F ~   Isobestic + Freezing + Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
            'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
            'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
            'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
            'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
            'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
            'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
            'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
            'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
            'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
            'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_ext1_null,
            family=sm.families.Gaussian()).fit()

        alternate_ext_null_model2 = sfm.glm(
            'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
            '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
            'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + '
            'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
            'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
            'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
            'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7  +  C(Animal)*end_Spline1 + C('
            'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
            'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_ext1_null,
            family=sm.families.Gaussian()).fit()

        alternate_ext_null_model3 = sfm.glm(
            'DF_F ~   Isobestic +Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
            '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
            'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
            'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
            ' C(Animal) + C(Animal)*Shock_Spline1 + C('
            'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
            'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
            'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
            'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7', data=design_mat_ext1_null,
            family=sm.families.Gaussian()).fit()

        alternate_ext_null_model4 = sfm.glm(
            'DF_F ~  Isobestic + Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
            '+ cube(Acceleration) + Freezing + Start_Spline1 + Start_Spline2 + '
            'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
            'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
            'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
            'Animal)*Shock_Spline2 + C(Animal)*Start_Spline1 + C('
            'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
            'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
            'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
            'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_ext1_null,
            family=sm.families.Gaussian()).fit()

        alternate_ext_null_model5 = sfm.glm(
            'DF_F ~  Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration)'
            '+ cube(Acceleration) + Freezing + Start_Spline1 + Start_Spline2 + '
            'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
            'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
            'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
            'Animal)*Shock_Spline2 + C(Animal)*Start_Spline1 + C('
            'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
            'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
            'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
            'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7', data=design_mat_ext1_null,
            family=sm.families.Gaussian()).fit()

        model = scipy.stats.chi2(6)
#%%
df_hist, df_ax = plt.subplots(2)
df_ax[0].hist(design_mat_ext1.DF_F, bins=50, density=True)
df_ax[1].hist(design_mat_ext1_null.DF_F, bins=50, density=True)
df_ax[0].set_ylabel('Pr. Density')
df_ax[1].set_ylabel('Pr. Density')
df_ax[1].set_xlabel(r'$\frac{dF}{F}$')

qq_plots, qq_ax = plt.subplots(1, 2)
sm.qqplot(design_mat_ext1_null.DF_F, scipy.stats.norm(np.mean(design_mat_ext1_null.DF_F), np.std(design_mat_ext1_null.DF_F)), line='45', ax=qq_ax[0])
sm.qqplot(design_mat_ext1.DF_F, scipy.stats.norm(np.mean(design_mat_ext1.DF_F), np.std(design_mat_ext1.DF_F)), line='45', ax=qq_ax[1])
#%%
compare_plot, compare_ax = plt.subplots()
compare_ax.plot(range(len(ext_null_shock_model.predict(
    design_mat_ext1_null[design_mat_ext1_null['Animal'] == np.float64(1)]))), ext_null_shock_model.predict(
    design_mat_ext1_null[design_mat_ext1_null['Animal'] == np.float64(1)]))
compare_ax.plot(range(len(design_mat_ext1_null[design_mat_ext1_null['Animal'] == 1].DF_F)),
                    design_mat_ext1_null[design_mat_ext1_null['Animal'] == 1].DF_F)
compare_ax.set_xlabel(r'$y_n$')
compare_ax.set_ylabel(r'$\frac{dF}{F}$')
compare_ax.legend(['Predicted', 'Observed'])
# %%
"""
Extinction 2
"""
ex2_c2_m6 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C2_EXT2_M6.csv",
    **{'task': 'ext2', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\BLA_EXT2_Freeze - m6.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C2_EXT2_M6DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex2_c2_m8 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C2_EXT2_M8.csv",
    **{'task': 'ext2', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\BLA_EXT2_Freeze - m8.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C2_EXT2_M8DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex2_c2_m9 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C2_EXT2_M9.csv",
    **{'task': 'ext2', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\BLA_EXT2_Freeze - m9.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C2_EXT2_M9DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex2_c3_m5 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C3_EXT2_NoShock_M5.csv",
    **{'task': 'ext2', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\BLA_C3_EXT2_M5.csv",
       'keystroke_offset': 3446.345696,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C3_EXT2_NoShock_M5-004DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex2_c3_m10 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C3_EXT2_NoShock_M10.csv",
    **{'task': 'ext2', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\BLA_C3_EXT2_M10.csv",
       'keystroke_offset': 7889.468832,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C3_EXT2_NoShock_M10-001DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

# ex2_c3_m12 = fp.fiberPhotometryCurve(r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C3_EXT2_NoShock_M12.csv", **{'task':'ext2', 'treatment':'no-shock','anymaze_file':r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\BLA_C3_EXT2_M12.csv", 'keystroke_offset':10105.574144, "DLC_file":r})

ex2_c3_m3 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C3_EXT2_Shock_M3.csv",
    **{'task': 'ext2', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\BLA_C3_EXT2_M3.csv",
       'keystroke_offset': 837.830496,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C3_EXT2_Shock_M3-002DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex2_c3_m4 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C3_EXT2_Shock_M4.csv",
    **{'task': 'ext2', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\BLA_C3_EXT2_M4.csv",
       'keystroke_offset': 1988.277088,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C3_EXT2_Shock_M4-003DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex2_c3_m7 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C3_EXT2_Shock_M7.csv",
    **{'task': 'ext2', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\BLA_C3_EXT2_M7.csv",
       'keystroke_offset': 5725.579232,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C3_EXT2_Shock_M7-009DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex2_c4_m3 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C4_EXT2_Shock_M3.csv",
    **{'task': 'ext2', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Unnamed experiment - BLA_C4_M3_EXT2.csv",
       'keystroke_offset': 512.121024,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C4_Ext2_Shock_M3DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex2_c4_m5 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C4_Ext2_NoShock_M5.csv",
    **{'task': 'ext2', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Unnamed experiment - BLA_C4_M5_EXT2.csv",
       'keystroke_offset': 1792.029024,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C4_Ext2_NoShock_M5DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex2_c4_m8 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C4_Ext2_NoShock_M8.csv",
    **{'task': 'ext2', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Unnamed experiment - BLA_C4_M8_EXT2.csv",
       'keystroke_offset': 5169.341664,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C4_Ext2_NoShock_M8DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex2_c4_m9 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C4_Ext2_NoShock_M9.csv",
    **{'task': 'ext2', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Unnamed experiment - BLA_C4_M9_EXT2.csv",
       'keystroke_offset': 6253.77344,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C4_Ext2_NoShock_M9DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex2_c4_m11 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Pho_BLA_C4_Ext2_NoShock_M11.csv",
    **{'task': 'ext2', 'treatment': 'no-shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Unnamed experiment - BLA_C4_M11_EXT2.csv",
       'keystroke_offset': 8537.516416,
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex2\Test_Video_BLA_C4_Ext2_NoShock_M11DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})
ex2_cohort = fp.fiberPhotometryExperiment(ex2_c2_m6, ex2_c2_m8, ex2_c2_m9, ex2_c3_m3, ex2_c3_m4, ex2_c3_m5, ex2_c3_m7,
                                          ex2_c3_m10, ex2_c4_m3, ex2_c4_m5, ex2_c4_m8, ex2_c4_m9, ex2_c4_m11)
#%%
i = 1
design_mat_ext2_shock = pd.DataFrame(data={},
                                     columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing',
                                              'Shock_Spline1',
                                              'Shock_Spline2', 'Shock_Spline3'
                                         , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7',
                                              'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                                         , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7', 'end_Spline1', 'end_Spline2',
                                              'end_Spline3'
                                         , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'Animal'],
                                     dtype="float64")
for trace in list(getattr(ex2_cohort, 'ext2-shock').values())[0]:
    shock_map, shock_dict = shock_splines.create_spline_map([1120, 1680, 2240, 2800],
                                                            len(trace.DF_F_Signals['GCaMP']))

    freeze_map, freeze_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['start_freezing'],
                                                               len(trace.DF_F_Signals['GCaMP']))

    init_map, init_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['end_freezing'],
                                                           len(trace.DF_F_Signals['GCaMP']))

    shock_map = np.vstack((trace.DF_F_Signals['GCaMP'], trace.DF_F_Signals['Isobestic_GCaMP'],
                           scipy.signal.resample(trace.behavioral_data['DLC']['velocity'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           scipy.signal.resample(trace.behavioral_data['DLC']['acceleration'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           trace.behavioral_data['Anymaze']['freeze_vector'][:len(trace.DF_F_Signals['GCaMP'])],
                           shock_map, freeze_map, init_map,
                           np.full(len(trace.DF_F_Signals['GCaMP']), i)))
    print(np.shape(shock_map))
    design_mat_ext2_shock = pd.concat((design_mat_ext2_shock, pd.DataFrame(data=shock_map.transpose(),
                                                                           columns=['DF_F', 'Isobestic', 'Velocity',
                                                                                    'Acceleration',
                                                                                    'Freezing', 'Shock_Spline1',
                                                                                    'Shock_Spline2', 'Shock_Spline3'
                                                                               , 'Shock_Spline4', 'Shock_Spline5',
                                                                                    'Shock_Spline6', 'Shock_Spline7',
                                                                                    'Start_Spline1', 'Start_Spline2',
                                                                                    'Start_Spline3'
                                                                               , 'Start_Spline4', 'Start_Spline5',
                                                                                    'Start_Spline6', 'Start_Spline7',
                                                                                    'end_Spline1', 'end_Spline2',
                                                                                    'end_Spline3'
                                                                               , 'end_Spline4', 'end_Spline5',
                                                                                    'end_Spline6',
                                                                                    'end_Spline7', 'Animal'],
                                                                           dtype="float64")),
                                      ignore_index=True, axis=0)
    i += 1
    # %%
    ext2_shock_model = sfm.glm(
        'DF_F ~  Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration) '
        '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7',
        data=design_mat_ext2_shock, family=sm.families.Gaussian()).fit()

# %%
i = 1
design_mat_ext2_noshock = pd.DataFrame(data={},
                                       columns=['DF_F', 'Isobestic', 'Velocity', 'Acceleration', 'Freezing',
                                                'Shock_Spline1',
                                                'Shock_Spline2', 'Shock_Spline3'
                                           , 'Shock_Spline4', 'Shock_Spline5', 'Shock_Spline6', 'Shock_Spline7',
                                                'Start_Spline1', 'Start_Spline2', 'Start_Spline3'
                                           , 'Start_Spline4', 'Start_Spline5', 'Start_Spline6', 'Start_Spline7',
                                                 'end_Spline1', 'end_Spline2',
                                                'end_Spline3'
                                           , 'end_Spline4', 'end_Spline5', 'end_Spline6', 'end_Spline7', 'Animal'],
                                       dtype="float64")
for trace in list(getattr(ex2_cohort, 'ext2-no-shock').values())[0]:
    shock_map, shock_dict = shock_splines.create_spline_map([1120, 1680, 2240, 2800],
                                                            len(trace.DF_F_Signals['GCaMP']))

    freeze_map, freeze_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['start_freezing'],
                                                               len(trace.DF_F_Signals['GCaMP']))

    init_map, init_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['end_freezing'],
                                                           len(trace.DF_F_Signals['GCaMP']))

    shock_map = np.vstack((trace.DF_F_Signals['GCaMP'], trace.DF_F_Signals['Isobestic_GCaMP'],
                           scipy.signal.resample(trace.behavioral_data['DLC']['velocity'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           scipy.signal.resample(trace.behavioral_data['DLC']['acceleration'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           trace.behavioral_data['Anymaze']['freeze_vector'][:len(trace.DF_F_Signals['GCaMP'])],
                           shock_map, freeze_map, init_map,
                           np.full(len(trace.DF_F_Signals['GCaMP']), i)))
    print(np.shape(shock_map))
    design_mat_ext2_noshock = pd.concat((design_mat_ext2_noshock, pd.DataFrame(data=shock_map.transpose(),
                                                                               columns=['DF_F', 'Isobestic', 'Velocity',
                                                                                        'Acceleration',
                                                                                        'Freezing', 'Shock_Spline1',
                                                                                        'Shock_Spline2', 'Shock_Spline3'
                                                                                   , 'Shock_Spline4', 'Shock_Spline5',
                                                                                        'Shock_Spline6',
                                                                                        'Shock_Spline7',
                                                                                        'Start_Spline1',
                                                                                        'Start_Spline2',
                                                                                        'Start_Spline3'
                                                                                   , 'Start_Spline4', 'Start_Spline5',
                                                                                        'Start_Spline6',
                                                                                        'Start_Spline7',

                                                                                        'end_Spline1', 'end_Spline2',
                                                                                        'end_Spline3'
                                                                                   , 'end_Spline4', 'end_Spline5',
                                                                                        'end_Spline6',
                                                                                        'end_Spline7', 'Animal'],
                                                                               dtype="float64")),
                                        ignore_index=True, axis=0)
    i += 1
# %%
ext2_noshock_model = sfm.glm(
    'DF_F ~  Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration) '
    '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
    'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
    'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
    'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
    'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
    'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
    'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
    'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
    'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 + C(Animal)*Start_Spline8 + C(Animal)*Start_Spline9+  C(Animal)*end_Spline1 + C('
    'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
    'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7 + C(Animal)*end_Spline8 + C(Animal)*end_Spline9',
    data=design_mat_ext2_noshock, family=sm.families.Gaussian()).fit()
#%%
df_hist, df_ax = plt.subplots(2)
df_ax[0].hist(design_mat_ext2_shock.DF_F, bins=50, density=True)
df_ax[1].hist(design_mat_ext2_noshock.DF_F, bins=50, density=True)
df_ax[0].set_ylabel('Pr. Density')
df_ax[1].set_ylabel('Pr. Density')
df_ax[1].set_xlabel(r'$\frac{dF}{F}$')

qq_plots, qq_ax = plt.subplots(1, 2)
sm.qqplot(design_mat_ext2_noshock.DF_F, scipy.stats.norm(np.mean(design_mat_ext2_noshock.DF_F), np.std(design_mat_ext2_noshock.DF_F)), line='45', ax=qq_ax[0])
sm.qqplot(design_mat_ext2_shock.DF_F, scipy.stats.norm(np.mean(design_mat_ext2_shock.DF_F), np.std(design_mat_ext2_shock.DF_F)), line='45', ax=qq_ax[1])
# %%
"""
Extinction 3
"""
ex3_c2_m6 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C2_EXT3_M6.csv",
    **{'task': 'ext3', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\BLA_EXT3_Freeze - m6.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C3_EXT3_NoShock_M6-002DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex3_c2_m8 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C2_EXT3_M8.csv",
    **{'task': 'ext3', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\BLA_EXT3_Freeze - m8.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C3_EXT3_Shock_M3-008DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex3_c2_m9 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C2_EXT3_M9.csv",
    **{'task': 'ext3', 'treatment': 'shock',
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\BLA_EXT3_Freeze - m9.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C3_EXT3_NoShock_M9-003DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex3_c3_m5 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C3_EXT3_NoShock_M5.csv",
    **{'task': 'ext3', 'treatment': 'no-shock', 'keystroke_offset': 6996.107264,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\BLA_C3_EXT3_M5.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C3_EXT3_NoShock_M5DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex3_c3_m10 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C3_EXT3_NoShock_M10.csv",
    **{'task': 'ext3', 'treatment': 'no-shock', 'keystroke_offset': 12079.995328,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\BLA_C3_EXT3_M10.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C3_EXT3_NoShock_M10-004DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex3_c3_m12 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C3_EXT3_NoShock_M12.csv",
    **{'task': 'ext3', 'treatment': 'no-shock', 'keystroke_offset': 14442.623104,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\BLA_C3_EXT3_M12.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C3_EXT3_NoShock_M12-005DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex3_c3_m3 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C3_EXT3_Shock_M3.csv",
    **{'task': 'ext3', 'treatment': 'shock', 'keystroke_offset': 1181.574016,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\BLA_C3_EXT3_M3.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C3_EXT3_Shock_M3-008DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex3_c3_m4 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C3_EXT3_Shock_M4.csv",
    **{'task': 'ext3', 'treatment': 'shock', 'keystroke_offset': 2314.226688,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\BLA_C3_EXT3_M4.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C3_EXT3_Shock_M4-001DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex3_c3_m7 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C3_EXT3_Shock_M7.csv",
    **{'task': 'ext3', 'treatment': 'shock', 'keystroke_offset': 3633.144256,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\BLA_C3_EXT3_M7.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C3_EXT3_Shock_M7-007DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex3_c4_m3 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C4_EXT3_Shock_M3.csv",
    **{'task': 'ext3', 'treatment': 'shock', 'keystroke_offset': 437.725696,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Unnamed experiment - BLA_C4_M3_EXT3.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C4_Ext3_Shock_M3DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex3_c4_m5 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C4_Ext3_NoShock_M5.csv",
    **{'task': 'ext3', 'treatment': 'no-shock', 'keystroke_offset': 1566.595552,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Unnamed experiment - BLA_C4_M5_EXT3.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C4_Ext3_NoShock_M5DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

ex3_c4_m8 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C4_Ext3_NoShock_M8.csv",
    **{'task': 'ext3', 'treatment': 'no-shock', 'keystroke_offset': 5176.39696,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Unnamed experiment - BLA_C4_M8_EXT3.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C4_Ext3_NoShock_M8DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})

# ex3_c4_m9 = fp.fiberPhotometryCurve(r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C4_Ext3_NoShock_M9.csv", **{'task':'ext3', 'treatment':'no-shock','keystroke_offset':7261.288384,'anymaze_file':r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Unnamed experiment - BLA_C4_M9_EXT3.csv"})

ex3_c4_m11 = fp.fiberPhotometryCurve(
    r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Pho_BLA_C4_Ext3_NoShock_M11.csv",
    **{'task': 'ext3', 'treatment': 'no-shock', 'keystroke_offset': 8823.546848,
       'anymaze_file': r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Unnamed experiment - BLA_C4_M11_EXT3.csv",
       "DLC_file": r"C:\Users\Ryan Senne\Documents\RLS_Team_Data\BLA_Extinction\Ex3\Test_Video_BLA_C4_Ext3_NoShock_M11DLC_resnet50_engram_round2_front_fcApr21shuffle1_400000.csv"})
ex3_cohort = fp.fiberPhotometryExperiment(ex3_c2_m6, ex3_c2_m8, ex3_c2_m9, ex3_c3_m3, ex3_c3_m4, ex3_c3_m5, ex3_c3_m7,
                                          ex3_c3_m10, ex3_c3_m12, ex3_c4_m3, ex3_c4_m5, ex3_c4_m8, ex3_c4_m11)

# %%
i = 1
design_mat_ext3_shock = pd.DataFrame(data={},
                                     columns=['DF_F', 'Isobestic', 'Velocity',
                                                                                        'Acceleration',
                                                                                        'Freezing', 'Shock_Spline1',
                                                                                        'Shock_Spline2', 'Shock_Spline3'
                                                                                   , 'Shock_Spline4', 'Shock_Spline5',
                                                                                        'Shock_Spline6',
                                                                                        'Shock_Spline7',
                                                                                        'Start_Spline1',
                                                                                        'Start_Spline2',
                                                                                        'Start_Spline3'
                                                                                   , 'Start_Spline4', 'Start_Spline5',
                                                                                        'Start_Spline6',
                                                                                        'Start_Spline7',

                                                                                        'end_Spline1', 'end_Spline2',
                                                                                        'end_Spline3'
                                                                                   , 'end_Spline4', 'end_Spline5',
                                                                                        'end_Spline6',
                                                                                        'end_Spline7', 'Animal'],
                                     dtype="float64")
for trace in list(getattr(ex3_cohort, 'ext3-shock').values())[0]:
    shock_map, shock_dict = shock_splines.create_spline_map([1120, 1680, 2240, 2800],
                                                            len(trace.DF_F_Signals['GCaMP']))

    freeze_map, freeze_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['start_freezing'],
                                                               len(trace.DF_F_Signals['GCaMP']))

    init_map, init_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['end_freezing'],
                                                           len(trace.DF_F_Signals['GCaMP']))

    shock_map = np.vstack((trace.DF_F_Signals['GCaMP'], trace.DF_F_Signals['Isobestic_GCaMP'],
                           scipy.signal.resample(trace.behavioral_data['DLC']['velocity'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           scipy.signal.resample(trace.behavioral_data['DLC']['acceleration'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           trace.behavioral_data['Anymaze']['freeze_vector'][:len(trace.DF_F_Signals['GCaMP'])],
                           shock_map, freeze_map, init_map,
                           np.full(len(trace.DF_F_Signals['GCaMP']), i)))
    print(np.shape(shock_map))
    design_mat_ext3_shock = pd.concat((design_mat_ext3_shock, pd.DataFrame(data=shock_map.transpose(),
                                                                           columns=['DF_F', 'Isobestic', 'Velocity',
                                                                                        'Acceleration',
                                                                                        'Freezing', 'Shock_Spline1',
                                                                                        'Shock_Spline2', 'Shock_Spline3'
                                                                                   , 'Shock_Spline4', 'Shock_Spline5',
                                                                                        'Shock_Spline6',
                                                                                        'Shock_Spline7',
                                                                                        'Start_Spline1',
                                                                                        'Start_Spline2',
                                                                                        'Start_Spline3'
                                                                                   , 'Start_Spline4', 'Start_Spline5',
                                                                                        'Start_Spline6',
                                                                                        'Start_Spline7',

                                                                                        'end_Spline1', 'end_Spline2',
                                                                                        'end_Spline3'
                                                                                   , 'end_Spline4', 'end_Spline5',
                                                                                        'end_Spline6',
                                                                                        'end_Spline7', 'Animal'],
                                                                           dtype="float64")),
                                      ignore_index=True, axis=0)
    i += 1
    # %%
    ext3_shock_model = sfm.glm(
        'DF_F ~  Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration) '
        '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
        'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
        'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
        'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
        'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
        'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
        'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
        'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
        'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
        'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
        'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7',
        data=design_mat_ext3_shock, family=sm.families.Gaussian()).fit()

# %%
i = 1
design_mat_ext3_noshock = pd.DataFrame(data={},
                                       columns=['DF_F', 'Isobestic', 'Velocity',
                                                                                        'Acceleration',
                                                                                        'Freezing', 'Shock_Spline1',
                                                                                        'Shock_Spline2', 'Shock_Spline3'
                                                                                   , 'Shock_Spline4', 'Shock_Spline5',
                                                                                        'Shock_Spline6',
                                                                                        'Shock_Spline7',
                                                                                        'Start_Spline1',
                                                                                        'Start_Spline2',
                                                                                        'Start_Spline3'
                                                                                   , 'Start_Spline4', 'Start_Spline5',
                                                                                        'Start_Spline6',
                                                                                        'Start_Spline7',

                                                                                        'end_Spline1', 'end_Spline2',
                                                                                        'end_Spline3'
                                                                                   , 'end_Spline4', 'end_Spline5',
                                                                                        'end_Spline6',
                                                                                        'end_Spline7', 'Animal'],
                                       dtype="float64")
for trace in list(getattr(ex3_cohort, 'ext3-no-shock').values())[0]:
    shock_map, shock_dict = shock_splines.create_spline_map([1120, 1680, 2240, 2800],
                                                            len(trace.DF_F_Signals['GCaMP']))

    freeze_map, freeze_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['start_freezing'],
                                                               len(trace.DF_F_Signals['GCaMP']))

    init_map, init_dict = freeze_splines.create_spline_map(trace.behavioral_data['Anymaze']['end_freezing'],
                                                           len(trace.DF_F_Signals['GCaMP']))

    shock_map = np.vstack((trace.DF_F_Signals['GCaMP'], trace.DF_F_Signals['Isobestic_GCaMP'],
                           scipy.signal.resample(trace.behavioral_data['DLC']['velocity'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           scipy.signal.resample(trace.behavioral_data['DLC']['acceleration'],
                                                 len(trace.DF_F_Signals['GCaMP'])),
                           trace.behavioral_data['Anymaze']['freeze_vector'][:len(trace.DF_F_Signals['GCaMP'])],
                           shock_map, freeze_map, init_map,
                           np.full(len(trace.DF_F_Signals['GCaMP']), i)))
    print(np.shape(shock_map))
    design_mat_ext3_noshock = pd.concat((design_mat_ext3_noshock, pd.DataFrame(data=shock_map.transpose(),
                                                                               columns=['DF_F', 'Isobestic', 'Velocity',
                                                                                        'Acceleration',
                                                                                        'Freezing', 'Shock_Spline1',
                                                                                        'Shock_Spline2', 'Shock_Spline3'
                                                                                   , 'Shock_Spline4', 'Shock_Spline5',
                                                                                        'Shock_Spline6',
                                                                                        'Shock_Spline7',
                                                                                        'Start_Spline1',
                                                                                        'Start_Spline2',
                                                                                        'Start_Spline3'
                                                                                   , 'Start_Spline4', 'Start_Spline5',
                                                                                        'Start_Spline6',
                                                                                        'Start_Spline7',

                                                                                        'end_Spline1', 'end_Spline2',
                                                                                        'end_Spline3'
                                                                                   , 'end_Spline4', 'end_Spline5',
                                                                                        'end_Spline6',
                                                                                        'end_Spline7', 'Animal'],
                                                                               dtype="float64")),
                                        ignore_index=True, axis=0)
    i += 1
# %%
ext3_noshock_model = sfm.glm(
    'DF_F ~  Velocity + square(Velocity) + cube(Velocity) + Acceleration + square(Acceleration) '
    '+ cube(Acceleration) + Freezing+ Shock_Spline1 + Shock_Spline2 + Shock_Spline3 + Shock_Spline4 + '
    'Shock_Spline5 + Shock_Spline6 + Shock_Spline7 + Start_Spline1 + Start_Spline2 + '
    'Start_Spline3 + Start_Spline4 + Start_Spline5 + Start_Spline6 + Start_Spline7 + '
    'end_Spline1 + end_Spline2 + end_Spline3 + end_Spline4 + end_Spline5 + '
    'end_Spline6 + end_Spline7 + C(Animal) + C(Animal)*Shock_Spline1 + C('
    'Animal)*Shock_Spline2 + C(Animal)*Shock_Spline3 + C(Animal)*Shock_Spline4 + C('
    'Animal)*Shock_Spline5 + C(Animal)*Shock_Spline6 + C(Animal)*Shock_Spline7 + C(Animal)*Start_Spline1 + C('
    'Animal)*Start_Spline2 + C(Animal)*Start_Spline3 + C(Animal)*Start_Spline4 + C('
    'Animal)*Start_Spline5 + C(Animal)*Start_Spline6 + C(Animal)*Start_Spline7 +  C(Animal)*end_Spline1 + C('
    'Animal)*end_Spline2 + C(Animal)*end_Spline3 + C(Animal)*end_Spline4 + C('
    'Animal)*end_Spline5 + C(Animal)*end_Spline6 + C(Animal)*end_Spline7',
    data=design_mat_ext3_noshock, family=sm.families.Gaussian()).fit()
#%%
df_hist, df_ax = plt.subplots(2)
df_ax[0].hist(design_mat_ext3_shock.DF_F, bins=50, density=True)
df_ax[1].hist(design_mat_ext3_noshock.DF_F, bins=50, density=True)
df_ax[0].set_ylabel('Pr. Density')
df_ax[1].set_ylabel('Pr. Density')
df_ax[1].set_xlabel(r'$\frac{dF}{F}$')

qq_plots, qq_ax = plt.subplots(1, 2)
sm.qqplot(design_mat_ext3_noshock.DF_F, scipy.stats.norm(np.mean(design_mat_ext3_noshock.DF_F), np.std(design_mat_ext3_noshock.DF_F)), line='45', ax=qq_ax[0])
sm.qqplot(design_mat_ext3_shock.DF_F, scipy.stats.norm(np.mean(design_mat_ext3_shock.DF_F), np.std(design_mat_ext3_shock.DF_F)), line='45', ax=qq_ax[1])

#%%
a = len(design_mat_ext1)
b = len(design_mat_ext1_null)
rng = np.random.default_rng()
c = np.concatenate((design_mat_ext1.DF_F, design_mat_ext1_null.DF_F))
for i in range(10000):
    boot_mat = np.zeros(10000)
    indices = rng.integers(0, a+b, a+b)
    array_perm = c[indices]
    boot_mat[i] = np.abs(np.mean(array_perm[:a]) - np.mean(array_perm[a:]))

#%%
z = grangercausalitytests(design_mat[['DF_F', 'Isobestic']])