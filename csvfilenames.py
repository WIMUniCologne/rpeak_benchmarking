import os

dirname = os.path.dirname(__file__)

# MIT Arrythmia Database
mitar_path = os.path.join(dirname, 'data/csv/MIT_Arrhythmia/')
mitar_all = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112", "113", "114", "115",
                  "116", "117", "118", "119", "121", "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
                  "209", "210", "212", "213", "214", "215", "217", "219", "220", "221", "222", "223", "228", "230", "231",
                  "232", "233", "234"]

# MIT Noise Stress Test
mitnst_path = os.path.join(dirname, 'data/csv/MIT_NSTDB/')
mitnst_all = ['118e_6', '118e00', '118e06', '118e12', '118e18', '118e24', '119e_6', '119e00', '119e06', '119e12', '119e18', '119e24']
mitnst_6 = ['118e_6', '119e_6']
mitnst0 = ['118e00', '119e00']
mitnst6 = ['118e06', '119e06']
mitnst12 = ['118e12', '119e12']
mitnst18 = ['118e18', '119e18']
mitnst24 = ['118e24', '119e24']


# MIT Long Termn
mitlt_path = os.path.join(dirname, 'data/csv/MITLongTerm/')
mitlt_all= ['14046', '14134', '14149', '14157', '14172', '14184', '15814']


# Fantasia Database
fantasia_path = os.path.join(dirname, 'data/csv/Fantasia/')
fantasia_all = ['f1o01', 'f1o02', 'f1o03', 'f1o04', 'f1o05', 'f1o06', 'f1o07', 'f1o08', 'f1o09', 'f1o10',
    'f1y01', 'f1y02', 'f1y03', 'f1y04', 'f1y05', 'f1y06', 'f1y07', 'f1y08', 'f1y09', 'f1y10',
    'f2o01', 'f2o02', 'f2o03', 'f2o04', 'f2o05', 'f2o06', 'f2o07', 'f2o08', 'f2o09', 'f2o10',
    'f2y01', 'f2y02', 'f2y03', 'f2y04', 'f2y05', 'f2y06', 'f2y07', 'f2y08', 'f2y09', 'f2y10']


# PTT Database
ptt_path = os.path.join(dirname, 'data/csv/PulseTransitTime/')
ptt_all= ['s1_sit', 's1_walk', 's1_run', 's2_sit', 's2_walk', 's2_run', 's3_sit', 's3_walk', 's3_run', 's4_sit', 's4_walk',
           's4_run', 's5_sit', 's5_walk', 's5_run', 's6_sit', 's6_walk', 's6_run', 's7_sit', 's7_walk', 's7_run', 's8_sit',
           's8_walk', 's8_run', 's9_sit', 's9_walk', 's9_run', 's10_sit', 's10_walk', 's10_run', 's11_sit', 's11_walk',
           's11_run', 's12_sit', 's12_walk', 's12_run', 's13_sit', 's13_walk', 's13_run', 's14_sit', 's14_walk', 's14_run',
           's15_sit', 's15_walk', 's15_run', 's16_sit', 's16_walk', 's16_run', 's17_sit', 's17_walk', 's17_run', 's18_sit',
           's18_walk', 's18_run', 's19_sit', 's19_walk', 's19_run', 's20_sit', 's20_walk', 's20_run', 's21_sit', 's21_walk',
           's21_run', 's22_sit', 's22_walk', 's22_run']
ptt_sit = ['s1_sit', 's2_sit', 's3_sit', 's4_sit', 's5_sit', 's6_sit', 's7_sit', 's8_sit', 's9_sit', 's10_sit', 's11_sit',
           's12_sit', 's13_sit', 's14_sit', 's15_sit', 's16_sit', 's17_sit', 's18_sit', 's19_sit', 's20_sit', 's21_sit',
           's22_sit']
ptt_walk = ['s1_walk', 's2_walk', 's3_walk', 's4_walk', 's5_walk', 's6_walk', 's7_walk', 's8_walk', 's9_walk', 's10_walk',
            's11_walk', 's12_walk', 's13_walk', 's14_walk', 's15_walk', 's16_walk', 's17_walk', 's18_walk', 's19_walk',
            's20_walk', 's21_walk', 's22_walk']
ptt_run = ['s1_run', 's2_run', 's3_run', 's4_run', 's5_run', 's6_run', 's7_run', 's8_run', 's9_run', 's10_run', 's11_run',
           's12_run', 's13_run', 's14_run', 's15_run', 's16_run', 's17_run', 's18_run', 's19_run', 's20_run', 's21_run', 's22_run']

