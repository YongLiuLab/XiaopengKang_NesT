import math
import os

from multiprocessing import Process

from datamanagement import datasets

aladin_path = r'E:/data/utils/reg_aladin.exe'
temp_mni_path = r'../atlas/MNI152NLin6_res-4x4x4_T1w_descr-brainmask.nii'

def reg_to(subjects, in_name='t1.nii', out_name='rt1_4mm.nii'):
    i = 1
    n = len(subjects)
    for subject in subjects:
        obs = subject.get_observation('baseline')
        t1 = obs.t1
        nii_path = t1.build_path(in_name)
        out_path = t1.build_path(out_name)
        aff_path = t1.build_path('rt1_aff.txt')
        print('{}/{}:{}'.format(i, n, subject.fullpath))
        i += 1

        if os.path.exists(out_path) and os.path.exists(aff_path):
            continue

        cmd = '{} -ref {} -flo {} -res {} -aff {} -voff'.format(aladin_path,
                                                        temp_mni_path,
                                                        nii_path,
                                                        out_path,
                                                        aff_path)
        os.system(cmd)

if __name__ == '__main__':
    centers_adni = datasets.load_dataset('../AD/ADNI')
    centers_mcad = datasets.load_dataset('../AD/MCAD')
    centers = datasets.load_dataset('../AD/EDSD') + centers_adni + centers_mcad

    subjects = []
    for center in centers:
        for k, subject in center.subjects.items():
            subjects.append(subject)

    process = 8
    n = len(subjects)
    print(n)
    n_each = math.ceil(n/process)

    chunks = [subjects[x:x+n_each] for x in range(0, len(subjects), n_each)]
    for i in range(process):
        p = Process(target=reg_to, args=(chunks[i],))
        p.start()
    print(len(subjects))