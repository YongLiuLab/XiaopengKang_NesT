import math
import os

from multiprocessing import Process
import nibabel as nib
from nilearn.image import resample_to_img

from datamanagement import datasets

temp_mni_path = r'/user/home/xpkang/data/atlas/ch2bet_2mm.nii'

def resample_to(subjects, template, in_name='t1.nii', out_name='t1_2mm.nii'):
    i = 1
    n = len(subjects)
    for subject in subjects:
        obs = subject.get_observation('baseline')
        t1 = obs['t1']
        nii_path = t1.build_path(in_name)
        out_path = t1.build_path(out_name)
        if os.path.exists(out_path):
            continue
        nii = nib.load(nii_path)
        new_nii = resample_to_img(nii, template)
        nib.save(new_nii, out_path)

        print('{}/{}:{}'.format(i, n, subject.fullpath))
        i += 1


if __name__ == '__main__':
    centers1 = datasets.load_dataset('/user/home/xpkang/data/AD/ADNI')
    centers2 = datasets.load_dataset('/user/home/xpkang/data/AD/EDSD')
    centers3 = datasets.load_dataset('/user/home/xpkang/data/AD/MCAD')
    centers = centers1 + centers2 + centers3
    subjects = []
    for center in centers:
        for subject in center.subjects:
            subjects.append(subject)

    process = 8
    n = len(subjects)
    print(n)
    n_each = math.ceil(n/process)

    chunks = [subjects[x:x+n_each] for x in range(0, len(subjects), n_each)]
    template = nib.load(temp_mni_path)
    for i in range(process):
        p = Process(target=resample_to, args=(chunks[i],template,))
        p.start()
    print(len(subjects))