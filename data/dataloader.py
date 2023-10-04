import torch
import torchio as tio
import copy
import os

def load_subjects(centers, img_pathes=['mri/mwp1t1_2mm.nii'], classes=2):
    subjects = []
    for center in centers:
        for subject in center.subjects:
            nii_pathes = [subject['baseline']['t1'].build_path(path) for path in img_pathes]
            for nii_path in nii_pathes:
                if not os.path.exists(nii_path):
                    print(f'No file for {center.name}:{subject.name}')

    for center in centers:
        for subject in center.subjects:
            nii_pathes = [subject['baseline']['t1'].build_path(path) for path in img_pathes]
            img_count = len(nii_pathes)
            label = subject.get_label()

            if classes==2:
                # if only perform AD/NC classification, pass MCI, set AD to 1
                if label == 1:
                    continue
                if label == 2:
                    label = 1

            # Must contain at least one image in __init__
            tio_subject = tio.Subject(
                img0=tio.ScalarImage(nii_pathes[0]),
                img_count=img_count,
                label=label,
                path=subject.fullpath,
                center=center.name,
                name=subject.name
            )

            for i in range(1, img_count):
                tio_subject.add_image(tio.ScalarImage(nii_pathes[i]), f'img{i}')

            subjects.append(tio_subject)
    return subjects

class ArguDataset(tio.SubjectsDataset):
    def __init__(self, subjects, 
                 transform=None,
                 load_getitem=True):
        super().__init__(subjects, transform, load_getitem)
    
    def __getitem__(self, index):
        subject = copy.deepcopy(self._subjects[index])
        if self.load_getitem:
            subject.load()
        if self._transform is not None:
            subject = self._transform(subject)

        datas = []
        for i in range(subject['img_count']):
            datas.append(subject[f'img{i}'][tio.DATA].float())
        datas = torch.cat(datas)
        label = subject['label']

        center = subject['center']
        name = subject['name']
        subject_name = f'{center}_{name}'
        return datas, label, subject_name