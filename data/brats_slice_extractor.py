#from matplotlib import pyplot as plt

#from iaPACS.helpers import bboxndim
import os
#import shutil
import numpy as np
import nibabel as nib
import pandas
import traceback
import itertools
import traceback

def bboxndim(img):
    """Calcule la bounding box d'un objet

    Parameters
    ----------
    img : np.array
        l'image dont on calcule la bounding box, la valeur du background doit être 0 ou False

    Returns
    -------
    liste[[xmin,xmax],[ymin,ymax],...[zmin,zmax]...]
        tableau 2D [ndim image entrée, 2]
    """
    dims = [np.any(img,axis=i) for i  in itertools.combinations(range(len(img.shape)),r=2)]
    min_max = [np.where(dim)[0][[0, -1]] for dim in dims[:: -1]]
    return min_max

def compute_best_slice(brats_path):
    """Calcule dans une dataframe la slice la plus étendue (en nombre de pixel !=0)

    Parameters
    ----------
    brats_path : str
        Lien vers la racine du dossier brats
    """
    brats_images = list(map(lambda x : os.path.join(brats_path,x),filter(lambda x : os.path.isdir(os.path.join(brats_path, x)), os.listdir(brats_path))))
    df = pandas.DataFrame(columns=['brats_instance','max_slice'])
    for sub_dir in brats_images:
        try :
            path = sub_dir
            images = os.listdir(path)
            seg_name = list(filter(lambda x : 'seg' in x, images))[0]
            seg_path = os.path.join(path, seg_name)
            seg = nib.load(seg_path)
            max_slice = np.argmax(np.sum(seg.get_fdata()!=0,(0,1)))

            df = df.append({'brats_instance':seg_name[0:15],'max_slice':max_slice},ignore_index=True)
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(sub_dir)
    df.to_csv(os.path.join(brats_path,'max_slice.csv'))

def extract_best_slice(brats_path,folder_to_save=None):
    """Sauvegarde la meilleure slice (en nombre de pixel !=0) dans un dossier.
    Si le sous dossier n'est pas précisé par l'utilisateur il sera dans le dossier du patient

    Parameters
    ----------
    brats_path : str
        Lien vers la racine du dossier brats
    folder_to_save : str, optional
        Le dossier dans lequel sauvegarder les slices extraites, by default None (dans le sous dossier du patient)
    """
    if folder_to_save and not os.path.isdir(folder_to_save):
        os.makedirs(folder_to_save)
    brats_images = list(map(lambda x : os.path.join(brats_path,x),filter(lambda x : os.path.isdir(os.path.join(brats_path, x)), os.listdir(brats_path))))
    irm_modality=None
    
    for sub_dir in brats_images:
        try :
            path = sub_dir
            images = os.listdir(path)
            new_dir = os.path.join(folder_to_save,f'{os.path.basename(sub_dir)}_best_slices') if folder_to_save else os.path.join(sub_dir,f'{os.path.basename(sub_dir)}_best_slices')
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
            seg_name = list(filter(lambda x : 'seg' in x, images))[0]
            seg_path = os.path.join(path, seg_name)
            seg = nib.load(seg_path)
            max_slice = np.argmax(np.sum(seg.get_fdata()!=0,(0,1)))

            max_seg = seg.get_fdata()[:,:,max_slice]
            nib.save(nib.Nifti1Image(max_seg, seg.affine, seg.header),os.path.join(new_dir,seg_name))

            for irm_modality in filter(lambda x : not os.path.isdir(x) ,[os.path.join(path,x) for x in filter(lambda x: 'seg' not in x,images)]):
                irm =  nib.load(irm_modality)
                best_slice = irm.get_fdata()[:,:,max_slice]
                nib.save(nib.Nifti1Image(best_slice, irm.affine, irm.header),os.path.join(new_dir,os.path.basename(irm_modality)))
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(sub_dir)
            print(irm_modality)

def extract_only_ROI(brats_path,folder_to_save=None):
    """Sauvegarde la zone dans laquelle la lesion est contenue dans un dossier.
    Si le sous dossier n'est pas précisé par l'utilisateur il sera dans le dossier du patient

    Parameters
    ----------
    brats_path : str
        Lien vers la racine du dossier brats
    folder_to_save : str, optional
        Le dossier dans lequel sauvegarder les volumes extraits, by default None (dans le sous dossier du patient)
    """
    if folder_to_save and not os.path.isdir(folder_to_save):
        os.makedirs(folder_to_save)
    brats_images = list(map(lambda x : os.path.join(brats_path,x),filter(lambda x : os.path.isdir(os.path.join(brats_path, x)), os.listdir(brats_path))))
    for sub_dir in brats_images:
        try :
            images = os.listdir(sub_dir)

            new_dir = os.path.join(folder_to_save,f'{os.path.basename(sub_dir)}') if folder_to_save else os.path.join(sub_dir,f'{os.path.basename(sub_dir)}_min_ROI')
            if  not os.path.isdir(new_dir):
                os.makedirs(new_dir)

            seg_name = list(filter(lambda x : 'seg' in x, images))[0]
            seg_path = os.path.join(sub_dir, seg_name)
            seg = nib.load(seg_path)
            bbox = bboxndim(seg.get_fdata())
            nib.save(nib.Nifti1Image(seg.get_fdata()[bbox[0][0]:bbox[0][1],bbox[1][0]:bbox[1][1],bbox[2][0]:bbox[2][1]], seg.affine, seg.header),os.path.join(new_dir,seg_name))
            for irm_modality in filter(lambda x : not os.path.isdir(x) ,[os.path.join(sub_dir,x) for x in filter(lambda x: 'seg' not in x,images)]):
                irm =  nib.load(irm_modality)
                nib.save(nib.Nifti1Image(irm.get_fdata()[bbox[0][0]:bbox[0][1],bbox[1][0]:bbox[1][1],bbox[2][0]:bbox[2][1]], irm.affine, irm.header),os.path.join(new_dir,os.path.basename(irm_modality)))
        except Exception as e:
            print(e)
            print(sub_dir)
            print(traceback.print_stack())

def extract_same_size_ROI(brats_path,folder_to_save=None,quantile=1,bbox_size=None):
    """Sauvegarde la zone dans laquelle la lesion est contenue dans un dossier. Toutes les zones ont la même taille, qui est la taille maximum qui contiens toutes les ROIs.
    Si le sous dossier n'est pas précisé par l'utilisateur il sera dans le dossier du patient

    Parameters
    ----------
    brats_path : str
        Lien vers la racine du dossier brats
    folder_to_save : str, optional
        Le dossier dans lequel sauvegarder les volumes extraits, by default None (dans le sous dossier du patient)
    quantile : float, optional
        le quantile a utiliser pour la séléction de la taille, 1 par défault, le maximum.
    """
    if folder_to_save and not os.path.isdir(folder_to_save):
        os.makedirs(folder_to_save)
    brats_images = list(map(lambda x : os.path.join(brats_path,x),filter(lambda x : os.path.isdir(os.path.join(brats_path, x)), os.listdir(brats_path))))
    irm_modality = None

    if not bbox_size:
        max_distances=[]
        for sub_dir in brats_images:
            images = os.listdir(sub_dir)
            
            seg_name = list(filter(lambda x : 'seg' in x, images))[0]
            seg_path = os.path.join(sub_dir, seg_name)
            seg = nib.load(seg_path)
            bbox = bboxndim(seg.get_fdata())
            max_distances.append(list(map(lambda x : x[1]-x[0],bbox)))
        
        max_distances = np.quantile(max_distances,q=quantile,axis=0).astype(int)
    else :
        max_distances = bbox_size
    max_distances = [distance if distance%2==0 else distance+1 for distance in max_distances]

    print(f'Taille des bounding box utilisées {max_distances}')
    for sub_dir in brats_images:
        try :
            images = os.listdir(sub_dir)

            new_dir = os.path.join(folder_to_save,f'{os.path.basename(sub_dir)}') if folder_to_save else os.path.join(sub_dir,f'{os.path.basename(sub_dir)}_max_ROI')
            if  not os.path.isdir(new_dir):
                os.mkdir(new_dir)

            seg_name = list(filter(lambda x : 'seg' in x, images))[0]
            seg_path = os.path.join(sub_dir, seg_name)
            seg = nib.load(seg_path)
            bbox = bboxndim(seg.get_fdata())
            shape = seg.get_fdata().shape
            milieux = [np.argmax(np.sum(seg.get_fdata(),axis=ax)) for ax in itertools.combinations(range(3),r=2)][::-1]
            box_recalee = np.array([[max(0, min(milieu - max_distances[i]//2, shape[i] - max_distances[i])),min(shape[i], max(milieu + max_distances[i]//2, 0 + max_distances[i]))] for i,milieu in enumerate(milieux)])
            #print(seg.get_fdata()[box_recalee[0][0]:box_recalee[0][1],box_recalee[1][0]:box_recalee[1][1],box_recalee[2][0]:box_recalee[2][1]].shape)
            nib.save(nib.Nifti1Image(seg.get_fdata()[box_recalee[0][0]:box_recalee[0][1],box_recalee[1][0]:box_recalee[1][1],box_recalee[2][0]:box_recalee[2][1]], seg.affine, seg.header),os.path.join(new_dir,seg_name))
            for irm_modality in filter(lambda x : not os.path.isdir(x) ,[os.path.join(sub_dir,x) for x in filter(lambda x: 'seg' not in x,images)]):
                irm =  nib.load(irm_modality)
                nib.save(nib.Nifti1Image(irm.get_fdata()[box_recalee[0][0]:box_recalee[0][1],box_recalee[1][0]:box_recalee[1][1],box_recalee[2][0]:box_recalee[2][1]], irm.affine, irm.header),os.path.join(new_dir,os.path.basename(irm_modality)))
        except Exception as e:
            print(e)
            print(sub_dir)
            print(traceback.print_stack())


if __name__ == "__main__":

    base_path = r'D:\Data\BRATS2021'
    brats_path = os.path.join(base_path,'RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021')
    brats_path_min_ROI = os.path.join(base_path,'Brats_only_lesion_min_ROI')
    brats_path_max_ROI = os.path.join(base_path,'Brats_only_lesion_max_ROI')
    print('Computing best slice indice and extracting')
    compute_best_slice(brats_path)
    extract_best_slice(brats_path,folder_to_save=None)
    print('Extracting only ROI')
    extract_only_ROI(brats_path,folder_to_save=brats_path_min_ROI)
    print('Extracting only ROI best slice')
    extract_best_slice(brats_path_min_ROI,folder_to_save=None)
    print('Extracting best ROI slice')
    compute_best_slice()
    print('Extracting big ROI')
    extract_same_size_ROI(brats_path,folder_to_save=brats_path_max_ROI, bbox_size=[84, 112, 84])
    print('Extracting best big ROI slice')
    compute_best_slice(brats_path_max_ROI)
    extract_best_slice(brats_path_max_ROI)