import os
import glob
import torch
import numpy as np
from PIL import Image
import tifffile as tiff
import SimpleITK as itk
from torch.utils.data import Dataset
from ddpmMed.utils.data import torch2np, normalize


class BRATS(Dataset):
    """
    A PyTorch Dataset utilized for Brain Tumor Segmentation Dataset
    """

    def __init__(self, path: str, validation: bool = False) -> None:
        """
        Initializes the dataset using a path to all patient folders
        :param path (str): directory containing all patient scans
        """
        self.path = path
        self.val = validation
        self.dataset = dict()

        # check for cuda
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Make sure path exists
        if not os.path.exists(self.path):
            raise RuntimeError("Given path ({}) does not exist.".format(self.path))

        # Path should be a directory to all MRI cases
        if not os.path.isdir(self.path):
            raise NotADirectoryError("Given path ({}) is not a directory".format(self.path))

        # List all sub folders within path, exclude files
        _folders = os.listdir(self.path)
        _folders = [os.path.join(self.path, item) for item in _folders]
        _folders = [item for item in _folders if os.path.isdir(item)]
        _total_items = len(_folders)

        for index, folder in enumerate(_folders):
            folder_name = os.path.basename(folder)

            # Ensure every folder has 5 items/files
            if len(os.listdir(folder)) != 5 and not self.val:
                raise FileNotFoundError("One or more missing files in folder: {}.\n"
                                        "Check contents {}".format(folder_name, os.listdir(folder)))

            # Get path to all MRI modalities and segmentation
            t1 = glob.glob(os.path.join(folder, "{}_t1.*".format(folder_name)))[0]
            t1ce = glob.glob(os.path.join(folder, "{}_t1ce*".format(folder_name)))[0]
            t2 = glob.glob(os.path.join(folder, "{}_t2*".format(folder_name)))[0]
            flair = glob.glob(os.path.join(folder, "{}_flair*".format(folder_name)))[0]
            if not self.val:
                label = glob.glob(os.path.join(folder, "{}_seg*".format(folder_name)))[0]

            # TODO: make sure all files are for the same patient

            # Save path for every MRI modality to dataset
            if self.val:
                self.dataset[index] = {
                    '3d': {
                        "t1": t1,
                        "t1ce": t1ce,
                        "t2": t2,
                        "flair": flair,
                    }
                }
            else:
                self.dataset[index] = {
                    '3d': {
                        "t1": t1,
                        "t1ce": t1ce,
                        "t2": t2,
                        "flair": flair,
                        "label": label
                    }
                }
            print("\rReading item(s) [{}/{}]".format(index, _total_items - 1), end="", flush=False)
        print("\nFinished Fetching all dataset items")

    def __len__(self) -> int:
        return len(self.dataset.keys())

    def __getitem__(self, index) -> dict:
        """ Reads and returns data in 3D """

        data = self.dataset[index]['3d']

        try:
            # MRI modalities
            t1 = itk.ReadImage(data['t1'])
            t1ce = itk.ReadImage(data['t1ce'])
            t2 = itk.ReadImage(data['t2'])
            flair = itk.ReadImage(data['flair'])
            label = None
            if not self.val:
                label = itk.ReadImage(data['label'])

            # TODO: Torch can not cast uint16 to tensors, check labels after casting!
            if self.val:
                data = {
                    't1': torch.tensor(itk.GetArrayFromImage(t1), dtype=torch.float32).to(self.device),
                    't1ce': torch.tensor(itk.GetArrayFromImage(t1ce), dtype=torch.float32).to(self.device),
                    't2': torch.tensor(itk.GetArrayFromImage(t2), dtype=torch.float32).to(self.device),
                    'flair': torch.tensor(itk.GetArrayFromImage(flair), dtype=torch.float32).to(self.device),
                }

                data = {
                    'volume': torch.cat([
                        data['t1'].unsqueeze(0),
                        data['t1ce'].unsqueeze(0),
                        data['t2'].unsqueeze(0),
                        data['flair'].unsqueeze(0)
                    ])
                }
            else:
                data = {
                    't1': torch.tensor(itk.GetArrayFromImage(t1), dtype=torch.float32).to(self.device),
                    't1ce': torch.tensor(itk.GetArrayFromImage(t1ce), dtype=torch.float32).to(self.device),
                    't2': torch.tensor(itk.GetArrayFromImage(t2), dtype=torch.float32).to(self.device),
                    'flair': torch.tensor(itk.GetArrayFromImage(flair), dtype=torch.float32).to(self.device),
                    'label': torch.tensor(itk.GetArrayFromImage(label).astype(dtype=np.int32),
                                          dtype=torch.int32).to(self.device)
                }
                data = {
                    'volume': torch.cat([
                        data['t1'].unsqueeze(0),
                        data['t1ce'].unsqueeze(0),
                        data['t2'].unsqueeze(0),
                        data['flair'].unsqueeze(0)
                    ]),
                    'label': data['label'].unsqueeze(0)
                }

        except Exception as ex:
            raise RuntimeError("unable to read data at index: {} in dataset.\nError:{}".format(index, ex))

        return data

    def export_2d(self,
                  output_folder: str,
                  bids_format: bool = True,
                  separate_labels: bool = False,
                  as_tensors: bool = False) -> None:
        """
        Exports 2D slices from each patient data in png format,
        and updates the dataset with new 2d slices.
        :param as_tensors:
        :param output_folder: Directory to where save the exported slices
        :param bids_format: Boolean to either export in BIDS format (Separate folders for each patient)
        :param separate_labels: A boolean to export the labels into a separate folder
               useful when using DDPM to learn only the MRI scan modalities for example.
        :return: None
        """
        # Create output folder if does not exist
        output_folder = os.path.join(output_folder, 'Exported 2D BRATS Data')
        labels_folder = output_folder
        if separate_labels:
            labels_folder = os.path.join(output_folder, 'masks')
        if not bids_format:
            output_folder = os.path.join(output_folder, 'scans')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            if separate_labels:
                os.makedirs(labels_folder)

        total_patients = len(self.dataset)
        for index in range(0, total_patients):

            # Get patient folder name
            data = self.dataset[index]
            folder = os.path.dirname(data['3d']['t1'])
            folder = os.path.basename(folder)
            name = folder

            if bids_format:
                folder = os.path.join(output_folder, folder)
            else:
                folder = output_folder

            # Create directory name
            if not os.path.exists(folder):
                os.makedirs(folder)

            # Generate slices on almost center of image
            slices = [70, 80, 90, 100, 110]
            for key in data['3d'].keys():

                for i in slices:
                    if key == 'label' and separate_labels:
                        if as_tensors:
                            image_path = os.path.join(labels_folder, '{}_{}_{}.pt'.format(name, key, i))
                        else:
                            image_path = os.path.join(labels_folder, '{}_{}_{}.png'.format(name, key, i))
                    else:
                        if as_tensors:
                            image_path = os.path.join(folder, '{}_{}_{}.pt'.format(name, key, i))
                        else:
                            image_path = os.path.join(folder, '{}_{}_{}.png'.format(name, key, i))
                    self.dataset[index]['2d'][key] = image_path
                    image = (itk.GetArrayFromImage(itk.ReadImage(data['3d'][key])))
                    if as_tensors:
                        torch.save(obj=torch.tensor(image),
                                   f=image_path)
                    else:
                        image = Image.fromarray(image[i, :, :])
                        image.convert('L')
                        image.save(image_path)
            print("\rExporting scan [{}/{}]".format((index + 1), total_patients), end='', flush=True)

    def export_stack(self,
                     output_path: str,
                     as_tensors: bool = False):
        """
        Exports sectional stacks from BRATS dataset scans and masks.
        :param as_tensors:
        :param output_path: a string path to output directory
        :return: sectional stacks of scans and annotated mask saved to output_path
        """

        # define output folders
        output_folder = os.path.join(output_path, 'Stacked 2D BRATS Data', 'scans')
        if not self.val:
            labels_folder = os.path.join(output_path, 'Stacked 2D BRATS Data', 'masks')
            if not os.path.exists(labels_folder):
                os.makedirs(labels_folder)

        # create directories even if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Generate stacked 2d images across every slice index
        slices = [70, 75, 80, 85, 90, 95, 100]
        total = len(self.dataset)

        # extension based on either tensor or not
        _ext = "pt" if as_tensors else "tiff"
        for index in range(0, total):

            # Get different MRI scans
            data = self.__getitem__(index)
            name = os.path.basename(self.dataset[index]['3d']['t1']).split('_t1')[0]
            t1 = data['t1']
            t1ce = data['t1ce']
            t2 = data['t2']
            flair = data['flair']
            if not self.val:
                label = data['label']

            # Export stacked images per slices
            for i in slices:

                # stack slices
                stacked = torch.stack([t1[i, :, :],
                                       t1ce[i, :, :],
                                       t2[i, :, :],
                                       flair[i, :, :]])

                # Save images/masks in tiff with original format
                out_file = os.path.join(output_folder, "{}_{}.{}".format(name, i, _ext))
                if not self.val:
                    label_file = os.path.join(labels_folder, "{}_{}.{}".format(name, i, _ext))
                if as_tensors:
                    torch.save(obj=stacked, f=out_file)
                    if not self.val:
                        torch.save(obj=label, f=label_file)
                else:
                    tiff.imsave(file=out_file,
                                data=normalize(torch2np(stacked)))
                    if not self.val:
                        tiff.imsave(file=label_file,
                                    data=torch2np(label[i, :, :]))
            print("\rExporting scan [{}/{}]".format((index + 1), total), end='', flush=True)

