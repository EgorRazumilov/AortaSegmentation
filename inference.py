import os
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from monai.inferers import SlidingWindowInferer
from collections import OrderedDict
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 15)

import vtk
import nrrd
import os
from tqdm import tqdm
import argparse

def replace_path_values(cfg):
    for key, value in cfg.items():
        if isinstance(value, str):
            cfg[key] = cfg[key].replace("${hydra:runtime.cwd}", os.getcwd())
        if isinstance(value, dict):
            cfg[key] = replace_path_values(cfg[key])
    return cfg


def inference_2d_as_3d(model, dataset_2D, patient_id=0, batch_size=16):
    pred_mask = []
    test_mask = []
    start_ind = sum([dataset_2D.images_depths[i] for i in range(patient_id)])
    with torch.no_grad():
        batch = []
        for i in tqdm(range(dataset_2D.images_depths[patient_id])):
            index = start_ind + i
            test_mask.append(dataset_2D[index][1])
            batch.append(dataset_2D[index][0])
            if len(batch) == batch_size or i == dataset_2D.images_depths[patient_id] - 1:
                pred = model(torch.stack(batch).to('cuda:1'))
                pred_mask.append(pred)
                batch = []
    return torch.cat(pred_mask).cpu(), torch.stack(test_mask)

def build_3d_model(mask, out_name, do_smooth_mesh):
    filename = out_name + '.nrrd'
    nrrd.write(filename, mask.cpu().numpy())

    reader = vtk.vtkNrrdReader()
    reader.SetFileName(filename)
    reader.Update()

    threshold = vtk.vtkImageThreshold()
    threshold.SetInputConnection(reader.GetOutputPort())
    threshold.ThresholdByLower(0.1)
    threshold.ReplaceInOn()
    threshold.SetInValue(0) 
    threshold.ReplaceOutOn()
    threshold.SetOutValue(1)
    threshold.Update()

    dmc = vtk.vtkFlyingEdges3D()
    dmc.SetInputConnection(threshold.GetOutputPort())
    dmc.ComputeNormalsOn()
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()

    if do_smooth_mesh:
        smoothing_iterations = 15
        pass_band = 0.001
        feature_angle = 120.0

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(dmc.GetOutputPort())
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(feature_angle)
        smoother.SetPassBand(pass_band)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

    writer = vtk.vtkOBJWriter()
    if do_smooth_mesh:
        writer.SetInputConnection(smoother.GetOutputPort())
    else:
        writer.SetInputConnection(dmc.GetOutputPort())
    writer.SetFileName(out_name + 'obj')
    writer.Write()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hydra_checkpoint_path', type=str, required=True,
                        help='Path to Hydra created checkpoint (in "outputs" folder by default)')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out_obj_file_name', type=str, required=True)
    parser.add_argument('--do_mesh_smooth', action='store_true')

    args = parser.parse_args()

    with initialize(version_base=None, config_path=os.path.join(args.hydra_checkpoint_path, ".hydra")):
        cfg = compose(config_name="config")
    
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict = replace_path_values(cfg_dict)
        cfg = OmegaConf.create(cfg_dict)
    
    model = instantiate(cfg.model)
    dataset = instantiate(cfg.test_dataset)
        
    checkpoint = torch.load(os.path.join(args.hydra_checkpoint_path, "checkpoints/last.ckpt"), map_location='cpu')
    
    model_state_dict = OrderedDict()
    for v, k in checkpoint['state_dict'].items():
        if v.startswith('model.'):
            model_state_dict[v[6:]] = k
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(args.device)

    if '3D' in cfg.test_dataset['_target_']:
        inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=1, overlap=0.7)
        inp, _ = dataset[0]
        inp = inp[None]
        out = inferer(model, inp)

        out = (F.sigmoid(out) > 0.5).float()[0, 0]
    else:
        out, _ = inference_2d_as_3d(model, dataset)
        out = (F.sigmoid(out) > 0.5).float()
        out = out.permute(1, 2, 3, 0)[0]

    build_3d_model(out, args.out_obj_file_name, args.do_mesh_smooth)


if __name__ == '__main__':
    main()
    
