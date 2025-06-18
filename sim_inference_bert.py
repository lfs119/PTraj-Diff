import argparse

from diff import modelling

parser = argparse.ArgumentParser()
parser.add_argument('--sim_ckpt', type=str, default=None, required=True)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--num_frames', type=int, default=1000)
parser.add_argument('--num_rollouts', type=int, default=100)
parser.add_argument('--af3', action='store_true') # base_path_cif
parser.add_argument('--base_path_cif', type=str, default="./af_output/ppi1")
parser.add_argument('--no_frames', action='store_true')
parser.add_argument('--tps', action='store_true')
parser.add_argument('--xtc', action='store_true')
parser.add_argument('--out_dir', type=str, default=".")
parser.add_argument('--split', type=str, default='splits/4AA_test.csv')
# diffusion group
parser.add_argument("--diffusion", action='store_true')
parser.add_argument("--num_timesteps", type=int, default=1000)
parser.add_argument("--beta_schedule", type=str, default="polynomial_2", choices=["linear", "quadratic", "polynomial_n", "cosine"])
parser.add_argument("--noise_precision", type=float, default=0.00001)
args = parser.parse_args()

import os, torch, mdtraj, tqdm, time
import numpy as np
from diff.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from diff.residue_constants import restype_order, restype_atom37_mask, restype_order_with_x
from diff.tensor_utils import tensor_tree_map
from diff.utils import atom14_to_pdb
import pandas as pd
from read_cif import find_cif_files_and_parent_dirs, structure_to_atom14
from Bio.PDB import MMCIFParser




os.makedirs(args.out_dir, exist_ok=True)



def get_batch(name, seqres, arr=None):
    if arr is None:
        arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}.npy', 'r') 

    if not args.tps: # else keep all frames
        arr = np.copy(arr[0:1]).astype(np.float32)  #（1， 4， 14， 3）

    frames = atom14_to_frames(torch.from_numpy(arr))
    seqres = torch.tensor([restype_order_with_x[c] for c in seqres])
    # seqres = torch.tensor([restype_order[c] for c in seqres])
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres[None])).float()
    L = len(seqres)
    mask = torch.ones(L)
    
    if args.no_frames:
        return {
            'atom37': atom37,
            'seqres': seqres,
            'mask': restype_atom37_mask[seqres],
        }
        
    torsions, torsion_mask = atom37_to_torsions(atom37, seqres[None])#    (1, 4, 7, 2)
    return {
        'torsions': torsions,
        'torsion_mask': torsion_mask[0],
        'trans': frames._trans,   # (1, 4, 3)
        'rots': frames._rots._rot_mats, # (1, 4, 3, 3)
        'seqres': seqres,
        'mask': mask, # (L,)
    }

def rollout(model, batch):

    #print('Start sim', batch['trans'][0,0,0])
    if args.no_frames:
        
        expanded_batch = {
            'atom37': batch['atom37'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
        }
    else:    
        expanded_batch = {
            'torsions': batch['torsions'].expand(-1, args.num_frames, -1, -1, -1),
            'torsion_mask': batch['torsion_mask'],
            'trans': batch['trans'].expand(-1, args.num_frames, -1, -1),
            'rots': batch['rots'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
        }
    atom14, _ = model.inference(expanded_batch)
    new_batch = {**batch}

    if args.no_frames:
        new_batch['atom37'] = torch.from_numpy(
            atom14_to_atom37(atom14[:,-1].cpu(), batch['seqres'][0].cpu())
        ).cuda()[:,None].float()
        
        
        
    else:
        frames = atom14_to_frames(atom14[:,-1])
        new_batch['trans'] = frames._trans[None]
        new_batch['rots'] = frames._rots._rot_mats[None]
        atom37 = atom14_to_atom37(atom14[0,-1].cpu(), batch['seqres'][0].cpu())
        torsions, _ = atom37_to_torsions(atom37, batch['seqres'][0].cpu())
        new_batch['torsions'] = torsions[None, None].cuda()

    return atom14, new_batch
    
    
def do(model, name, seqres):
    if args.af3:
        cif_files = find_cif_files_and_parent_dirs(args.base_path_cif)
        print(len(cif_files))
        for cifitem in cif_files:
            mmcifparser = MMCIFParser()
            structure = mmcifparser.get_structure('example', cifitem['file_path'])
            arr = structure_to_atom14(structure, model.args.crop)
            item = get_batch(name, seqres, arr)
            batch = next(iter(torch.utils.data.DataLoader([item])))
            batch = tensor_tree_map(lambda x: x.cuda(), batch)  
            
            all_atom14 = []
            start = time.time()
            for _ in tqdm.trange(args.num_rollouts):
                atom14, batch = rollout(model, batch)
                # print(atom14[0,0,0,1], atom14[0,-1,0,1])
                all_atom14.append(atom14)

            print(time.time() - start)
            all_atom14 = torch.cat(all_atom14, 1)
           
            
            path = os.path.join(args.out_dir, cifitem['parent_dir']+f'{name}.pdb')
            atom14_to_pdb(all_atom14[0].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)

            if args.xtc:
                traj = mdtraj.load(path)
                traj.superpose(traj)
                traj.save(os.path.join(args.out_dir, cifitem['parent_dir']+f'{name}.xtc'))
                traj[0].save(os.path.join(args.out_dir, cifitem['parent_dir']+f'{name}.pdb'))
    else:
        item = get_batch(name, seqres)
        batch = next(iter(torch.utils.data.DataLoader([item])))

        batch = tensor_tree_map(lambda x: x.cuda(), batch)  
        
        all_atom14 = []
        start = time.time()
        for _ in tqdm.trange(args.num_rollouts):
            atom14, batch = rollout(model, batch)
            # print(atom14[0,0,0,1], atom14[0,-1,0,1])
            all_atom14.append(atom14)

        print(time.time() - start)
        all_atom14 = torch.cat(all_atom14, 1)
    
        
        path = os.path.join(args.out_dir, f'{name}.pdb')
        atom14_to_pdb(all_atom14[0].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)

        if args.xtc:
            traj = mdtraj.load(path)
            traj.superpose(traj)
            traj.save(os.path.join(args.out_dir, f'{name}.xtc'))
            traj[0].save(os.path.join(args.out_dir, f'{name}.pdb'))

   

@torch.no_grad()
def main():
    model = modelling.BertForDiffusionBase.load_from_checkpoint(args.sim_ckpt)
    model.eval().to('cuda')
    add_ckpt = os.path.splitext(os.path.basename(args.sim_ckpt))[0]
    args.out_dir = os.path.join(args.out_dir, add_ckpt)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    
    df = pd.read_csv(args.split, index_col='name')
    for name in df.index:
        if args.pdb_id and name not in args.pdb_id:
            continue
        do(model, name, df.seqres[name])
        

main()

