"""ProtoMotions unified deployment pipeline with full GPU batches."""
import json
from pathlib import Path
import torch
import yaml
from torch_graph import TensorGraph
from motion_challenge.simulation.tensor_math import normalize, multiply, conjugate, rotate
from motion_challenge.validation import sha256_file


class TensorProtoMotions:
    def __init__(self,root,runtime,joint_names,slots,device,converted_weights):
        root=Path(root)
        if sha256_file(root/'policy.yaml') != runtime['inference_config_sha256']:
            raise ValueError('ProtoMotions inference configuration changed')
        self.cfg=yaml.safe_load((root/'policy.yaml').read_text())
        if self.cfg['control']['pd_target_max_accel'] is not None or self.cfg['control']['action_ema_alpha'] != 1.:
            raise ValueError('This checkpoint specifies unfiltered joint targets')
        self.device=torch.device(device);self.slots=slots;self.names=self.cfg['joint_names']
        self.order=[joint_names.index(n) for n in self.names]
        self.inverse=[self.names.index(n) for n in joint_names]
        self.waist=[self.names.index('waist_'+axis+'_joint') for axis in ('yaw','roll','pitch')]
        self.last=torch.zeros((slots,1,29),device=device)
        self.model=TensorGraph(converted_weights,device,runtime['weights_sha256']).eval()

    def reset(self,slots):
        self.last[slots]=0

    def torso(self,root_orientation,joints):
        rotation=normalize(root_orientation.double())
        for axis,joint in zip((2,0,1),self.waist):
            half=.5*joints[...,joint].double()
            local=torch.zeros((*half.shape,4),device=self.device,dtype=torch.float64)
            local[...,axis]=half.sin();local[...,3]=half.cos()
            rotation=multiply(rotation,local)
        return normalize(rotation).float()

    def observations(self,state,reference):
        joints=state.joint_position_rad[:,self.order]
        future=reference.joint_position_rad[:,:,self.order]
        return {
            'current_anchor_rot':self.torso(state.root_orientation_xyzw,joints),
            'current_dof_pos':joints.float(),
            'current_dof_vel':state.joint_velocity_rad_s[:,self.order].float(),
            'current_root_local_ang_vel':rotate(conjugate(normalize(state.root_orientation_xyzw.double())),state.root_angular_velocity_world_rad_s.double()).float(),
            'historical_processed_actions':self.last,
            'mimic_future_anchor_rot':self.torso(reference.root_orientation_xyzw,future),
            'mimic_future_dof_pos':future.float(),
            'mimic_future_dof_vel':reference.joint_velocity_rad_s[:,:,self.order].float(),
        }

    @torch.inference_mode()
    def act(self,state,reference):
        output=self.model(**self.observations(state,reference))
        self.last.copy_(output['joint_pos_targets'][:,None])
        target=output['joint_pos_targets'][:,self.inverse]
        zero=torch.zeros_like(target)
        return torch.stack((target,zero,output['stiffness_targets'][:,self.inverse],output['damping_targets'][:,self.inverse],zero),dim=1)


ADAPTER_CLASS='TensorProtoMotions'
NATIVE_CLASS='ProtoMotions'
