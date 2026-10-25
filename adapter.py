"""Official ProtoMotions unified-pipeline ONNX integration, G1 BONES deploy."""
from motion_challenge.adapters.onnx import cpu_session
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from motion_challenge.sdk import ControlCommand
from motion_challenge.validation import sha256_file


class ProtoMotions:
    def __init__(self,root,config):
        self.cfg=yaml.safe_load((root/'policy.yaml').read_text())
        weights=(root/config['weights']).resolve()
        if sha256_file(weights)!=config['weights_sha256']:
            raise ValueError('ProtoMotions checkpoint hash mismatch')
        if sha256_file(root/'policy.yaml')!=config['inference_config_sha256']:
            raise ValueError('ProtoMotions config hash mismatch')
        self.session=cpu_session(weights,config['weights_sha256'])
        if self.cfg['control']['pd_target_max_accel'] is not None or self.cfg['control']['action_ema_alpha']!=1.0:
            raise ValueError('this checkpoint adapter requires explicit unfiltered deployment configuration')
        self.last=np.zeros((1,1,29),dtype=np.float32)

    def reset(self,state,reference,seed):
        self.indices=[state.joint_names.index(n) for n in self.cfg['joint_names']]
        self.inverse=[self.cfg['joint_names'].index(n) for n in state.joint_names]
        self.last.fill(0)

    def act(self,state,reference):
        anchor=next(b for b in state.bodies if b.name=='torso_link')
        future=[next(b for b in f.bodies if b.name=='torso_link').orientation_xyzw for f in reference]
        q=np.asarray(state.joint_position_rad)[self.indices]
        qd=np.asarray(state.joint_velocity_rad_s)[self.indices]
        inputs={
            'current_anchor_rot':np.asarray(anchor.orientation_xyzw)[None],
            'current_dof_pos':q[None], 'current_dof_vel':qd[None],
            'current_root_local_ang_vel':Rotation.from_quat(state.root_orientation_xyzw).inv().apply(state.root_angular_velocity_world_rad_s)[None],
            'historical_processed_actions':self.last,
            'mimic_future_anchor_rot':np.asarray(future)[None],
            'mimic_future_dof_pos':np.asarray([f.joint_position_rad for f in reference])[None][:,:,self.indices],
            'mimic_future_dof_vel':np.asarray([f.joint_velocity_rad_s for f in reference])[None][:,:,self.indices],
        }
        outputs=self.session.run(None,{n:x.astype(np.float32) for n,x in inputs.items()})
        values={n.name:x for n,x in zip(self.session.get_outputs(),outputs)}
        self.last[:]=values['joint_pos_targets'].reshape(1,1,29)
        def ordered(name):return tuple(float(x) for x in values[name].reshape(29)[self.inverse])
        return ControlCommand(ordered('joint_pos_targets'),(0.,)*29,
                              ordered('stiffness_targets'),ordered('damping_targets'),(0.,)*29)

    def close(self):self.session=None


def create(root,config):return ProtoMotions(root,config)
