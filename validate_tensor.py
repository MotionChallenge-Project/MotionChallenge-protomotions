"""Compare strict tensor inputs and controls with the original deploy adapter."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import types

import mujoco
import numpy as np
from onnx.reference import ReferenceEvaluator
import torch
import yaml
from motion_challenge.sdk import BodyState, DeployableStateBatch, MotionReferenceBatch, ReferenceFrame, StatePacket
from motion_challenge.validation import sha256_file


class Session:
    def __init__(self,oracle):self.oracle=oracle
    def run(self,names,feed):
        self.feed={k:v.copy() for k,v in feed.items()}
        return self.oracle.run(names,feed)
    def get_outputs(self):
        return [types.SimpleNamespace(name=n) for n in ('actions','joint_pos_targets','stiffness_targets','damping_targets')]


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--converted',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime=json.loads((root/'runtime.mjwarp.json').read_text())
    cfg=yaml.safe_load((root/'policy.yaml').read_text())
    if 'onnxruntime' not in sys.modules:sys.modules['onnxruntime']=types.ModuleType('onnxruntime')
    from adapter import ProtoMotions
    from tensor_adapter import TensorProtoMotions
    names=tuple(reversed(cfg['joint_names']));slots=7
    batched=TensorProtoMotions(root,runtime,names,slots,args.device,args.converted)
    original=[];oracle=ReferenceEvaluator(str(args.weights))
    for _ in range(slots):
        a=ProtoMotions.__new__(ProtoMotions);a.cfg=cfg;a.last=np.zeros((1,1,29),np.float32);a.session=Session(oracle);original.append(a)
    model=mujoco.MjModel.from_xml_path(str(root/'assets/g1_reference.xml'));data=mujoco.MjData(model)
    qids=[int(model.joint(n).qposadr[0]) for n in names];anchor=model.body('torso_link').id
    def body(q,pos,quat):
        data.qpos[:]=model.qpos0;data.qpos[qids]=q;data.qpos[:3]=pos;data.qpos[3:7]=quat[[3,0,1,2]]
        mujoco.mj_kinematics(model,data)
        return BodyState('torso_link',tuple(data.xpos[anchor]),tuple(data.xquat[anchor,[1,2,3,0]]),(0.,)*3,(0.,)*3)
    rng=np.random.default_rng(19855);max_input=0.;max_command=0.
    batched.reset(torch.arange(slots,device=args.device))
    for step in range(24):
        q=rng.normal(0,.2,(slots,29));qd=rng.normal(0,.2,(slots,29));quat=rng.normal(size=(slots,4));quat/=np.linalg.norm(quat,axis=-1,keepdims=True);omega=rng.normal(0,.3,(slots,3))
        offsets=np.asarray(runtime['reference_offsets_s']);k=len(offsets);requested=step*.02+np.broadcast_to(offsets,(slots,k));times=requested.clip(0,.37)
        rj=rng.normal(0,.2,(slots,k,29));rp=rng.normal(0,.2,(slots,k,3));rp[...,2]+=.75;rq=rng.normal(size=(slots,k,4));rq/=np.linalg.norm(rq,axis=-1,keepdims=True)
        jv=rng.normal(0,.2,(slots,k,29));lv=rng.normal(0,.2,(slots,k,3));av=rng.normal(0,.2,(slots,k,3))
        def tensor(x):return torch.tensor(np.array(x),device=args.device)
        state=DeployableStateBatch(names,tensor(np.full(slots,step*.02)),*[tensor(x) for x in (q,qd,quat,omega)])
        refs=MotionReferenceBatch(*[tensor(x) for x in (requested,times,rj,rp,rq,jv,lv,av)])
        if step==11:batched.reset(torch.tensor([1,4],device=args.device))
        feed={k:v.clone() for k,v in batched.observations(state,refs).items()};actual=batched.act(state,refs).cpu().numpy()
        for row,a in enumerate(original):
            packet=StatePacket(step*.02,names,tuple(q[row]),tuple(qd[row]),(10.,20.,30.),tuple(quat[row]),(9.,8.,7.),tuple(omega[row]),(body(q[row],np.zeros(3),quat[row]),))
            frames=tuple(ReferenceFrame(float(requested[row,i]),float(times[row,i]),tuple(rj[row,i]),tuple(rp[row,i]),tuple(rq[row,i]),tuple(jv[row,i]),tuple(lv[row,i]),tuple(av[row,i]),(body(rj[row,i],rp[row,i],rq[row,i]),)) for i in range(k))
            if step==0 or step==11 and row in (1,4):a.reset(packet,frames,0)
            prior=a.last.copy();command=a.act(packet,frames)
            expected=np.stack([getattr(command,f.name) for f in dataclasses.fields(command)])
            for name,value in feed.items():
                v=value[row:row+1].cpu().numpy()
                error=float(np.max(abs(v-a.session.feed[name])));max_input=max(max_input,error)
                np.testing.assert_allclose(v,a.session.feed[name],atol=2e-6,rtol=2e-6,err_msg=name)
            error=float(np.max(abs(actual[row]-expected)));max_command=max(max_command,error)
            np.testing.assert_allclose(actual[row],expected,atol=5e-5,rtol=2e-5)
            targets=a.last.copy();a.last[:]=prior
            poisoned=dataclasses.replace(packet,root_position_world_m=(-1e8,1e8,-1e8),root_linear_velocity_world_m_s=(1e8,-1e8,1e8))
            other=a.act(poisoned,frames)
            np.testing.assert_array_equal(np.stack([getattr(other,f.name) for f in dataclasses.fields(other)]),expected)
            a.last[:]=targets
        batched.last.copy_(tensor(np.concatenate([a.last for a in original])))
    report={'status':'passed','steps':24,'batch':slots,'joint_order':'reversed canonical names','partial_reset_step':11,'partial_reset_slots':[1,4],
            'observation_max_abs_error':max_input,'pd_command_max_abs_error':max_command,'forbidden_actual_position_linear_velocity_counterfactual':'passed',
            'checkpoint_sha256':sha256_file(args.weights),'graph_sha256':sha256_file(args.converted/'graph.json'),
            'adapter_source_sha256':{p.name:sha256_file(p) for p in root.glob('tensor*.py')},
            'reference':'Original adapter and ONNX; independent CPU MuJoCo body rotations; matched final target history',
            'kinematics_sha256':sha256_file(root/'assets/g1_reference.xml')}
    (root/'validation/tensor-observation-parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
