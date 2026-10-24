"""Compare the adapter against pinned ProtoMotions deployment functions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from adapter import ProtoMotions
from motion_challenge.physics import World, forward_kinematics
from motion_challenge.sdk import ReferenceFrame


def definitions(path, scope, functions=(), methods=()):
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == 'MotionPlayer':
            node.body = [m for m in node.body if isinstance(m, ast.FunctionDef) and m.name in methods]
            nodes.append(node)
    exec(compile(ast.Module(ast.parse('from __future__ import annotations').body + nodes, []), str(path), 'exec'), scope)


class RecordingSession:
    def __init__(self, session): self.session = session
    def __getattr__(self, name): return getattr(self.session, name)
    def run(self, names, inputs):
        self.inputs = {k: v.copy() for k, v in inputs.items()}
        self.outputs = self.session.run(names, inputs)
        return self.outputs


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'weights', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    pins = json.loads((root / 'native-sources.json').read_text())
    for name, expected in pins['files'].items():
        assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == expected, name
    sys.path.insert(0, str(args.source.resolve()))
    from deployment.state_utils import mujoco_wxyz_to_xyzw
    from deployment.tracker_inputs import build_tracker_onnx_inputs, init_tracker_alignment, align_future_motion_refs
    runtime = json.loads((root / 'runtime.json').read_text())
    runtime['weights'] = str(args.weights.resolve())
    assert runtime['source_commit'] == pins['source_commit']
    policy = ProtoMotions(root, runtime)
    cfg = yaml.safe_load((args.source / 'data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml').read_text())
    assert cfg == policy.cfg
    np.testing.assert_array_equal(runtime['reference_offsets_s'], cfg['motion']['future_dt_seconds'])
    assert runtime['control_period_s'] == cfg['timing']['control_dt']
    assert cfg['control']['action_ema_alpha'] == 1 and cfg['control']['pd_target_max_accel'] is None
    scope = {'np': np, 'mujoco_wxyz_to_xyzw': mujoco_wxyz_to_xyzw,
             '_STATE_KEYS': ('dof_pos', 'dof_vel', 'body_rot', 'body_pos', 'body_vel', 'body_ang_vel')}
    runner = args.source / 'deployment/test_tracker_mujoco.py'
    definitions(runner, scope, functions=('read_robot_state',))
    definitions(args.source / 'deployment/motion_utils.py', scope, methods=('get_state_at_frame', 'get_future_references'))
    # Execute the author's original PD-processing block with its pinned filter settings.
    run = next(n for n in ast.parse(runner.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    loop = next(n for n in ast.walk(run) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'frame_idx')
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pd_targets' for t in n.targets))
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prev_actions' for t in n.targets))
    postprocess = compile(ast.Module(loop.body[start:end + 1], []), str(runner), 'exec')
    world = World(root / 'motion-challenge/robots/g1-29dof-cpu-v0.json')
    anchor = world.model.body('torso_link').id - 1
    assert tuple(cfg['joint_names']) == world.joint_names
    rng = np.random.default_rng(241024)
    shuffle = rng.permutation(29)
    names = tuple(np.asarray(world.joint_names)[shuffle])
    order = [names.index(n) for n in world.joint_names]
    errors = {k: 0. for k in ('inputs', 'outputs', 'motor_target', 'stiffness', 'damping', 'history', 'reset_history', 'endpoint_reference')}
    def compare(k, a, b): errors[k] = max(errors[k], float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    recorder = RecordingSession(policy.session)
    policy.session = recorder
    for reset in range(4):
        frames, cached = [], []
        for step in range(40):
            rot = R.from_euler('xyz', [.1, np.pi / 2 - 1e-5, .4]) if step % 10 == 0 else R.random(random_state=rng)
            world.data.qpos[:] = np.r_[rng.normal(0, .4, 2), .8, rot.as_quat()[[3, 0, 1, 2]], rng.normal(0, .3, 29)]
            world.data.qvel[:] = rng.normal(0, .5, 35)
            forward_kinematics(world.model, world.data)
            st = world.state()
            frames.append(ReferenceFrame(step * .02, step * .02, st.joint_position_rad, st.root_position_world_m,
                st.root_orientation_xyzw, st.joint_velocity_rad_s, st.root_linear_velocity_world_m_s,
                st.root_angular_velocity_world_rad_s, st.bodies))
            cached.append(scope['read_robot_state'](world.data, anchor))
        player = scope['MotionPlayer'].__new__(scope['MotionPlayer'])
        player._num_frames = len(frames)
        for key in ('dof_pos', 'dof_vel', 'body_rot', 'body_pos'):
            setattr(player, '_' + key, np.stack([x[key] for x in cached]))
        player._body_vel = player._body_ang_vel = np.zeros_like(player._body_pos)
        native_state = dict(np=np, pd_target_max_accel=None, use_ema=False, action_ema_alpha=1., prev_pd=None, prev_prev_pd=None, ema_prev_targets=None, prev_actions=None)
        alignment = None
        for step in range(40):
            f = frames[step]
            world.data.qpos[:] = world._qpos(f)
            if step:
                world.data.qpos[7:] += rng.normal(0, .1, 29)
                world.data.qpos[3:7] = R.random(random_state=rng).as_quat()[[3, 0, 1, 2]]
            world.data.qvel[:] = rng.normal(0, .5, 35)
            forward_kinematics(world.model, world.data)
            state = world.state()
            from dataclasses import replace
            state = replace(state, joint_names=names, joint_position_rad=tuple(np.asarray(state.joint_position_rad)[shuffle]), joint_velocity_rad_s=tuple(np.asarray(state.joint_velocity_rad_s)[shuffle]))
            offsets = cfg['motion']['future_step_indices']
            refs = tuple(replace(frames[min(step + off, 39)], joint_position_rad=tuple(np.asarray(frames[min(step + off, 39)].joint_position_rad)[shuffle]), joint_velocity_rad_s=tuple(np.asarray(frames[min(step + off, 39)].joint_velocity_rad_s)[shuffle])) for off in offsets)
            if step == 0:
                policy.reset(state, refs, reset)
                compare('reset_history', policy.last, 0.)
            robot = scope['read_robot_state'](world.data, anchor)
            if alignment is None:
                alignment = init_tracker_alignment(robot['body_pos'][anchor], robot['body_rot'][anchor], cached[0]['body_pos'][anchor], cached[0]['body_rot'][anchor])
            future = align_future_motion_refs(player.get_future_references(step, offsets), alignment)
            inputs = build_tracker_onnx_inputs(onnx_input_names=cfg['_runtime']['onnx_in_names'], onnx_name_to_key=cfg['_runtime']['onnx_name_to_in_key'], dof_pos=robot['dof_pos'], dof_vel=robot['dof_vel'], anchor_rot=robot['body_rot'][anchor], root_local_ang_vel=robot['root_local_ang_vel'], future_refs=future, anchor_body_index=anchor, num_dofs=29, prev_actions=native_state['prev_actions'])
            expected = recorder.session.run(None, inputs)
            native_state['ort_out'] = expected
            exec(postprocess, native_state)
            command = policy.act(state, refs)
            for name in inputs: compare('inputs', recorder.inputs[name], inputs[name])
            for a, b in zip(recorder.outputs, expected): compare('outputs', a, b)
            compare('motor_target', np.asarray(command.joint_position_target_rad)[order], native_state['pd_targets'])
            compare('stiffness', np.asarray(command.kp_nm_rad)[order], expected[2].reshape(29))
            compare('damping', np.asarray(command.kd_nm_s_rad)[order], expected[3].reshape(29))
            compare('history', policy.last.reshape(29), native_state['prev_actions'])
            if step >= 32:
                compare('endpoint_reference', recorder.inputs['mimic_future_dof_pos'][0], future['dof_pos'])
            # Keep identical external action context for the next independent comparison.
            native_state['prev_actions'] = policy.last.reshape(29).copy()
    limits = {k: 2e-4 for k in errors}
    for k in ('stiffness', 'damping', 'reset_history', 'endpoint_reference'): limits[k] = 0.
    report = {'passed': all(errors[k] <= limits[k] for k in errors), 'cases': 160, 'resets': 4,
        'max_abs_error': errors, 'limits': limits, 'source_commit': pins['source_commit'],
        'checkpoint_sha256': runtime['weights_sha256'],
        'scope': 'Original deployment state reader, input assembler, fixed initial heading alignment, cached MotionPlayer future lookup with endpoint clamping, original unified ONNX graph and native PD/history processing. Shared canonical robot, shuffled SDK joints and near-vertical poses. Dataset retargeting and author simulator dynamics are outside this audit.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__': main()
