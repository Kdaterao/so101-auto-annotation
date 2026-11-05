"""Regression checks for package discovery and fresh/resumed Linux setup failures."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('install_requirements', ROOT / 'scripts/install_requirements.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class DependencyProfilesTests(unittest.TestCase):
    def test_every_setup_profile_contains_dependencies(self):
        profiles = installer.read_profiles(ROOT / 'requirements.txt')
        for name in ('grounded', 'annotation', 'molmo2', 'robot', 'tapnet', 'viewer'):
            self.assertTrue(profiles[name], name)
        self.assertIn('transformers==5.5.4', profiles['annotation'])
        self.assertIn('transformers==4.57.6', profiles['grounded'])
        self.assertIn('-e ./lerobot[dataset]', profiles['annotation'])

    def test_installs_and_checks_package_in_requested_interpreter(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'requirements.txt').write_text('# [profile:annotation]\n#| numpy\n')
            (root / 'lerobot').mkdir()
            (root / 'lerobot/pyproject.toml').write_text('')
            def run(command, **kwargs):
                calls.append((command, kwargs))
                return subprocess.CompletedProcess(command, 0)
            with patch.object(installer, 'ROOT', root), patch.object(sys, 'argv', [
                'install_requirements.py', '--profile', 'annotation', '--installer', 'uv',
                '--python', '/target/venv/bin/python', '--pin', 'torch==2.10.0',
            ]), patch.object(installer.subprocess, 'run', side_effect=run):
                installer.main()
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0][:5], ['uv', 'pip', 'install', '--python', '/target/venv/bin/python'])
        self.assertIn('torch==2.10.0', calls[0][0])
        self.assertEqual(calls[1][0][-3:], ['--no-deps', '-e', str(root)])
        self.assertEqual(calls[2][0][:3], ['/target/venv/bin/python', '-I', '-c'])
        self.assertNotEqual(calls[2][1]['cwd'], root)

    def test_empty_catalog_fails_before_invoking_installer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'requirements.txt').write_text('-e .[dev,viewer]\n')
            with patch.object(installer, 'ROOT', root), patch.object(installer.subprocess, 'run') as run:
                with self.assertRaisesRegex(SystemExit, 'no dependency profiles'):
                    installer.main()
                run.assert_not_called()


class LinuxSetupFlowTests(unittest.TestCase):
    """Exercise real Bash flow with fake system/package/GPU commands, never apt or GPU installs."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='so101-setup-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'scripts/setup').mkdir(parents=True)
        shutil.copy(ROOT / 'scripts/setup/linux_setup.sh', self.root / 'scripts/setup/linux_setup.sh')
        shutil.copy(ROOT / 'requirements.txt', self.root / 'requirements.txt')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.log = self.root / 'commands.jsonl'
        # These wrappers log operations instead of modifying the host or downloading.
        self.fake_python = self.root / 'fake-python'
        self.fake_python.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys
with open(os.environ['COMMAND_LOG'], 'a') as f: f.write(json.dumps(['python', *sys.argv[1:]])+'\\n')
source = sys.stdin.read() if '-' in sys.argv[1:] else ''
if 'torch.ones' in source and os.environ.get('FAIL_CUDA'): sys.exit(9)
if any(arg.endswith('verify_setup.py') for arg in sys.argv):
    if os.environ.get('FAIL_VERIFY'): sys.exit(7)
    (pathlib.Path(sys.argv[0]).parent.parent / '.setup-ready.json').write_text('{{}}')
''')
        self.fake_python.chmod(0o755)
        dispatcher = f'''#!{sys.executable}
import json, os, pathlib, shutil, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['COMMAND_LOG'], 'a') as f: f.write(json.dumps([name, *args])+'\\n')
if name == 'uname': print('Linux')
if name == 'uv' and args[0] == 'venv':
    target = pathlib.Path(args[-1]) / 'bin'
    target.mkdir(parents=True)
    shutil.copy(os.environ['FAKE_PYTHON'], target / 'python')
if name == 'git':
    if args[:2] == ['submodule', 'update']:
        target = pathlib.Path(args[-1]); target.mkdir(exist_ok=True)
        (target / 'pyproject.toml').touch()
    if args and args[0] == 'clone':
        target = pathlib.Path(args[-1]) / 'launch_scripts'; target.mkdir(parents=True)
        (target / 'sft.py').touch()
'''
        for name in ('uname', 'uv', 'git', 'curl', 'ffmpeg', 'nvidia-smi'):
            path = self.bin / name
            path.write_text(dispatcher)
            path.chmod(0o755)
        self.env = dict(os.environ, PATH=f'{self.bin}:/usr/bin:/bin', SKIP_APT='1',
                        COMMAND_LOG=str(self.log), FAKE_PYTHON=str(self.fake_python),
                        VIRTUAL_ENV='/unrelated/active/venv')

    def invoke(self, *args):
        return subprocess.run(['bash', str(self.root / 'scripts/setup/linux_setup.sh'), *args],
                              env=self.env, cwd='/tmp', text=True, capture_output=True)

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_fresh_annotation_setup_uses_target_venv_and_checks_without_pythonpath(self):
        result = self.invoke('--profile', 'annotation')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        target = str(self.root / '.venv-annotation/bin/python')
        installs = [command for command in commands if command[:3] == ['uv', 'pip', 'install']]
        self.assertTrue(installs)
        for command in installs:
            self.assertEqual(command[command.index('--python') + 1], target)
        self.assertIn(['python', '-I', str(self.root / 'scripts/verify_setup.py'), '--profile', 'annotation'], commands)
        self.assertTrue((self.root / '.venv-annotation/.setup-ready.json').exists())
        self.assertIn('Setup verified', result.stdout)
        # Re-running repairs/checks the existing environment instead of recreating it.
        result = self.invoke('--profile', 'annotation')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(c[:2] == ['uv', 'venv'] for c in self.commands()), 1)

    def test_failure_of_cuda_or_final_verification_never_reports_success(self):
        for setting, expected_code in (('FAIL_CUDA', 9), ('FAIL_VERIFY', 7)):
            self.env[setting] = '1'
            result = self.invoke('--profile', 'annotation')
            self.assertEqual(result.returncode, expected_code, result.stderr)
            self.assertNotIn('Setup verified', result.stdout)
            self.assertIn('Setup failed during', result.stderr)
            self.assertFalse((self.root / '.venv-annotation/.setup-ready.json').exists())
            del self.env[setting]

    def test_check_only_does_not_install_or_download(self):
        target = self.root / '.venv-annotation/bin'
        target.mkdir(parents=True)
        shutil.copy(self.fake_python, target / 'python')
        result = self.invoke('--profile', 'annotation', '--check')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(c[0] in {'uv', 'git', 'curl', 'nvidia-smi'} for c in self.commands()))

    def test_missing_profile_fails_before_system_or_package_install(self):
        (self.root / 'requirements.txt').write_text('-e .[dev,viewer]\n')
        result = self.invoke('--profile', 'annotation')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing requirements profile', result.stderr)
        self.assertEqual(self.commands(), [['uname', '-s']])

    def test_all_profiles_are_checked_in_their_own_environments(self):
        result = self.invoke('--profile', 'all')
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ('.venv-grounded', '.venv-molmo2', '.venv', '.venv-annotation'):
            self.assertTrue((self.root / name / '.setup-ready.json').exists(), name)
        self.assertEqual(result.stdout.count('Setup verified'), 4)


if __name__ == '__main__':
    unittest.main()
