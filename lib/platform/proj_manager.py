from pathlib import Path
import os
from .monitor import TimeMonitor, TimeMonitorDisabled

class ProjManager():
    root = Path(__file__).parent.parent.parent # root / lib / utils / path_manager.py
    assert (root.exists()), 'Can\'t find the path of project root.'

    configs = root / 'configs'  # Generally, you are not supposed to access deep config through path.
    inputs  = Path(os.environ.get('HSMR_INPUTS_ROOT', root / 'data_inputs')).expanduser().resolve()
    outputs = Path(os.environ.get('HSMR_OUTPUTS_ROOT', root / 'data_outputs')).expanduser().resolve()
    assert (configs.exists()), 'Make sure you have a \'configs\' folder in the root directory.'
    inputs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    # Default values.
    cfg = None
    time_monitor = TimeMonitorDisabled()

    @staticmethod
    def init_with_cfg(cfg):
        ProjManager.cfg = cfg
        ProjManager.exp_outputs = Path(cfg.output_dir)
        if cfg.get('enable_time_monitor', False):
            ProjManager.time_monitor = TimeMonitor(ProjManager.exp_outputs, record_birth_block=False)
