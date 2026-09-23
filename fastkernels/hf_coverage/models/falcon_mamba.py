"""FalconMamba adds parameter-free RMS operations to the precision-adapted Mamba stack."""

from fastkernels.hf_coverage.models.mamba import build_mamba, load_state_dict_into, make_workloads
from fastkernels.hf_coverage.patches.mamba_conv_precision import MambaConvPrecision
from fastkernels.tasks.baseline.L1.linear import Matmul
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm


class FalconMambaMixer(MambaConvPrecision):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.rms_eps = config.mixer_rms_eps
        self.dt_matmul = Matmul()

    def _ssm_transform(self, x, *, dt_contiguous=False):
        parameters = self.x_proj(x)
        dt, b, c = parameters.split((self.time_step_rank, self.ssm_state_size, self.ssm_state_size), dim=-1)
        dt = RMSNorm.forward_native(dt, None, self.rms_eps, self.time_step_rank)
        b = RMSNorm.forward_native(b, None, self.rms_eps, self.ssm_state_size)
        c = RMSNorm.forward_native(c, None, self.rms_eps, self.ssm_state_size)
        dt = self.dt_matmul(dt, self.dt_proj.weight).transpose(-2, -1)
        return (dt.contiguous() if dt_contiguous else dt), b, c


def build_from_config(config, device, dtype):
    return build_mamba(config, device, dtype, mixer_factory=FalconMambaMixer)
