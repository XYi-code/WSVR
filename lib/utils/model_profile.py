import time

import torch
import torch.nn as nn


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_model_size_mb(model):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def _profile_with_thop(model, input_size, output_feature, device):
    from thop import profile

    class _Probe(nn.Module):
        def __init__(self, backbone, feature_name):
            super().__init__()
            self.backbone = backbone
            self.feature_name = feature_name

        def forward(self, x):
            return self.backbone(x, self.feature_name)

    probe = _Probe(model, output_feature).to(device)
    probe.eval()
    dummy = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    with torch.no_grad():
        flops, params = profile(probe, inputs=(dummy,), verbose=False)
    return flops / 1e6, params


def _profile_with_hooks(model, input_size, output_feature, device):
    flops = [0]
    hooks = []

    def conv_hook(module, inp, out):
        x = inp[0]
        batch = x.shape[0]
        out_h, out_w = out.shape[2], out.shape[3]
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups
        flops[0] += (
            batch
            * out_h
            * out_w
            * (in_channels // groups)
            * kernel_h
            * kernel_w
            * out_channels
            * 2
        )

    def linear_hook(module, inp, out):
        batch = inp[0].shape[0]
        flops[0] += batch * module.in_features * module.out_features * 2

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    dummy = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    model.eval()
    with torch.no_grad():
        model(dummy, output_feature)

    for hook in hooks:
        hook.remove()

    _, params = count_parameters(model)
    return flops[0] / 1e6, params


def estimate_mflops(model, input_size, output_feature='l2feat', device=None):
    device = device or next(model.parameters()).device
    try:
        return _profile_with_thop(model, input_size, output_feature, device)
    except Exception:
        return _profile_with_hooks(model, input_size, output_feature, device)


@torch.no_grad()
def benchmark_inference(model, input_size, output_feature='l2feat', device=None,
                        batch_size=1, warmup=20, repeat=100):
    device = device or next(model.parameters()).device
    model.eval()
    dummy = torch.randn(batch_size, 3, input_size[0], input_size[1], device=device)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            model(dummy, output_feature)
        torch.cuda.synchronize()

        starter.record()
        for _ in range(repeat):
            model(dummy, output_feature)
        ender.record()
        torch.cuda.synchronize()
        elapsed_ms = starter.elapsed_time(ender) / repeat
    else:
        for _ in range(warmup):
            model(dummy, output_feature)
        start = time.perf_counter()
        for _ in range(repeat):
            model(dummy, output_feature)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeat

    fps = 1000.0 / elapsed_ms * batch_size
    return {
        'latency_ms': elapsed_ms,
        'throughput_fps': fps,
        'batch_size': batch_size,
    }


def summarize_model(model, input_size, output_feature='l2feat', device=None,
                    batch_size=1, repeat=100):
    device = device or next(model.parameters()).device
    total_params, trainable_params = count_parameters(model)
    size_mb = estimate_model_size_mb(model)
    mflops, _ = estimate_mflops(model, input_size, output_feature, device)
    timing = benchmark_inference(
        model, input_size, output_feature, device, batch_size=batch_size, repeat=repeat
    )

    summary = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': size_mb,
        'mflops': mflops,
        'latency_ms': timing['latency_ms'],
        'throughput_fps': timing['throughput_fps'],
        'benchmark_batch_size': batch_size,
        'input_size': tuple(input_size),
        'output_feature': output_feature,
    }
    return summary


def format_model_summary(summary):
    lines = [
        '======== Model Profile ========',
        'Parameters      : {:,} total / {:,} trainable'.format(
            summary['total_params'], summary['trainable_params']
        ),
        'Model size      : {:.2f} MB'.format(summary['model_size_mb']),
        'Compute         : {:.2f} MFLOPs (input {})'.format(
            summary['mflops'], summary['input_size']
        ),
        'Inference       : {:.3f} ms/image (batch={}, {:.1f} FPS)'.format(
            summary['latency_ms'] / summary['benchmark_batch_size'],
            summary['benchmark_batch_size'],
            summary['throughput_fps'],
        ),
        'Feature head    : {}'.format(summary['output_feature']),
        '==============================',
    ]
    return '\n'.join(lines)


def print_model_summary(model, input_size, output_feature='l2feat', device=None,
                        batch_size=1, repeat=100):
    summary = summarize_model(
        model, input_size, output_feature, device, batch_size=batch_size, repeat=repeat
    )
    print(format_model_summary(summary))
    return summary
