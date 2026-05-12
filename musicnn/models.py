"""PyTorch implementation of musicnn/vgg audio tagging models.

Weights are mutually intelligible with the original TF v1 checkpoints.
Use ``load_tf_checkpoint(model, ckpt_path)`` to transfer weights.

TF -> PyTorch weight conversion rules
--------------------------------------
  Conv2d kernel : TF [H, W, C_in, C_out] -> PT [C_out, C_in, H, W]  (.permute(3,2,0,1))
  Dense kernel  : TF [in, out]            -> PT [out, in]             (.T)
  BatchNorm     : gamma->weight, beta->bias,
                  moving_mean->running_mean, moving_variance->running_var  (direct copy)
  Biases        : direct copy (same shape)

TF checkpoint variable mapping (MusicNN)
-----------------------------------------
  batch_normalization/...     -> model.input_bn
  conv2d/...                  -> model.timbral_f74.conv
  batch_normalization_1/...   -> model.timbral_f74.bn
  conv2d_1/...                -> model.timbral_f77.conv
  batch_normalization_2/...   -> model.timbral_f77.bn
  conv2d_2/...                -> model.tempo_s1.conv
  batch_normalization_3/...   -> model.tempo_s1.bn
  conv2d_3/...                -> model.tempo_s2.conv
  batch_normalization_4/...   -> model.tempo_s2.bn
  conv2d_4/...                -> model.tempo_s3.conv
  batch_normalization_5/...   -> model.tempo_s3.bn
  conv2d_5/...                -> model.midend_conv1.conv
  batch_normalization_6/...   -> model.midend_conv1.bn
  conv2d_6/...                -> model.midend_conv2.conv
  batch_normalization_7/...   -> model.midend_conv2.bn
  conv2d_7/...                -> model.midend_conv3.conv
  batch_normalization_8/...   -> model.midend_conv3.bn
  batch_normalization_9/...   -> model.flat_bn
  dense/...                   -> model.dense
  batch_normalization_10/...  -> model.dense_bn
  dense_1/...                 -> model.logits

TF checkpoint variable mapping (VGG)
--------------------------------------
  batch_normalization/...     -> model.input_bn
  1CNN/...                    -> model.conv1
  batch_normalization_1/...   -> model.bn1
  2CNN/...                    -> model.conv2
  batch_normalization_2/...   -> model.bn2
  3CNN/...                    -> model.conv3
  batch_normalization_3/...   -> model.bn3
  4CNN/...                    -> model.conv4
  batch_normalization_4/...   -> model.bn4
  5CNN/...                    -> model.conv5
  batch_normalization_5/...   -> model.bn5
  dense/...                   -> model.output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import configuration as config

# Match TF v1 BatchNorm defaults: epsilon=0.001, decay=0.99 (PyTorch momentum = 1 - decay)
_BN_EPS = 1e-3
_BN_MOMENTUM = 0.01


# ---------------------------------------------------------------------------
# Public factory (mirrors original TF define_model signature, minus tensors)
# ---------------------------------------------------------------------------

def define_model(model_name, num_classes):
    """Return an nn.Module for *model_name* with *num_classes* outputs."""
    if model_name in ('MTT_musicnn', 'MSD_musicnn'):
        return MusicnnModel(num_classes, num_filt_midend=64, num_units_backend=200)
    elif model_name == 'MSD_musicnn_big':
        return MusicnnModel(num_classes, num_filt_midend=512, num_units_backend=500)
    elif model_name in ('MTT_vgg', 'MSD_vgg'):
        return VGGModel(num_classes)
    else:
        raise ValueError('Model not implemented!')


# ---------------------------------------------------------------------------
# Building-block modules
# ---------------------------------------------------------------------------

class TimbralBlock(nn.Module):
    """Conv2d(valid) -> ReLU -> BN -> MaxPool(all-freq) -> squeeze freq dim."""

    def __init__(self, filters, kernel_h, kernel_w):
        super().__init__()
        self.conv = nn.Conv2d(1, filters, (kernel_h, kernel_w), padding=0)
        self.bn   = nn.BatchNorm2d(filters, eps=_BN_EPS, momentum=_BN_MOMENTUM)

    def forward(self, x):
        # x: [B, 1, T+6, F]  (pre-padded ±3 on time axis)
        x = F.relu(self.conv(x))              # [B, filters, T, F-k+1]
        x = self.bn(x)
        x = F.max_pool2d(x, (1, x.shape[3])) # [B, filters, T, 1]
        return x.squeeze(3)                   # [B, filters, T]


class TempoBlock(nn.Module):
    """Conv2d(same-time) -> ReLU -> BN -> MaxPool(all-freq) -> squeeze freq dim."""

    def __init__(self, filters, kernel_h, n_mels):
        super().__init__()
        self.conv = nn.Conv2d(1, filters, (kernel_h, 1), padding=0)
        self.bn   = nn.BatchNorm2d(filters, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        # TF 'same' padding for stride=1: total = k-1, pad_before = (k-1)//2
        self.pad_before = (kernel_h - 1) // 2
        self.pad_after  = kernel_h - 1 - self.pad_before
        self.n_mels = n_mels

    def forward(self, x):
        # x: [B, 1, T, F]
        x = F.pad(x, (0, 0, self.pad_before, self.pad_after)) # [B, 1, T, F] (same T)
        x = F.relu(self.conv(x))                               # [B, filters, T, F]
        x = self.bn(x)
        x = F.max_pool2d(x, (1, self.n_mels))                 # [B, filters, T, 1]
        return x.squeeze(3)                                    # [B, filters, T]


class MidendLayer(nn.Module):
    """pad(±3 time) -> Conv2d(valid, [7, in_w]) -> ReLU -> BN -> permute.

    Operates on tensors in layout [B, 1, T, W] (channel=1, features in W).
    Produces output in the same layout [B, 1, T, n_filt], mirroring the
    TF transpose [0,1,3,2] after each conv in the midend.
    """

    def __init__(self, in_width, n_filt):
        super().__init__()
        self.conv = nn.Conv2d(1, n_filt, (7, in_width), padding=0)
        self.bn   = nn.BatchNorm2d(n_filt, eps=_BN_EPS, momentum=_BN_MOMENTUM)

    def forward(self, x):
        # x: [B, 1, T, in_width]
        x = F.pad(x, (0, 0, 3, 3))      # [B, 1, T+6, in_width]
        x = F.relu(self.conv(x))          # [B, n_filt, T, 1]
        x = self.bn(x)
        return x.permute(0, 3, 2, 1)     # [B, 1, T, n_filt]


# ---------------------------------------------------------------------------
# MusicNN model
# ---------------------------------------------------------------------------

class MusicnnModel(nn.Module):
    """Musically-motivated CNN with timbral/temporal front-end.

    Forward returns the same tuple as the original TF define_model:
      (logits, timbral, temporal, cnn1, cnn2, cnn3, mean_pool, max_pool, penultimate)

    Feature tensor shapes (matching TF convention: time first):
      timbral, temporal, cnn1, cnn2, cnn3 : [B, T, features]
      mean_pool, max_pool                 : [B, total_features]
      penultimate                         : [B, num_units_backend]
      logits                              : [B, num_classes]
    """

    def __init__(self, num_classes, n_mels=config.N_MELS, num_filt_frontend=1.6,
                 num_filt_midend=64, num_units_backend=200):
        super().__init__()
        f  = num_filt_frontend
        f74 = int(f * 128)
        f77 = int(f * 128)
        sf  = int(f * 32)
        total_frontend = f74 + f77 + sf * 3

        # --- frontend ---
        self.input_bn    = nn.BatchNorm2d(1, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.timbral_f74 = TimbralBlock(f74, 7, int(0.4 * n_mels))
        self.timbral_f77 = TimbralBlock(f77, 7, int(0.7 * n_mels))
        self.tempo_s1    = TempoBlock(sf, 128, n_mels)
        self.tempo_s2    = TempoBlock(sf,  64, n_mels)
        self.tempo_s3    = TempoBlock(sf,  32, n_mels)

        # --- midend ---
        self.midend_conv1 = MidendLayer(total_frontend,  num_filt_midend)
        self.midend_conv2 = MidendLayer(num_filt_midend, num_filt_midend)
        self.midend_conv3 = MidendLayer(num_filt_midend, num_filt_midend)

        # --- backend ---
        total_features = total_frontend + 3 * num_filt_midend
        flat_size      = 2 * total_features  # max_pool ++ mean_pool

        self.flat_bn  = nn.BatchNorm1d(flat_size, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.dense    = nn.Linear(flat_size, num_units_backend)
        self.dense_bn = nn.BatchNorm1d(num_units_backend, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.logits   = nn.Linear(num_units_backend, num_classes)

        self._f74 = f74
        self._f77 = f77

    def forward(self, x):
        # x: [B, T, F]

        # ---- frontend ----
        x4d   = x.unsqueeze(1)                          # [B, 1, T, F]
        x4d   = self.input_bn(x4d)
        x_pad = F.pad(x4d, (0, 0, 3, 3))               # [B, 1, T+6, F]

        f74 = self.timbral_f74(x_pad)                   # [B, f74, T]
        f77 = self.timbral_f77(x_pad)                   # [B, f77, T]
        s1  = self.tempo_s1(x4d)                        # [B, sf,  T]
        s2  = self.tempo_s2(x4d)
        s3  = self.tempo_s3(x4d)

        frontend_feats = torch.cat([f74, f77, s1, s2, s3], dim=1)  # [B, total_frontend, T]

        # ---- midend ----
        # Rearrange to [B, 1, T, total_frontend] to mirror TF's NHWC layout
        fe = frontend_feats.permute(0, 2, 1).unsqueeze(1)  # [B, 1, T, total_frontend]

        conv1_t = self.midend_conv1(fe)          # [B, 1, T, n_filt]
        conv2_t = self.midend_conv2(conv1_t)     # [B, 1, T, n_filt]
        res2    = conv2_t + conv1_t
        conv3_t = self.midend_conv3(res2)        # [B, 1, T, n_filt]
        res3    = conv3_t + res2

        # Dense skip-connection concat along feature (W) dim
        midend_feats = torch.cat([fe, conv1_t, res2, res3], dim=3)  # [B, 1, T, total_features]

        # ---- backend ----
        max_pool  = midend_feats.max(dim=2)[0]   # [B, 1, total_features]
        mean_pool = midend_feats.mean(dim=2)      # [B, 1, total_features]
        tmp_pool  = torch.cat([max_pool, mean_pool], dim=2)  # [B, 1, 2*total]
        flat      = tmp_pool.flatten(1)           # [B, 2*total]

        flat      = self.flat_bn(flat)
        flat      = F.dropout(flat, p=0.5, training=self.training)
        dense_out = F.relu(self.dense(flat))
        dense_out = self.dense_bn(dense_out)
        dense_out = F.dropout(dense_out, p=0.5, training=self.training)
        out       = self.logits(dense_out)

        # ---- feature extraction outputs (time-first to match TF shapes) ----
        timbral  = torch.cat([f74, f77], dim=1).permute(0, 2, 1)   # [B, T, f74+f77]
        temporal = torch.cat([s1, s2, s3], dim=1).permute(0, 2, 1) # [B, T, 3*sf]
        cnn1     = conv1_t.squeeze(1)                               # [B, T, n_filt]
        cnn2     = res2.squeeze(1)                                  # [B, T, n_filt]
        cnn3     = res3.squeeze(1)                                  # [B, T, n_filt]
        mean_out = mean_pool.squeeze(1)                             # [B, total_features]
        max_out  = max_pool.squeeze(1)                              # [B, total_features]

        return out, timbral, temporal, cnn1, cnn2, cnn3, mean_out, max_out, dense_out


# ---------------------------------------------------------------------------
# VGG model
# ---------------------------------------------------------------------------

def _vgg_pool_output_size(h, w, pool_size, stride):
    """Apply one max-pool step (valid padding) and return (h_out, w_out)."""
    return (h - pool_size[0]) // stride[0] + 1, (w - pool_size[1]) // stride[1] + 1


class VGGModel(nn.Module):
    """VGG-style CNN for audio tagging.

    Forward returns the same tuple as the original TF vgg():
      (output, pool1, pool2, pool3, pool4, pool5)

    All pooling feature maps are in PyTorch NCHW format [B, C, T, F].
    """

    def __init__(self, num_classes, num_filters=128,
                 n_mels=config.N_MELS, n_frames=187):
        super().__init__()
        nf = num_filters

        self.input_bn = nn.BatchNorm2d(1,  eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.conv1    = nn.Conv2d(1,  nf, 3, padding=1)
        self.bn1      = nn.BatchNorm2d(nf, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.conv2    = nn.Conv2d(nf, nf, 3, padding=1)
        self.bn2      = nn.BatchNorm2d(nf, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.conv3    = nn.Conv2d(nf, nf, 3, padding=1)
        self.bn3      = nn.BatchNorm2d(nf, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.conv4    = nn.Conv2d(nf, nf, 3, padding=1)
        self.bn4      = nn.BatchNorm2d(nf, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.conv5    = nn.Conv2d(nf, nf, 3, padding=1)
        self.bn5      = nn.BatchNorm2d(nf, eps=_BN_EPS, momentum=_BN_MOMENTUM)

        # Compute flattened size from pooling geometry (valid padding throughout)
        h, w = n_frames, n_mels
        h, w = _vgg_pool_output_size(h, w, (4, 1), (2, 2))
        h, w = _vgg_pool_output_size(h, w, (2, 2), (2, 2))
        h, w = _vgg_pool_output_size(h, w, (2, 2), (2, 2))
        h, w = _vgg_pool_output_size(h, w, (2, 2), (2, 2))
        h, w = _vgg_pool_output_size(h, w, (4, 4), (4, 4))
        self.output = nn.Linear(nf * h * w, num_classes)

    def forward(self, x):
        # x: [B, T, F]
        x = x.unsqueeze(1)                                  # [B, 1, T, F]
        x = self.input_bn(x)

        x     = F.relu(self.conv1(x))
        x     = self.bn1(x)
        pool1 = F.max_pool2d(x, (4, 1), stride=(2, 2))
        x     = F.dropout(pool1, p=0.25, training=self.training)

        x     = F.relu(self.conv2(x))
        x     = self.bn2(x)
        pool2 = F.max_pool2d(x, (2, 2), stride=(2, 2))
        x     = F.dropout(pool2, p=0.25, training=self.training)

        x     = F.relu(self.conv3(x))
        x     = self.bn3(x)
        pool3 = F.max_pool2d(x, (2, 2), stride=(2, 2))
        x     = F.dropout(pool3, p=0.25, training=self.training)

        x     = F.relu(self.conv4(x))
        x     = self.bn4(x)
        pool4 = F.max_pool2d(x, (2, 2), stride=(2, 2))
        x     = F.dropout(pool4, p=0.25, training=self.training)

        x     = F.relu(self.conv5(x))
        x     = self.bn5(x)
        pool5 = F.max_pool2d(x, (4, 4), stride=(4, 4))

        flat   = F.dropout(pool5.flatten(1), p=0.5, training=self.training)
        output = self.output(flat)
        return output, pool1, pool2, pool3, pool4, pool5


# ---------------------------------------------------------------------------
# TF checkpoint -> PyTorch weight transfer
# ---------------------------------------------------------------------------

# Maps TF variable prefix -> PyTorch state_dict prefix + conversion type
_MUSICNN_MAP = [
    ('batch_normalization',    'input_bn',         'bn'),
    ('conv2d',                 'timbral_f74.conv',  'conv'),
    ('batch_normalization_1',  'timbral_f74.bn',    'bn'),
    ('conv2d_1',               'timbral_f77.conv',  'conv'),
    ('batch_normalization_2',  'timbral_f77.bn',    'bn'),
    ('conv2d_2',               'tempo_s1.conv',     'conv'),
    ('batch_normalization_3',  'tempo_s1.bn',       'bn'),
    ('conv2d_3',               'tempo_s2.conv',     'conv'),
    ('batch_normalization_4',  'tempo_s2.bn',       'bn'),
    ('conv2d_4',               'tempo_s3.conv',     'conv'),
    ('batch_normalization_5',  'tempo_s3.bn',       'bn'),
    ('conv2d_5',               'midend_conv1.conv', 'conv'),
    ('batch_normalization_6',  'midend_conv1.bn',   'bn'),
    ('conv2d_6',               'midend_conv2.conv', 'conv'),
    ('batch_normalization_7',  'midend_conv2.bn',   'bn'),
    ('conv2d_7',               'midend_conv3.conv', 'conv'),
    ('batch_normalization_8',  'midend_conv3.bn',   'bn'),
    ('batch_normalization_9',  'flat_bn',           'bn'),
    ('dense',                  'dense',             'dense'),
    ('batch_normalization_10', 'dense_bn',          'bn'),
    ('dense_1',                'logits',            'dense'),
]

_VGG_MAP = [
    ('batch_normalization',   'input_bn', 'bn'),
    ('1CNN',                  'conv1',    'conv'),
    ('batch_normalization_1', 'bn1',      'bn'),
    ('2CNN',                  'conv2',    'conv'),
    ('batch_normalization_2', 'bn2',      'bn'),
    ('3CNN',                  'conv3',    'conv'),
    ('batch_normalization_3', 'bn3',      'bn'),
    ('4CNN',                  'conv4',    'conv'),
    ('batch_normalization_4', 'bn4',      'bn'),
    ('5CNN',                  'conv5',    'conv'),
    ('batch_normalization_5', 'bn5',      'bn'),
    ('dense',                 'output',   'dense'),
]


def _load_tf_var(reader, name):
    return torch.tensor(reader.get_tensor(name))


def tf_checkpoint_to_state_dict(model, ckpt_path):
    """Build a PyTorch ``state_dict`` from a TF v1 checkpoint.

    Args:
        model: a ``MusicnnModel`` or ``VGGModel`` instance (used only to
               determine the correct variable map).
        ckpt_path: path to the TF checkpoint directory or file prefix.

    Returns:
        A ``dict`` compatible with ``model.load_state_dict()``.

    Requires ``tensorflow`` to be installed (TF is used only for reading the
    checkpoint; the returned state_dict is pure PyTorch).
    """
    import tensorflow as tf  # optional dep; only needed for conversion

    reader = tf.train.load_checkpoint(ckpt_path)
    var_map = _MUSICNN_MAP if isinstance(model, MusicnnModel) else _VGG_MAP
    sd = {}

    for tf_prefix, pt_prefix, kind in var_map:
        if kind == 'conv':
            # TF kernel: [H, W, C_in, C_out] -> PT: [C_out, C_in, H, W]
            kernel = _load_tf_var(reader, f'{tf_prefix}/kernel')
            sd[f'{pt_prefix}.weight'] = kernel.permute(3, 2, 0, 1).contiguous()
            sd[f'{pt_prefix}.bias']   = _load_tf_var(reader, f'{tf_prefix}/bias')

        elif kind == 'dense':
            # TF kernel: [in, out] -> PT: [out, in]
            kernel = _load_tf_var(reader, f'{tf_prefix}/kernel')
            sd[f'{pt_prefix}.weight'] = kernel.T.contiguous()
            sd[f'{pt_prefix}.bias']   = _load_tf_var(reader, f'{tf_prefix}/bias')

        elif kind == 'bn':
            sd[f'{pt_prefix}.weight']       = _load_tf_var(reader, f'{tf_prefix}/gamma')
            sd[f'{pt_prefix}.bias']         = _load_tf_var(reader, f'{tf_prefix}/beta')
            sd[f'{pt_prefix}.running_mean'] = _load_tf_var(reader, f'{tf_prefix}/moving_mean')
            sd[f'{pt_prefix}.running_var']  = _load_tf_var(reader, f'{tf_prefix}/moving_variance')
            # PyTorch BN also tracks num_batches; initialise to 0 (inference-ready)
            sd[f'{pt_prefix}.num_batches_tracked'] = torch.zeros(1, dtype=torch.long)

    return sd


def load_tf_checkpoint(model, ckpt_path):
    """Load a TF v1 checkpoint directly into *model* (in-place).

    Args:
        model: a ``MusicnnModel`` or ``VGGModel`` instance.
        ckpt_path: path to the TF checkpoint directory or file prefix.

    Requires ``tensorflow`` to be installed.
    """
    sd = tf_checkpoint_to_state_dict(model, ckpt_path)
    model.load_state_dict(sd, strict=False)  # strict=False: ignores extra keys   


