#!/usr/bin/env python3
"""
Neural Activity Prediction Model
NU/NeuroBench Challenge -- Codabench Submission

Cross-domain transfer learning: ATLAS particle physics foundation model
(3.3M parameters, pretrained on charged particle track reconstruction)
adapted to predict future neural activity from monkey electrophysiology.

Architecture:
    Backbone:  ATLAS encoder [2048, 1024, 512] -> latent_dim=128  (pretrained + fine-tuned)
    Stage 1:   FeaturePredictionModel  -- waveform features -> predicted features
    Stage 2:   StepGenerationModel     -- features + timestep embed -> 10 future steps

Pipeline:
    (N, 20, C, 9) raw -> step-normalize -> v2 features -> sliding windows (30 ch)
    -> feat_pred -> step_gen -> unsort -> denormalize -> average -> (N, 20, C)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Feature names & normalization constants ───────────────────────

FEATURE_NAMES = [
    "weighted_trough_loc",
    "std",
    "channel_id_norm",
    "mean",
    "slope",
    "pulse_depth",
]

NORM_PARAMS = {
    "weighted_trough_loc": {"scale": 0.5, "offset": 0.5},
    "std":                 {"scale": 0.35, "offset": 0.36},
    "channel_id_norm":     {"scale": 0.5, "offset": 0.5},
    "mean":                {"scale": 3.0, "offset": 0.0},
    "slope":               {"scale": 0.2, "offset": 0.0},
    "pulse_depth":         {"scale": 0.5, "offset": 0.0},
}

STEP_NORM = {
    "affi": {
        "mean": np.array([
            398.2398, 408.2537, 407.3385, 391.8193, 398.0586, 402.612 , 400.0261, 383.8351, 365.5427, 387.8381,
            388.7793, 385.6788, 376.1789, 365.4136, 361.6481, 378.4985, 373.5794, 375.7334, 370.3866, 365.883 ,
            388.5212, 368.3586, 373.2055, 373.0877, 366.0438, 362.7155, 360.3305, 368.5988, 374.0713, 405.2508,
            359.6451, 367.8805, 363.764 , 377.1427, 375.4678, 376.1659, 371.3202, 370.929 , 364.2762, 382.5019,
            379.9953, 369.8248, 367.2499, 365.7916, 379.4256, 378.5709, 378.6866, 369.5418, 365.2947, 364.9153,
            382.7554, 385.9119, 380.9252, 376.7478, 368.8704, 364.9566, 387.8487, 382.0086, 381.1153, 380.6841,
            389.6674, 378.6277, 388.0274, 387.02  , 387.793 , 388.094 , 380.4501, 368.481 , 379.199 , 382.4036,
            388.0366, 387.6086, 382.7637, 373.6339, 381.5114, 382.3389, 380.5301, 392.0878, 379.4022, 377.0385,
            380.1636, 372.5632, 382.7395, 377.4392, 374.9749, 365.6471, 357.9486, 370.6919, 378.814 , 393.6864,
            384.1412, 388.7488, 382.0126, 377.346 , 388.7935, 376.0868, 373.6558, 381.4505, 379.3377, 364.4356,
            370.1677, 385.554 , 348.743 , 341.7047, 348.0982, 356.3163, 322.5133, 324.9892, 339.5469, 349.3748,
            300.997 , 325.4959, 333.7683, 319.3019, 333.6313, 341.344 , 325.2344, 336.2257, 341.0079, 346.2759,
            340.1257, 331.8737, 362.3951, 353.4448, 345.1878, 407.0017, 408.947 , 402.8412, 400.2647, 387.718 ,
            387.6309, 393.8091, 393.751 , 386.7593, 386.3888, 389.7453, 394.7683, 395.9758, 398.6558, 390.8903,
            391.7109, 390.4142, 400.7912, 375.5885, 383.825 , 392.3912, 400.9619, 391.0313, 390.4772, 379.2448,
            387.9584, 400.8473, 391.4395, 382.7602, 389.0478, 390.8492, 399.6907, 396.5501, 373.1089, 384.6072,
            400.6416, 393.0165, 376.124 , 370.5183, 379.2096, 399.4548, 382.8904, 369.8085, 361.7243, 360.701 ,
            374.6114, 364.9732, 342.8665, 357.6543, 323.2183, 299.0821, 358.7011, 324.3446, 312.067 , 381.5127,
            360.6299, 350.6426, 376.3227, 356.8442, 381.5496, 379.6137, 384.8708, 402.9993, 388.0214, 376.0944,
            384.3892, 398.6049, 399.7222, 376.3179, 400.7738, 397.1113, 388.1109, 383.418 , 384.8227, 361.7387,
            374.3177, 355.4911, 353.8096, 361.5011, 352.2234, 342.0674, 343.4465, 351.0499, 353.2472, 357.751 ,
            350.8528, 356.5818, 366.525 , 360.0802, 363.4112, 374.5293, 385.3535, 384.6245, 383.0294, 365.7304,
            381.9877, 378.7314, 370.3735, 386.3086, 393.5127, 373.5748, 390.1253, 390.5843, 383.6805, 377.7746,
            378.0596, 363.1122, 374.0213, 353.9514, 360.9124, 369.65  , 348.6674, 350.8099, 366.5587,
        ]),
        "std": np.array([
             353.216 ,  356.5786,  363.508 ,  359.7334,  329.0089,  377.3051,  343.9421,  341.9629,  359.0111,  349.0308,
             337.0789,  339.5953,  349.566 ,  366.267 ,  360.731 ,  330.6873,  340.4649,  340.15  ,  343.2564,  351.6664,
             609.1463,  338.7962,  338.918 ,  339.6145,  341.9951,  348.4565,  353.698 ,  330.6265,  334.6211,  639.8029,
             354.9815,  349.2397,  356.0069,  352.4637,  336.0867,  338.3694,  347.6414,  366.3496,  359.1468,  334.6247,
             336.7753,  358.4272,  372.3419,  356.5782,  333.9304,  338.0801,  339.7134,  352.0932,  403.3893,  373.9573,
             334.2223,  355.807 ,  339.7997,  340.542 ,  350.7129,  390.0538,  338.679 ,  334.956 ,  332.9186,  342.4619,
             344.6805,  338.4892,  332.7728,  333.7761,  337.0929,  341.2531,  342.3089,  357.3029,  335.4471,  371.6158,
             334.2717,  335.0776,  342.9653,  343.0813,  341.5679,  352.8679,  348.9458,  358.5537,  335.6658,  469.8647,
             392.3832,  348.3266,  354.9896,  351.1716,  343.95  ,  395.8676,  359.9884,  343.6121,  338.4249,  891.24  ,
             891.0171,  901.7217,  897.949 ,  900.3432,  899.8405,  913.8394,  889.6879,  891.9618,  902.8546,  887.5838,
             908.3887,  898.4026,  894.1701,  906.4004,  892.8808,  898.6528,  899.5892,  896.4086,  882.3332,  887.5757,
             900.9699,  902.5755,  390.8041,  901.1795,  385.0743,  344.9462,  365.1428,  365.8085,  358.575 ,  382.4445,
             404.4269,  377.7615,  390.8253,  397.5562,  378.5366,  447.093 ,  445.1623,  456.0628,  394.6517,  394.8723,
             388.856 ,  355.4663,  344.7347,  418.3246,  449.3517,  467.513 ,  393.9863,  435.4049,  397.2134,  364.9612,
             349.7438,  359.0825,  905.8476,  408.9485,  411.5926,  404.2299,  914.5613,  910.3408,  909.7143,  423.1995,
             427.7164,  899.4839,  904.6694,  902.3612,  905.2612,  480.5961,  395.9434,  384.3216,  447.4434,  342.1404,
             421.1453,  415.9826,  376.9398,  354.6328,  336.3863,  461.6659,  410.4735,  386.7829,  362.2631,  343.7977,
             892.9358,  924.7964,  911.7246,  902.684 ,  895.4483,  905.2677, 1002.2716,  955.647 ,  910.7639,  891.3701,
             902.3837,  896.715 ,  907.4984,  369.3926,  370.5268,  900.6643,  907.9128,  934.6895,  436.3018,  408.4524,
             927.8994,  909.766 ,  923.3193,  407.9901,  899.3516,  920.7813,  902.2217,  893.004 ,  894.5742,  929.0908,
             897.0083,  367.3316,  347.6788,  353.4974,  415.9122,  356.0772,  371.6596,  413.3538,  361.3157,  358.4509,
             399.2406,  360.0962,  397.4548,  390.1204,  383.1072,  337.0927,  895.6808,  348.71  ,  343.3605,  360.8343,
             350.8848,  368.6781,  378.4226,  340.2016,  339.5665,  350.646 ,  342.9405,  334.9496,  393.2599,  374.735 ,
             350.4951,  393.886 ,  343.4194,  412.6601,  380.0609,  351.0314,  395.6722,  352.9254,  353.7392,
        ]),
    },
    "beignet": {
        "mean": np.array([
            440.4156, 458.8973, 466.252 , 453.8111, 426.1744, 430.2894, 446.2215, 428.6954, 415.6985, 404.1126,
            427.019 , 423.2048, 409.2367, 381.1966, 370.4919, 379.105 , 385.5849, 407.8796, 377.7388, 370.7402,
            385.9699, 354.175 , 369.7957, 399.7762, 407.4662, 333.1507, 342.9739, 365.2923, 373.2792, 383.4561,
            388.5207, 336.8159, 365.9718, 402.6892, 392.0071, 412.8559, 421.2639, 374.8439, 400.8518, 402.8733,
            374.1842, 398.9347, 399.8687, 365.5472, 368.1826, 367.9321, 378.0701, 379.1348, 367.9474, 379.7562,
            373.1121, 376.8618, 382.5976, 395.8525, 358.1065, 350.0374, 351.2107, 317.6901, 343.8719, 342.3218,
            349.5149, 335.7925, 303.362 , 330.7363, 296.8724, 335.2859, 336.4776, 331.5972, 313.2148, 310.2594,
            339.3096, 298.2499, 332.0829, 308.4798, 318.5638, 308.5447, 334.8289, 347.1171, 314.4886, 315.4048,
            315.8258, 294.9614, 333.1029, 301.7169, 320.6288, 339.199 , 326.0601, 338.6326, 314.0782,
        ]),
        "std": np.array([
             522.0331,  569.9134,  421.5869,  412.4845,  472.3036,  367.5318,  338.8985,  111.7059,  359.9432,  452.8213,
             387.8908,  477.6709,  736.7105,  518.8829,  395.7192,  404.2395,  349.1204,  504.439 ,  547.8138,  603.7041,
             525.8327,  491.4913,  491.9085, 1393.3386, 1051.5448,  678.0904,  652.8908,  458.0757,  382.8436,  678.3305,
             371.8489,  820.5604,  561.6216,  452.028 ,  572.3075,  558.6714,  406.522 ,  421.7457,  413.4623,  733.6855,
             627.0844,  381.1944,  356.5563,  566.0315,  918.9009,  718.6306,  707.0248,  458.8312,  823.7801,  478.717 ,
             317.9972,  913.6069,  522.9631,  764.3921,  553.1275,  492.9274,  400.4474, 1065.4606,  361.2094,  485.7806,
             736.3306,  582.7508,  302.3215,  320.7742,  447.9622,  789.5448,  650.5738,  908.0305,  304.7886,  407.4645,
             371.9066,  505.955 ,  784.2832,  416.1056,  318.1406,  387.2977,  364.4997,  461.4548,  465.2846,  641.4669,
             757.7988,  490.0786,  354.3224,  412.0212,  506.727 ,  523.0894,  872.4887,  331.0988,  333.5367,
        ]),
    },
}


# ── Weight loading ────────────────────────────────────────────────

def _assign_weights_from_h5(model, h5_path):
    """Load HDF5 weights into a model — name-based, with positional fallback."""
    import h5py

    data = {}
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            key = name[:-2] if name.endswith(':0') else name
            data[key] = obj[()]
    with h5py.File(h5_path, 'r') as f:
        f.visititems(_visit)

    def _var_path(var):
        vname = getattr(var, 'path', None) or var.name
        return vname[:-2] if vname.endswith(':0') else vname

    # Try 1: name-based suffix matching
    loaded = 0
    for var in model.weights:
        vname = _var_path(var)
        for key, arr in data.items():
            if vname == key or vname.endswith('/' + key) or key.endswith('/' + vname):
                var.assign(arr)
                loaded += 1
                break
    if loaded > 0:
        return loaded

    # Try 2: positional assignment (sorted keys vs model weight order)
    arrays = sorted(data.items(), key=lambda x: x[0])
    model_vars = model.weights
    if len(arrays) != len(model_vars):
        raise ValueError(
            f"Weight count mismatch: h5 has {len(arrays)}, model has {len(model_vars)}")
    for (h5_key, arr), var in zip(arrays, model_vars):
        if arr.shape != tuple(var.shape):
            raise ValueError(
                f"Shape mismatch for {_var_path(var)}: h5 {arr.shape} vs model {tuple(var.shape)}")
        var.assign(arr)
    return len(arrays)


# ── Feature computation ───────────────────────────────────────────

def compute_features_all_channels(wf):
    """Compute 6 v2 features for all channels from a normalized waveform slice.

    Args:
        wf: (N, T=10, C) — step-normalized band 0 waveform

    Returns:
        dict of arrays, each shape (N, C)
    """
    N, T, C = wf.shape
    wct = wf.astype(np.float64).transpose(0, 2, 1)  # (N, C, T)

    tau = 0.3
    t_pos = np.linspace(0, 1, T)
    wct_shifted = wct - wct.max(axis=-1, keepdims=True)
    w = np.exp(wct_shifted / tau)
    weighted_trough_loc = (w * t_pos).sum(-1) / w.sum(-1)

    feat_std  = np.std(wct, axis=-1)
    feat_mean = np.mean(wct, axis=-1)

    t = np.arange(T, dtype=np.float64)
    t_c = t - t.mean()
    slope = np.sum(t_c * (wct - feat_mean[:, :, None]), axis=-1) / np.sum(t_c ** 2)
    pulse_depth = feat_mean - np.min(wct, axis=-1)

    return {
        "weighted_trough_loc": weighted_trough_loc,
        "std":                 feat_std,
        "channel_id_norm":     np.zeros((N, C), dtype=np.float64),
        "mean":                feat_mean,
        "slope":               slope,
        "pulse_depth":         pulse_depth,
    }


# ── Model architecture ────────────────────────────────────────────

def _build_encoder():
    """ATLAS backbone encoder Sequential: [2048, 1024, 512] -> latent_dim=128."""
    return keras.Sequential([
        keras.layers.Dense(2048, name='encoder_dense_0'),
        keras.layers.Activation('gelu', name='encoder_act_0'),
        keras.layers.Dense(1024, name='encoder_dense_1'),
        keras.layers.Activation('gelu', name='encoder_act_1'),
        keras.layers.Dropout(0.05,      name='encoder_dropout_1'),
        keras.layers.Dense(512,  name='encoder_dense_2'),
        keras.layers.Activation('gelu', name='encoder_act_2'),
        keras.layers.Dropout(0.1,       name='encoder_dropout_2'),
        keras.layers.Dense(128,  name='latent'),
    ], name='encoder')


class FeaturePredictionModel(keras.Model):
    """Encoder -> Dense([1024, 1024], gelu) -> (30, 6) predicted features."""

    def __init__(self):
        super().__init__(name='feature_prediction_model')
        self.encoder = _build_encoder()
        self.regression_head = keras.Sequential([
            keras.layers.Dense(1024, name='head_dense_0'),
            keras.layers.Activation('gelu', name='head_act_0'),
            keras.layers.Dropout(0.05,      name='head_dropout_0'),
            keras.layers.Dense(1024, name='head_dense_1'),
            keras.layers.Activation('gelu', name='head_act_1'),
            keras.layers.Dropout(0.05,      name='head_dropout_1'),
            keras.layers.Dense(180,  name='head_output'),
        ], name='regression_head')
        self.regression_head.build((None, 128))

    def call(self, inputs, training=False):
        x, mask = inputs
        batch_size = tf.shape(x)[0]
        latent = self.encoder(tf.reshape(x, [batch_size, -1]), training=training)
        output = tf.reshape(self.regression_head(latent, training=training),
                            [batch_size, 30, 6])
        return {'output': output, 'latent': latent}


class StepGenerationModel(keras.Model):
    """Encoder + timestep embedding(32) -> shared MLP -> (30, 10) step predictions."""

    def __init__(self):
        super().__init__(name='step_generation_model')
        self.encoder = _build_encoder()
        self.timestep_embedding = keras.layers.Embedding(10, 32, name='timestep_embedding')
        self.timestep_embedding.build((None,))
        self.shared_mlp = keras.Sequential([
            keras.layers.Dense(256, name='mlp_dense_0'),
            keras.layers.Activation('gelu', name='mlp_act_0'),
            keras.layers.Dropout(0.1,       name='mlp_dropout_0'),
            keras.layers.Dense(128, name='mlp_dense_1'),
            keras.layers.Activation('gelu', name='mlp_act_1'),
            keras.layers.Dropout(0.1,       name='mlp_dropout_1'),
            keras.layers.Dense(30,  name='mlp_output'),
        ], name='shared_mlp')
        self.shared_mlp.build((None, 160))  # latent_dim=128 + embed_dim=32

    def call(self, inputs, training=False):
        x, mask = inputs
        batch_size = tf.shape(x)[0]
        latent = self.encoder(tf.reshape(x, [batch_size, -1]), training=training)

        t_embeds = self.timestep_embedding(tf.range(10))              # (10, 32)
        latent_t  = tf.tile(tf.expand_dims(latent,    1), [1, 10, 1])  # (B, 10, 128)
        t_embeds_t = tf.tile(tf.expand_dims(t_embeds, 0), [batch_size, 1, 1])  # (B, 10, 32)
        conditioned = tf.reshape(tf.concat([latent_t, t_embeds_t], axis=-1), [-1, 160])

        pred = tf.reshape(self.shared_mlp(conditioned, training=training),
                          [batch_size, 10, 30])
        return {'output': tf.transpose(pred, [0, 2, 1]), 'latent': latent}


# ── Inference pipeline ────────────────────────────────────────────

def _assemble_window_features(feats_raw, w_channels, sort_order, ch_id_sorted,
                               N_idx, n_select, N):
    feat_input = np.zeros((N, n_select, len(FEATURE_NAMES)), dtype=np.float32)
    sorted_ch = w_channels[sort_order]
    for fi, fname in enumerate(FEATURE_NAMES):
        p = NORM_PARAMS[fname]
        if fname == "channel_id_norm":
            feat_input[:, :, fi] = ((ch_id_sorted - p["offset"]) / p["scale"]).astype(np.float32)
        else:
            feat_input[:, :, fi] = ((feats_raw[fname][N_idx, sorted_ch] - p["offset"]) / p["scale"]).astype(np.float32)
    return feat_input


def _run_batched(model, x, masks, batch_size=512):
    out = []
    for i in range(0, len(x), batch_size):
        xb = tf.constant(x[i:i+batch_size])
        mb = tf.constant(masks[i:i+batch_size])
        out.append(model((xb, mb), training=False)['output'].numpy())
    return np.concatenate(out, axis=0)


def _run_chain(X, monkey_name, feat_model, step_model, n_select=30, stride=1, batch_size=512):
    """End-to-end chain inference.

    Args:
        X: (N, 20, C, F) raw input array — first 10 steps meaningful, last 10 masked
    Returns:
        (N, 20, C) — first 10 steps raw passthrough, last 10 predicted
    """
    N, T, C, F = X.shape
    step_mean = STEP_NORM[monkey_name]["mean"]
    step_std  = STEP_NORM[monkey_name]["std"]

    band0        = X[:, :, :, 0].astype(np.float32)                   # (N, 20, C)
    band0_normed = (band0 - step_mean[None, None, :]) / step_std[None, None, :]
    feats_raw    = compute_features_all_channels(band0_normed[:, :10, :])

    window_starts   = np.arange(0, C - n_select + 1, stride)
    n_windows       = len(window_starts)
    win_ch          = window_starts[:, None] + np.arange(n_select)[None, :]
    raw_ch_id_norm  = np.arange(n_select, dtype=np.float64) / (n_select - 1)

    pred_sum   = np.zeros((N, C, 10), dtype=np.float64)
    pred_count = np.zeros(C, dtype=np.int32)
    masks      = np.ones((N, n_select), dtype=np.float32)

    for w_idx in range(n_windows):
        w_channels  = win_ch[w_idx]
        sort_order  = np.argsort(-feats_raw["std"][:, w_channels], axis=1)
        N_idx       = np.arange(N)[:, None]
        ch_id_sorted = raw_ch_id_norm[sort_order]

        feat_input       = _assemble_window_features(
            feats_raw, w_channels, sort_order, ch_id_sorted, N_idx, n_select, N)
        pred_feat        = _run_batched(feat_model, feat_input, masks, batch_size)
        pred_steps_normed = _run_batched(step_model, pred_feat, masks, batch_size)

        inv_order  = np.argsort(sort_order, axis=1)
        pred_raw   = (pred_steps_normed[N_idx, inv_order, :] *
                      step_std[w_channels][None, :, None] +
                      step_mean[w_channels][None, :, None])

        for pos in range(n_select):
            ch = w_channels[pos]
            pred_sum[:, ch, :] += pred_raw[:, pos, :]
            pred_count[ch] += 1

    pred_avg = (pred_sum / pred_count[None, :, None]).astype(np.float32)

    output = np.zeros((N, 20, C), dtype=np.float32)
    output[:, :10, :] = band0[:, :10, :]
    output[:, 10:, :] = pred_avg.transpose(0, 2, 1)
    return output


# ── Codabench Model class ─────────────────────────────────────────

class Model:
    def __init__(self, monkey_name=""):
        self.monkey_name = monkey_name
        if monkey_name == "affi":
            self.n_channels = 239
        elif monkey_name == "beignet":
            self.n_channels = 89
        else:
            raise ValueError(f"Unknown monkey: {monkey_name!r}")
        self.feat_model = None
        self.step_model = None

    def load(self):
        monkey_key  = "affi" if self.monkey_name == "affi" else "bei"
        feat_weights = os.path.join(_DIR, f"feat_pred_{monkey_key}.weights.h5")
        step_weights = os.path.join(_DIR, f"step_gen_{monkey_key}.weights.h5")

        self.feat_model = FeaturePredictionModel()
        self.feat_model((tf.zeros((1, 30, 6)), tf.ones((1, 30))), training=False)
        n = _assign_weights_from_h5(self.feat_model, feat_weights)
        print(f"✓ feat_pred ({n} variables) from {os.path.basename(feat_weights)}")

        self.step_model = StepGenerationModel()
        self.step_model((tf.zeros((1, 30, 6)), tf.ones((1, 30))), training=False)
        n = _assign_weights_from_h5(self.step_model, step_weights)
        print(f"✓ step_gen ({n} variables) from {os.path.basename(step_weights)}")

    def predict(self, X):
        """
        Args:
            X: (N, 20, C, 9) — first 10 steps meaningful, last 10 masked
        Returns:
            (N, 20, C) — first 10 steps raw passthrough, last 10 predicted
        """
        return _run_chain(X, self.monkey_name, self.feat_model, self.step_model)
