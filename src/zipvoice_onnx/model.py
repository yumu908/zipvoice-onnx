import numpy as np
import onnxruntime as ort


def get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
) -> np.ndarray:
    timesteps = np.linspace(t_start, t_end, num_step + 1, dtype=np.float32)
    timesteps = (t_shift * timesteps / (1 + (t_shift - 1) * timesteps)).astype(np.float32)
    return timesteps


class OnnxModel:
    def __init__(self, text_encoder_path: str, fm_decoder_path: str, num_thread: int = 1):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = num_thread
        session_opts.intra_op_num_threads = num_thread

        self.session_opts = session_opts
        self.init_text_encoder(text_encoder_path)
        self.init_fm_decoder(fm_decoder_path)

    def init_text_encoder(self, model_path: str):
        self.text_encoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_fm_decoder(self, model_path: str):
        self.fm_decoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])

    def run_text_encoder(self, tokens: np.ndarray, prompt_tokens: np.ndarray, 
                        prompt_features_len: np.ndarray, speed: np.ndarray) -> np.ndarray:
        out = self.text_encoder.run(
            [self.text_encoder.get_outputs()[0].name],
            {
                self.text_encoder.get_inputs()[0].name: tokens,
                self.text_encoder.get_inputs()[1].name: prompt_tokens,
                self.text_encoder.get_inputs()[2].name: prompt_features_len,
                self.text_encoder.get_inputs()[3].name: speed,
            },
        )
        return out[0]

    def run_fm_decoder(self, t: np.ndarray, x: np.ndarray, text_condition: np.ndarray,
                      speech_condition: np.ndarray, guidance_scale: np.ndarray) -> np.ndarray:
        out = self.fm_decoder.run(
            [self.fm_decoder.get_outputs()[0].name],
            {
                self.fm_decoder.get_inputs()[0].name: t,
                self.fm_decoder.get_inputs()[1].name: x,
                self.fm_decoder.get_inputs()[2].name: text_condition,
                self.fm_decoder.get_inputs()[3].name: speech_condition,
                self.fm_decoder.get_inputs()[4].name: guidance_scale,
            },
        )
        return out[0]


def sample(model: OnnxModel, tokens: list, prompt_tokens: list, 
          prompt_features: np.ndarray, speed: float = 1.0, t_shift: float = 0.5,
          guidance_scale: float = 1.0, num_step: int = 16) -> np.ndarray:
    tokens = np.array([tokens], dtype=np.int64)
    prompt_tokens = np.array([prompt_tokens], dtype=np.int64)
    prompt_features_len = np.array([prompt_features.shape[1]], dtype=np.int64)
    speed_tensor = np.array([speed], dtype=np.float32)

    text_condition = model.run_text_encoder(
        tokens, prompt_tokens, prompt_features_len, speed_tensor
    )

    batch_size, num_frames, _ = text_condition.shape
    feat_dim = model.feat_dim

    timesteps = get_time_steps(
        t_start=0.0, t_end=1.0, num_step=num_step, t_shift=t_shift
    )
    np.random.seed(666)
    x = np.random.randn(batch_size, num_frames, feat_dim).astype(np.float32)
    
    pad_width = ((0, 0), (0, num_frames - prompt_features.shape[1]), (0, 0))
    speech_condition = np.pad(prompt_features, pad_width, mode='constant').astype(np.float32)
    
    guidance_scale_tensor = np.array(guidance_scale, dtype=np.float32)

    for step in range(num_step):
        t_step = np.array(timesteps[step], dtype=np.float32)
        dt_step = np.float32(timesteps[step + 1] - timesteps[step])
        v = model.run_fm_decoder(
            t=t_step,
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale_tensor,
        )
        x = (x + v * dt_step).astype(np.float32)

    x = x[:, prompt_features_len[0]:, :]
    return x

