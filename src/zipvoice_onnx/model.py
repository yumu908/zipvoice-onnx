import numpy as np
import onnxruntime as ort


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

