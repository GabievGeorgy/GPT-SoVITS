"""
Quality-first WebAPI (v3)

This API keeps the same /tts contract as api_v2.py, but intentionally disables
streaming/fragment/parallel optimizations to maximize reliability and output
cleanliness. Requests that attempt to enable streaming are rejected.

Run (same flags as api_v2):
  python api_v3.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import traceback
import time
import wave
from io import BytesIO
from typing import Generator, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tools.i18n.i18n import I18nAuto

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import splits as _splits

i18n = I18nAuto()
cut_method_names = get_cut_method_names()


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    import subprocess

    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            str(rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-vn",
            "-f",
            "adts",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def _linear_fade_in_out(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    if fade_samples <= 0:
        return audio
    n = int(audio.shape[0])
    if n == 0:
        return audio
    fade_samples = min(fade_samples, n // 2)
    if fade_samples <= 0:
        return audio

    out = audio.astype(np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    out[:fade_samples] *= ramp
    out[-fade_samples:] *= ramp[::-1]
    return out


def _crossfade_concat(a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
    if a.size == 0:
        return b
    if b.size == 0:
        return a
    if fade_samples <= 0:
        return np.concatenate([a, b], axis=0)

    fade_samples = int(min(fade_samples, a.shape[0], b.shape[0]))
    if fade_samples <= 0:
        return np.concatenate([a, b], axis=0)

    a_tail = a[-fade_samples:].astype(np.float32)
    b_head = b[:fade_samples].astype(np.float32)
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    mixed = a_tail * (1.0 - ramp) + b_head * ramp
    return np.concatenate([a[:-fade_samples], mixed, b[fade_samples:]], axis=0)


def _ensure_punct_tail(text: str, lang: str) -> str:
    t = (text or "").strip("\n")
    if not t:
        return t
    if t[-1] not in _splits:
        t += "a?," if lang != "en" else "."
    return t


def _validate_quality_request(req: dict, languages: set) -> Optional[JSONResponse]:
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    prompt_lang: str = req.get("prompt_lang", "")
    media_type: str = req.get("media_type", "wav")
    text_split_method: str = req.get("text_split_method", "cut5")
    max_attempts = req.get("max_attempts", 3)

    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    if text_lang.lower() not in languages:
        return JSONResponse(status_code=400, content={"message": f"text_lang: {text_lang} is not supported"})
    if prompt_lang.lower() not in languages:
        return JSONResponse(status_code=400, content={"message": f"prompt_lang: {prompt_lang} is not supported"})
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    if text_split_method not in cut_method_names:
        return JSONResponse(status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"})

    try:
        max_attempts = int(max_attempts)
    except Exception:
        return JSONResponse(status_code=400, content={"message": "max_attempts must be an integer"})
    if not (1 <= max_attempts <= 10):
        return JSONResponse(status_code=400, content={"message": "max_attempts must be in [1, 10]"})

    # Quality-first mode: reject streaming/fragment modes explicitly.
    if streaming_mode not in [False, 0, None]:
        return JSONResponse(
            status_code=400,
            content={"message": "api_v3 is quality-first and does not support streaming_mode; set streaming_mode=0/false"},
        )
    if return_fragment not in [False, None]:
        return JSONResponse(
            status_code=400,
            content={"message": "api_v3 does not support return_fragment; request non-streaming /tts instead"},
        )

    return None


def _quality_tts(tts: TTS, req: dict) -> tuple[int, np.ndarray]:
    """
    Quality-first synthesis:
      - segment text (optional) via TextPreprocessor
      - for each segment: T2S -> SoVITS decode
      - crossfade between segments + optional pause
    """
    text: str = req.get("text", "")
    text_lang: str = (req.get("text_lang", "") or "").lower()
    ref_audio_path: str = req.get("ref_audio_path", "")
    aux_ref_audio_paths: list = req.get("aux_ref_audio_paths", []) or []
    prompt_text: str = req.get("prompt_text", "") or ""
    prompt_lang: str = (req.get("prompt_lang", "") or "").lower()

    # Match WebUI slider defaults.
    top_k: int = int(req.get("top_k", 15))
    top_p: float = float(req.get("top_p", 1))
    temperature: float = float(req.get("temperature", 1))
    repetition_penalty: float = float(req.get("repetition_penalty", 1.35))
    speed_factor: float = float(req.get("speed_factor", 1.0))
    fragment_interval: float = float(req.get("fragment_interval", 0.3))
    text_split_method: str = req.get("text_split_method", "cut5")
    max_attempts: int = int(req.get("max_attempts", 3))
    debug: bool = bool(req.get("debug", False))

    # Reference audio + prompt semantic
    tts.set_ref_audio(ref_audio_path)

    # Optional auxiliary reference audios
    if aux_ref_audio_paths:
        tts.prompt_cache["aux_ref_audio_paths"] = aux_ref_audio_paths
        tts.prompt_cache["refer_spec"] = [tts.prompt_cache["refer_spec"][0]]
        for path in aux_ref_audio_paths:
            if path in [None, ""]:
                continue
            if not os.path.exists(path):
                continue
            tts.prompt_cache["refer_spec"].append(tts._get_ref_spec(path))

    no_prompt_text = prompt_text.strip() == ""
    if not no_prompt_text:
        prompt_text = _ensure_punct_tail(prompt_text, prompt_lang)
        phones, bert_features, norm_text = tts.text_preprocessor.segment_and_extract_feature_for_text(
            prompt_text, prompt_lang, tts.configs.version
        )
        tts.prompt_cache["prompt_text"] = prompt_text
        tts.prompt_cache["prompt_lang"] = prompt_lang
        tts.prompt_cache["phones"] = phones
        tts.prompt_cache["bert_features"] = bert_features
        tts.prompt_cache["norm_text"] = norm_text

    # Segment and extract features for target text
    text = _ensure_punct_tail(text, text_lang)
    segments = tts.text_preprocessor.preprocess(text, text_lang, text_split_method, tts.configs.version)
    if not segments:
        return 16000, np.zeros(int(16000), dtype=np.int16)

    device = tts.configs.device
    dtype = tts.precision
    output_sr = int(tts.configs.sampling_rate if not tts.configs.use_vocoder else tts.vocoder_configs["sr"])

    refer_audio_spec: List = []
    sv_emb = [] if tts.is_v2pro else None
    for spec, audio_tensor in tts.prompt_cache["refer_spec"]:
        spec = spec.to(dtype=dtype, device=device)
        refer_audio_spec.append(spec)
        if tts.is_v2pro:
            # audio_tensor can be None if not v2pro, but guarded above
            sv_emb.append(tts.sv_model.compute_embedding3(audio_tensor))

    # Always pass the prompt semantic tokens extracted from the reference audio.
    # `prompt_text` only affects BERT/phones concatenation; it should NOT disable the semantic prompt.
    prompt = tts.prompt_cache["prompt_semantic"].unsqueeze(0).to(device)

    fade_ms = float(req.get("fade_ms", 12.0))
    fade_samples = int(output_sr * fade_ms / 1000.0)
    pause_samples = int(max(0.0, fragment_interval) * output_sr)
    pause = np.zeros(pause_samples, dtype=np.float32) if pause_samples > 0 else None

    all_audio: np.ndarray = np.zeros(0, dtype=np.float32)

    for seg_i, seg in enumerate(segments):
        seg_phones: List[int] = seg["phones"]
        seg_bert = seg["bert_features"]

        if not no_prompt_text:
            all_phones = tts.prompt_cache["phones"] + seg_phones
            all_bert = torch.cat(
                [tts.prompt_cache["bert_features"].to(device), seg_bert.to(device)],
                dim=1,
            ).unsqueeze(0)
        else:
            all_phones = seg_phones
            all_bert = seg_bert.to(device).unsqueeze(0)

        all_phoneme_ids = torch.LongTensor(all_phones).to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        with torch.no_grad():
            # The AR model is stochastic; sometimes it samples EOS too early (especially on RU).
            # Retry a few times and pick the longest semantic continuation for this segment.
            attempt_idxs: List[int] = []
            best_idx = -1
            best_semantic: Optional[torch.Tensor] = None
            for attempt in range(max_attempts):
                pred_semantic, idx = tts.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    all_bert,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=int(tts.configs.hz * tts.configs.max_sec),
                    repetition_penalty=repetition_penalty,
                )
                idx = int(idx)
                attempt_idxs.append(idx)
                semantic = pred_semantic[:, -idx:].detach()
                if idx > best_idx:
                    best_idx = idx
                    best_semantic = semantic

            if best_semantic is None:
                raise RuntimeError("T2S produced empty semantic tokens")

            pred_semantic = best_semantic.unsqueeze(0)
            if debug and max_attempts > 1:
                tail = best_semantic[0, -min(12, best_idx) :].tolist() if best_idx > 0 else []
                print(
                    f"[api_v3][t2s] phones={len(seg_phones)} attempts={attempt_idxs} picked={best_idx} tail={tail}"
                )

            # Decode this segment only (highest reliability).
            phones_tensor = torch.LongTensor(seg_phones).to(device).unsqueeze(0)
            audio_t = tts.vits_model.decode(
                pred_semantic,
                phones_tensor,
                refer_audio_spec,
                speed=speed_factor,
                sv_emb=sv_emb,
            ).detach()[0, 0, :]

        seg_audio = audio_t.float().cpu().numpy()
        max_audio = float(np.max(np.abs(seg_audio))) if seg_audio.size else 0.0
        if max_audio > 1.0:
            seg_audio = seg_audio / max_audio

        seg_audio = _linear_fade_in_out(seg_audio, fade_samples)
        if debug:
            seg_text = (seg.get("norm_text", "") or "").replace("\n", "\\n")
            print(
                f"[api_v3][seg] i={seg_i} phones={len(seg_phones)} audio_samples={int(seg_audio.shape[0])} "
                f"sec={seg_audio.shape[0] / float(output_sr):.3f} text='{seg_text}'"
            )
        all_audio = _crossfade_concat(all_audio, seg_audio, fade_samples)

        if pause is not None and pause.size > 0:
            all_audio = np.concatenate([all_audio, pause], axis=0)

    # Final safety: keep in [-1, 1]
    max_audio = float(np.max(np.abs(all_audio))) if all_audio.size else 0.0
    if max_audio > 1.0:
        all_audio = all_audio / max_audio

    audio_i16 = (all_audio * 32768.0).astype(np.int16)
    if debug:
        print(f"[api_v3][out] audio_samples={int(audio_i16.shape[0])} sec={audio_i16.shape[0] / float(output_sr):.3f}")
    return output_sr, audio_i16


APP = FastAPI()


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16
    # v3-only: optional fade control (ms)
    fade_ms: float = 12.0
    # v3-only: retry AR decoding and pick the longest continuation (quality-first).
    max_attempts: int = 3
    # v3-only: print per-request debug information to stdout.
    debug: bool = False
    # v2 internal option, accepted for strict validation
    return_fragment: bool = False


parser = argparse.ArgumentParser(description="GPT-SoVITS api v3 (quality-first)")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer config")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default=9880, help="default: 9880")
args = parser.parse_args()

tts_config = TTS_Config(args.tts_config)
print(tts_config)
tts_pipeline = TTS(tts_config)
_synth_lock = threading.RLock()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        raise SystemExit(0)


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/set_refer_audio")
async def set_refer_audio(refer_audio_path: str = None):
    try:
        if refer_audio_path in [None, ""]:
            return JSONResponse(status_code=400, content={"message": "refer_audio_path is required"})
        with _synth_lock:
            tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        with _synth_lock:
            tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        with _synth_lock:
            tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


async def _tts_handle(req: dict):
    check_res = _validate_quality_request(req, set(tts_config.languages))
    if check_res is not None:
        return check_res

    media_type = req.get("media_type", "wav")
    try:
        if bool(req.get("debug", False)):
            rid = f"{int(time.time() * 1000)}-{os.getpid()}"
            print(
                "[api_v3][req] "
                f"id={rid} "
                f"text_lang={req.get('text_lang')} prompt_lang={req.get('prompt_lang')} "
                f"cut={req.get('text_split_method')} "
                f"top_k={req.get('top_k')} top_p={req.get('top_p')} temp={req.get('temperature')} "
                f"rep_pen={req.get('repetition_penalty')} "
                f"max_attempts={req.get('max_attempts')} "
                f"text_len={len((req.get('text') or '').strip())}"
            )
        with _synth_lock:
            sr, audio_i16 = _quality_tts(tts_pipeline, req)
        audio_bytes = pack_audio(BytesIO(), audio_i16, sr, media_type).getvalue()
        return Response(audio_bytes, media_type=f"audio/{media_type}")
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})


@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 15,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
    streaming_mode: Union[bool, int] = False,
    overlap_length: int = 2,
    min_chunk_length: int = 16,
    fade_ms: float = 12.0,
    max_attempts: int = 3,
    debug: bool = False,
):
    req = {
        "text": text,
        "text_lang": (text_lang or "").lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": (prompt_lang or "").lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": bool(split_bucket),
        "fragment_interval": float(fragment_interval),
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
        "overlap_length": int(overlap_length),
        "min_chunk_length": int(min_chunk_length),
        "fade_ms": float(fade_ms),
        "max_attempts": int(max_attempts),
        "debug": bool(debug),
        "return_fragment": False,
    }
    return await _tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    # match api_v2 behavior: lowercase langs
    if "text_lang" in req and req["text_lang"] is not None:
        req["text_lang"] = str(req["text_lang"]).lower()
    if "prompt_lang" in req and req["prompt_lang"] is not None:
        req["prompt_lang"] = str(req["prompt_lang"]).lower()
    return await _tts_handle(req)


if __name__ == "__main__":
    uvicorn.run(APP, host=args.bind_addr, port=args.port, log_level="info")
