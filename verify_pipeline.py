"""
Fish Audio S2 Pro 하이브리드 파이프라인 — 빌드 + 검증
Kaggle 노트북에서 cell 단위로 순차 실행한다.

Cell 0: 소스에서 s2 바이너리 빌드 (CUDA) + /kaggle/working/s2-binary/ 에 복제
Cell 1: GPU 환경 확인
Cell 2: s2.cpp 서버 기동
Cell 3: /generate_tokens 토큰 검증
Cell 4: fish-speech 보코더 로드
Cell 5: 토큰 → WAV 디코딩 (핵심)
Cell 6: Voice Cloning 검증
Cell 7: 듀얼 GPU 병렬 검증

빌드 완료 후 /kaggle/working/s2-binary/ 를 다운로드하면
다른 노트북에서 소스 빌드 없이 바이너리만으로 실행 가능.

GPU 설정:
  GPU_IDS 리스트를 수정하여 사용할 GPU를 선택한다.
  단일 GPU: GPU_IDS = [0]
  듀얼 GPU: GPU_IDS = [0, 1]  (기본값)
"""

# ============================================================
# 설정 — 여기서 GPU, 경로를 조정
# ============================================================
GPU_IDS = [0, 1]  # 사용할 GPU 인덱스 리스트

# Git repo — 포크한 s2.cpp 소스
S2_REPO = "https://github.com/GGJJack/s2.cpp.git"
S2_BRANCH = "main"

# 모델 / 토크나이저 (Kaggle Dataset으로 미리 업로드해둔 경로)
MODEL = "/kaggle/input/datasets/jjackchoi/s2-pro-gguf/s2-pro-q8_0.gguf"
TOKENIZER = "/kaggle/input/datasets/jjackchoi/s2-pro-gguf/tokenizer.json"

# 보코더 — fish-speech의 codec.pth (인코더+디코더 통합, 디코더만 사용)
# 기존 Dataset 중 하나를 사용:
#   jjackchoi/fish-speech-s2-pro/codec.pth (FP16)
#   jjackchoi/fish-audio-s2-pro-8bit/codec.pth (FP8 — T4 미지원, 비추)
VOCODER_CKPT = "/kaggle/input/datasets/jjackchoi/fish-speech-s2-pro/codec.pth"

# Voice Cloning 참조 — s2.cpp 레포에 포함된 샘플 음성 사용
# Cell 0에서 클론한 소스 안의 sample/Puck_kor.wav
REFERENCE_WAV = "/kaggle/working/s2.cpp/sample/Puck_kor.wav"
REFERENCE_TEXT = "여러분, 오늘은 정말 놀라운 이야기를 가지고 왔습니다. 우리가 매일 밟고 서 있는 이 땅, 대한민국 말이에요. 인구 5천만 명의 이 작은 나라가 어떻게 세계 최강국 미국을 여러 지표에서 앞서고 있는지, 아직도 모르는 분들이 정말 많습니다. 인터넷 속도는 세계 1위, 의료 수준은 미국을 훌쩍 뛰어넘고, 기대 수명도 더 깁니다. 그런데 정작 한국인들은 이 사실을 잘 모르고 있어요. 숫자는 거짓말을 하지 않습니다. 지금부터 그 놀라운 진실을 하나씩 풀어드릴게요."

BASE_PORT = 3030

# 빌드 결과물 복제 경로 — 이 폴더를 통째로 다운로드하면 됨
DIST_DIR = "/kaggle/working/s2-binary"

# ============================================================
# Cell 0: s2.cpp 소스 빌드 (CUDA) + 바이너리 복제
# ============================================================
import subprocess
import os
import shutil

print("=== s2.cpp CUDA 빌드 시작 ===")

S2CPP_DIR = "/kaggle/working/s2.cpp"
BUILD_DIR = os.path.join(S2CPP_DIR, "build")

# --- libcuda.so stub 링크 (Kaggle 컨테이너에서 CUDA::cuda_driver 해결용) ---
# Kaggle T4 환경에서는 런타임 libcuda.so가 없고 stubs 디렉토리에만 존재.
# CMake FindCUDAToolkit이 CUDA::cuda_driver IMPORTED 타겟을 만들려면
# libcuda.so가 CUDAToolkit_LIBRARY_DIR 안에 있어야 한다.
# 해결: stubs의 libcuda.so를 lib64/에 심볼릭 링크 → lib64를 라이브러리 경로로 지정
CUDA_STUBS_DIR = None
CUDA_LIB64 = "/usr/local/cuda/lib64"
link_dst = os.path.join(CUDA_LIB64, "libcuda.so")

# stubs 디렉토리 탐색
stubs_candidates = [
    "/usr/local/cuda/lib64/stubs",
    "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
]
for d in stubs_candidates:
    if os.path.isdir(d):
        CUDA_STUBS_DIR = d
        break

# stubs에서 못 찾으면 넓은 범위 탐색
if CUDA_STUBS_DIR is None:
    find_r = subprocess.run(
        ["find", "/usr/local/cuda", "-maxdepth", "4", "-name", "libcuda.so*", "-type", "f"],
        capture_output=True, text=True, timeout=15)
    for candidate in find_r.stdout.strip().split("\n"):
        if candidate.strip():
            CUDA_STUBS_DIR = os.path.dirname(candidate.strip())
            break

if CUDA_STUBS_DIR:
    print(f"CUDA stubs dir: {CUDA_STUBS_DIR}")
    # stubs/libcuda.so → lib64/libcuda.so 심볼릭 링크 생성
    stub_libcuda = os.path.join(CUDA_STUBS_DIR, "libcuda.so")
    if os.path.exists(stub_libcuda) and not os.path.exists(link_dst):
        subprocess.run(["ln", "-sf", stub_libcuda, link_dst], check=False)
        print(f"libcuda.so 링크: {stub_libcuda} → {link_dst}")
    # libcuda.so.1 링크도 필요할 수 있음
    link_dst_1 = os.path.join(CUDA_LIB64, "libcuda.so.1")
    if not os.path.exists(link_dst_1):
        subprocess.run(["ln", "-sf", link_dst, link_dst_1], check=False)
else:
    print("경고: CUDA stubs 디렉토리를 찾을 수 없음")

# --- 소스 클론 ---
if not os.path.isdir(S2CPP_DIR):
    subprocess.run([
        "git", "clone", "--recursive", "--depth=1",
        "-b", S2_BRANCH,
        S2_REPO,
        S2CPP_DIR
    ], check=True)
    print("소스 클론 완료")
else:
    print("소스 이미 존재, 클론 생략")

# --- CMake configure + build ---
# Kaggle T4에서는 런타임 libcuda.so가 없어 CUDA::cuda_driver 타겟이 생성되지 않음.
# ggml CMakeLists.txt:
#   if (GGML_CUDA_NO_VMM) → cuda_driver 링크 스킵
#   else() → target_link_libraries(ggml-cuda PRIVATE CUDA::cuda_driver)
# GGML_CUDA_NO_VMM=ON으로 VMM을 비활성화하면 cuda_driver 의존 자체를 제거.
cmake_args = [
    "cmake",
    "-B", BUILD_DIR,
    "-S", S2CPP_DIR,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DS2_CUDA=ON",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DGGML_CUDA_NO_VMM=ON",
    f"-DCUDAToolkit_ROOT=/usr/local/cuda",
]

subprocess.run(cmake_args, check=True)
print("CMake configure 완료")

cpu_count = os.cpu_count() or 4
subprocess.run([
    "cmake", "--build", BUILD_DIR,
    "--config", "Release",
    "--parallel", str(cpu_count),
], check=True)
print("빌드 완료")

BUILD_BINARY = os.path.join(BUILD_DIR, "s2")
assert os.path.isfile(BUILD_BINARY), f"바이너리 없음: {BUILD_BINARY}"

# --- 배포용 디렉토리에 바이너리 + .so 복제 ---
os.makedirs(DIST_DIR, exist_ok=True)

# s2 바이너리 복사
shutil.copy2(BUILD_BINARY, os.path.join(DIST_DIR, "s2"))
os.chmod(os.path.join(DIST_DIR, "s2"), 0o755)


def find_ggml_libs(search_dir):
    """libggml*.so 파일들을 재귀 탐색"""
    libs = []
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if f.startswith("libggml") and ".so" in f:
                libs.append(os.path.join(root, f))
    return libs


def copy_libs(lib_paths, dest_dir):
    """심볼릭 링크를 실제 파일로 풀어서 복사"""
    copied = []
    for lp in lib_paths:
        real = os.path.realpath(lp)
        dest = os.path.join(dest_dir, os.path.basename(lp))
        if not os.path.exists(dest):
            shutil.copy2(real, dest)
            copied.append(os.path.basename(lp))
    return copied


# libggml*.so (빌드 산출물)
build_libs = find_ggml_libs(BUILD_DIR)
if build_libs:
    copied = copy_libs(build_libs, DIST_DIR)
    print(f"libggml .so {len(copied)}개 복사: {copied}")

# --- 바이너리 검증 ---
S2_ENV = os.environ.copy()
S2_ENV["LD_LIBRARY_PATH"] = DIST_DIR + ":" + S2_ENV.get("LD_LIBRARY_PATH", "")


def verify_binary(binary, env=None):
    try:
        r = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=10, env=env)
        return r.returncode == 0
    except Exception:
        return False


S2_BINARY = os.path.join(DIST_DIR, "s2")
assert verify_binary(S2_BINARY, S2_ENV), "빌드된 바이너리 실행 검증 실패!"

size_mb = os.path.getsize(S2_BINARY) / 1024 / 1024
print(f"\n바이너리 준비 완료: {S2_BINARY} ({size_mb:.1f} MB)")

# --- 배포 디렉토리 내용 출력 ---
print(f"\n=== 배포 디렉토리: {DIST_DIR} ===")
print("이 폴더를 통째로 다운로드하면 다른 환경에서도 사용 가능합니다.")
for fname in sorted(os.listdir(DIST_DIR)):
    fsize = os.path.getsize(os.path.join(DIST_DIR, fname)) / 1024 / 1024
    print(f"  {fname} ({fsize:.1f} MB)")

# ============================================================
# Cell 1: GPU 환경 확인
# ============================================================
import torch

print("=== GPU 환경 확인 ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"\nGPU {i}: {props.name}")
    print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute capability: {props.major}.{props.minor}")

for gid in GPU_IDS:
    assert gid < torch.cuda.device_count(), f"GPU {gid} 요청됨, 실제 GPU 수: {torch.cuda.device_count()}"
print(f"\n사용할 GPU: {GPU_IDS}")

# ============================================================
# Cell 2: s2.cpp 서버 기동 (첫 번째 GPU)
# ============================================================
import time
import requests

gpu0 = GPU_IDS[0]
port0 = BASE_PORT

# 서버 로그를 파일로 출력 (PIPE는 버퍼가 차면 프로세스가 블록됨)
SERVER_LOG = "/kaggle/working/s2_server.log"
server_log_f = open(SERVER_LOG, "w")

s2_cmd = [
    S2_BINARY,
    "-m", MODEL,
    "-t", TOKENIZER,
    "-c", str(gpu0),
    "--server",
    "--no-codec",
    "-P", str(port0),
    "-H", "0.0.0.0",
    "--ctx-size", "2048",
]
print(f"실행 명령: {' '.join(s2_cmd)}")

proc = subprocess.Popen(
    s2_cmd,
    stdout=server_log_f,
    stderr=subprocess.STDOUT,  # stderr도 같은 로그 파일로
    env=S2_ENV,
)

print(f"서버 기동 대기 중 (GPU {gpu0}, port {port0})...")
server_ready = False
for i in range(120):  # 모델 로딩이 느릴 수 있으므로 120초
    # 프로세스가 이미 죽었는지 확인
    ret = proc.poll()
    if ret is not None:
        print(f"서버 프로세스 종료됨 (exit code: {ret})")
        break
    try:
        r = requests.get(f"http://127.0.0.1:{port0}/health", timeout=2)
        if r.status_code == 200:
            print(f"서버 준비 완료 ({i+1}초)")
            server_ready = True
            break
    except:
        time.sleep(1)

if not server_ready:
    print("서버 기동 실패 — 로그:")
    server_log_f.flush()
    with open(SERVER_LOG, "r") as f:
        log_content = f.read()
    print(log_content[-3000:] if len(log_content) > 3000 else log_content)

# ============================================================
# Cell 3: /generate_tokens 호출 및 토큰 확인
# ============================================================
import numpy as np

text = "안녕하세요, 이것은 테스트 음성입니다."

resp = requests.post(
    f"http://127.0.0.1:{port0}/generate_tokens",
    files={"text": (None, text)},
    timeout=120
)

print(f"Status: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('Content-Type')}")
print(f"Body size: {len(resp.content)} bytes")

if resp.status_code == 200:
    num_codebooks = int(resp.headers.get("X-Num-Codebooks", 0))
    n_frames = int(resp.headers.get("X-Num-Frames", 0))

    tokens_flat = np.frombuffer(resp.content, dtype=np.int32).copy()

    print(f"\n=== 토큰 분석 ===")
    print(f"Flat shape: {tokens_flat.shape}")
    print(f"Num codebooks: {num_codebooks}")
    print(f"Num frames: {n_frames}")
    print(f"dtype: {tokens_flat.dtype}")
    print(f"Min: {tokens_flat.min()}, Max: {tokens_flat.max()}")

    if num_codebooks > 0 and n_frames > 0:
        tokens = tokens_flat.reshape(num_codebooks, n_frames)
        print(f"Reshaped: {tokens.shape}")
        print(f"First codebook, first 20 frames: {tokens[0, :20]}")
    else:
        tokens = tokens_flat
        print(f"헤더에 shape 정보 없음, flat으로 사용: {tokens.shape}")
        print(f"First 20 tokens: {tokens[:20]}")
else:
    print(f"에러: {resp.text[:500]}")

# ============================================================
# Cell 4: fish-speech 보코더 로드
# ============================================================
# 최신 fish-speech는 vqgan → dac 모듈로 변경됨.
# load_model(config, ckpt, device) → model.from_indices(indices) 로 디코딩.
#
# 주의: fish-speech pyproject.toml이 torch==2.8.0을 요구하지만
# Kaggle은 구버전 torch를 사용하므로 --no-deps로 설치하고
# 추론에 필요한 최소 의존성만 수동 설치한다.
# 또한 fish_speech.utils.__init__.py가 pytorch_lightning을 import하므로
# 훈련용 import를 제거하는 패치가 필요하다.
import sys

FISH_SPEECH_DIR = "/kaggle/working/fish-speech"
if not os.path.isdir(FISH_SPEECH_DIR):
    subprocess.run([
        "git", "clone", "--depth=1",
        "https://github.com/fishaudio/fish-speech",
        FISH_SPEECH_DIR
    ], check=True)

# --- 의존성 설치 ---
print("fish-speech 의존성 설치 중...")

# 시스템 패키지
subprocess.run(["apt-get", "update", "-y"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "--fix-missing",
                 "portaudio19-dev", "libsox-dev", "ffmpeg"], capture_output=True)

# fish-speech를 --no-deps로 설치 (torch==2.8.0 충돌 회피)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", FISH_SPEECH_DIR],
    check=True,
)

# 추론에 필요한 최소 패키지만 설치 (Kaggle 기존 torch 유지)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "hydra-core>=1.3.2", "omegaconf", "loguru", "pyrootutils",
    "descript-audio-codec", "einops", "einx[torch]==0.2.2",
    "click", "soundfile", "natsort", "rich", "safetensors",
], check=True)
print("최소 의존성 설치 완료")

# --- fish_speech/utils/__init__.py 패치 ---
# pytorch_lightning / lightning 의존 import 제거 (추론에 불필요)
utils_init_path = os.path.join(
    FISH_SPEECH_DIR, "fish_speech", "utils", "__init__.py")
if os.path.exists(utils_init_path):
    with open(utils_init_path, "r") as f:
        content = f.read()
    # 훈련용 import를 빈 줄로 대체
    patched = content
    patched = patched.replace(
        "from .instantiators import instantiate_callbacks, instantiate_loggers", "")
    patched = patched.replace(
        "from .logger import RankedLogger", "")
    patched = patched.replace(
        "from .logging_utils import log_hyperparameters", "")
    patched = patched.replace(
        "from .rich_utils import enforce_tags, print_config_tree", "")
    patched = patched.replace(
        "from .utils import extras, get_metric_value, set_seed, task_wrapper", "")
    if patched != content:
        with open(utils_init_path, "w") as f:
            f.write(patched)
        print("utils/__init__.py 패치 완료 (훈련용 import 제거)")

# fish-speech 스코핑 버그 패치
ref_loader_path = os.path.join(
    FISH_SPEECH_DIR, "fish_speech", "inference_engine", "reference_loader.py")
if os.path.exists(ref_loader_path):
    subprocess.run(["sed", "-i",
        "s/import torchaudio.io._load_audio_fileobj/from torchaudio.io import _load_audio_fileobj/",
        ref_loader_path])
    print("스코핑 버그 패치 완료")

sys.path.insert(0, FISH_SPEECH_DIR)

# --- 보코더 로드 ---
assert os.path.isfile(VOCODER_CKPT), f"보코더 체크포인트 없음: {VOCODER_CKPT}"

from fish_speech.models.dac.inference import load_model as load_vocoder

vocoder = load_vocoder("modded_dac_vq", VOCODER_CKPT, device=f"cuda:{gpu0}")
SAMPLE_RATE = vocoder.sample_rate

print(f"보코더 로드 완료 (GPU {gpu0}), sample_rate={SAMPLE_RATE}")

for i in range(torch.cuda.device_count()):
    mem = torch.cuda.memory_allocated(i) / 1024**3
    print(f"GPU {i} VRAM 사용: {mem:.2f} GB")

# ============================================================
# Cell 5: 토큰 → WAV 디코딩 (핵심 검증)
# ============================================================
# fish-speech inference.py 패턴:
#   indices shape: (1, codebooks, frames)  — long 텐서
#   audio = model.from_indices(indices)
import soundfile as sf
from IPython.display import Audio as IPAudio, display

print(f"입력 토큰: shape={tokens.shape}, dtype={tokens.dtype}")

# (codebooks, frames) → (1, codebooks, frames)
token_tensor = torch.from_numpy(tokens).long().unsqueeze(0).to(f"cuda:{gpu0}")
print(f"텐서 shape: {token_tensor.shape}")

try:
    with torch.no_grad():
        audio = vocoder.from_indices(token_tensor)

    # audio shape: (1, 1, samples) or (1, samples)
    audio_np = audio[0, 0].float().cpu().numpy()
    print(f"\n=== 디코딩 결과 ===")
    print(f"Audio shape: {audio_np.shape}")
    print(f"Sample rate: {SAMPLE_RATE}")
    print(f"Duration: {len(audio_np) / SAMPLE_RATE:.2f}초")
    print(f"Min: {audio_np.min():.4f}, Max: {audio_np.max():.4f}")

    sf.write("/kaggle/working/test_output.wav", audio_np, SAMPLE_RATE)
    print("저장 완료: /kaggle/working/test_output.wav")

    display(IPAudio(audio_np, rate=SAMPLE_RATE))

    print("\n토큰 → 오디오 변환 성공!")

except Exception as e:
    print(f"\n디코딩 실패: {e}")
    import traceback; traceback.print_exc()
    print("\n디버깅 정보:")
    print(f"  token_tensor shape: {token_tensor.shape}, dtype: {token_tensor.dtype}")
    print(f"  token 값 범위: min={tokens.min()}, max={tokens.max()}")
    print(f"  vocoder type: {type(vocoder).__name__}")

# ============================================================
# Cell 6: Voice Cloning 토큰 검증
# ============================================================
text = "안녕하세요, 보이스 클로닝 테스트입니다."

if not os.path.isfile(REFERENCE_WAV):
    print(f"참조 음성 파일 없음: {REFERENCE_WAV}")
    print("Voice Cloning 검증을 건너뜁니다. REFERENCE_WAV 경로를 확인하세요.")
else:
    with open(REFERENCE_WAV, "rb") as ref_fh:
        resp = requests.post(
            f"http://127.0.0.1:{port0}/generate_tokens",
            files={
                "text": (None, text),
                "reference": ("reference.wav", ref_fh, "audio/wav"),
                "reference_text": (None, REFERENCE_TEXT),
            },
            timeout=120
        )

    print(f"Status: {resp.status_code}")

    if resp.status_code == 200:
        vc_codebooks = int(resp.headers.get("X-Num-Codebooks", 0))
        vc_frames = int(resp.headers.get("X-Num-Frames", 0))

        tokens_vc_flat = np.frombuffer(resp.content, dtype=np.int32).copy()
        if vc_codebooks > 0 and vc_frames > 0:
            tokens_vc = tokens_vc_flat.reshape(vc_codebooks, vc_frames)
        else:
            tokens_vc = tokens_vc_flat

        print(f"VC 토큰: shape={tokens_vc.shape}")

        token_tensor_vc = torch.from_numpy(tokens_vc).long().unsqueeze(0).to(f"cuda:{gpu0}")
        with torch.no_grad():
            audio = vocoder.from_indices(token_tensor_vc)

        audio_np = audio[0, 0].float().cpu().numpy()
        sf.write("/kaggle/working/test_vc_output.wav", audio_np, SAMPLE_RATE)
        display(IPAudio(audio_np, rate=SAMPLE_RATE))

        print("Voice Cloning 디코딩 성공 — 음색이 참조 음성과 유사한지 귀로 확인하세요")
    else:
        print(f"에러: {resp.text[:500]}")

# ============================================================
# Cell 7: 듀얼 GPU 병렬 검증
# ============================================================
if len(GPU_IDS) < 2:
    print("GPU가 1장이므로 듀얼 GPU 검증을 건너뜁니다.")
else:
    from concurrent.futures import ThreadPoolExecutor

    gpu1 = GPU_IDS[1]
    port1 = BASE_PORT + 1

    SERVER_LOG_1 = "/kaggle/working/s2_server_gpu1.log"
    server_log_f1 = open(SERVER_LOG_1, "w")

    proc1 = subprocess.Popen(
        [S2_BINARY,
         "-m", MODEL,
         "-t", TOKENIZER,
         "-c", str(gpu1),
         "--server",
         "--no-codec",
         "-P", str(port1),
         "-H", "0.0.0.0",
         "--ctx-size", "2048"],
        stdout=server_log_f1,
        stderr=subprocess.STDOUT,
        env=S2_ENV,
    )

    vocoder_1 = load_vocoder("modded_dac_vq", VOCODER_CKPT, device=f"cuda:{gpu1}")

    print(f"GPU {gpu1} 서버 기동 대기 (port {port1})...")
    server1_ready = False
    for i in range(120):
        ret = proc1.poll()
        if ret is not None:
            print(f"GPU {gpu1} 서버 프로세스 종료됨 (exit code: {ret})")
            break
        try:
            r = requests.get(f"http://127.0.0.1:{port1}/health", timeout=2)
            if r.status_code == 200:
                print(f"GPU {gpu1} 서버 준비 완료 ({i+1}초)")
                server1_ready = True
                break
        except:
            time.sleep(1)

    if not server1_ready:
        print("GPU 1 서버 기동 실패 — 로그:")
        server_log_f1.flush()
        with open(SERVER_LOG_1, "r") as f:
            print(f.read()[-2000:])
    else:
        for gid in GPU_IDS[:2]:
            mem = torch.cuda.memory_allocated(gid) / 1024**3
            total = torch.cuda.get_device_properties(gid).total_mem / 1024**3
            print(f"GPU {gid}: {mem:.2f} / {total:.1f} GB")

        vocoders = {gpu0: vocoder, gpu1: vocoder_1}
        ports = {gpu0: port0, gpu1: port1}

        def run_tts(gpu_id, text, output_path):
            """단일 워커의 전체 파이프라인"""
            port = ports[gpu_id]
            start = time.time()

            resp = requests.post(
                f"http://127.0.0.1:{port}/generate_tokens",
                files={"text": (None, text)},
                timeout=120
            )
            cb = int(resp.headers.get("X-Num-Codebooks", 0))
            nf = int(resp.headers.get("X-Num-Frames", 0))
            tok = np.frombuffer(resp.content, dtype=np.int32).copy()
            if cb > 0 and nf > 0:
                tok = tok.reshape(cb, nf)

            with torch.no_grad():
                audio = vocoders[gpu_id].from_indices(
                    torch.from_numpy(tok).long().unsqueeze(0).to(f"cuda:{gpu_id}")
                )

            audio_np = audio[0, 0].float().cpu().numpy()
            sf.write(output_path, audio_np, SAMPLE_RATE)

            elapsed = time.time() - start
            duration = len(audio_np) / SAMPLE_RATE
            return gpu_id, elapsed, duration

        texts = [
            "첫 번째 워커의 테스트 음성입니다.",
            "두 번째 워커에서 동시에 처리하는 음성입니다.",
        ]

        print("\n=== 병렬 실행 시작 ===")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for idx, (gid, txt) in enumerate(zip(GPU_IDS[:2], texts)):
                f = executor.submit(run_tts, gid, txt, f"/kaggle/working/parallel_{idx}.wav")
                futures.append(f)

            for f in futures:
                gid, elapsed, dur = f.result()
                print(f"GPU {gid}: 렌더링 {elapsed:.2f}초, 오디오 {dur:.2f}초")

        print("\n듀얼 GPU 병렬 파이프라인 검증 완료")
