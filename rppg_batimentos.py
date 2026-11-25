"""
Estimativa de frequencia cardiaca (BPM) a partir de video facial - rPPG
(fotopletismografia remota).

Pipeline:
  1. Deteccao do rosto (Haar cascade do OpenCV) e definicao de ROIs de pele:
     testa e bochechas (evita olhos, boca e barba).
  2. Media RGB das ROIs em cada quadro -> sinal bruto RGB(t).
  3. Extracao do sinal de pulso pelos metodos:
       - GREEN  (Verkruysse et al., 2008)
       - CHROM  (de Haan & Jeanne, 2013)
       - POS    (Wang et al., 2017)  <- metodo principal
  4. Filtro passa-banda Butterworth 0,7-3,5 Hz (42-210 BPM).
  5. BPM = pico do espectro (Welch) - no video inteiro e em janelas
     deslizantes (10 s, passo 1 s) para ver a variacao ao longo do tempo.

Uso:
  python3 rppg_batimentos.py                      # processa os dois videos
  python3 rppg_batimentos.py video1.mp4 video2.mp4
  Referencia opcional (oximetro/relogio):  --ref SemMovimento.mp4=65-75

Saidas (pasta resultados/):
  <video>_sinal.png     sinal rPPG filtrado + espectro
  <video>_bpm_tempo.png BPM ao longo do tempo
  <video>_bpm.csv       BPM por janela
  resumo.csv            tabela final para o artigo
"""
import argparse
import csv
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

F_MIN, F_MAX = 0.7, 3.5          # Hz  (42 - 210 BPM)
JANELA_S, PASSO_S = 10.0, 1.0    # janela deslizante
DESCARTE_INICIO_S = 1.0          # ignora o 1o segundo (ajuste de camera)

# Referencias medidas por sensor de contato (ajuste com o seu oximetro/relogio)
REFERENCIAS_PADRAO = {
    "SemMovimento": (65, 75),
    "ComMovimento": (90, 100),
}


# ----------------------------------------------------------------------------
# 1-2. Leitura do video e extracao do sinal RGB
# ----------------------------------------------------------------------------
def rois_do_rosto(x, y, w, h):
    """ROIs relativas a caixa do rosto: testa e bochechas."""
    testa = (x + int(0.30 * w), y + int(0.08 * h), int(0.40 * w), int(0.15 * h))
    boch_e = (x + int(0.15 * w), y + int(0.50 * h), int(0.20 * w), int(0.15 * h))
    boch_d = (x + int(0.65 * w), y + int(0.50 * h), int(0.20 * w), int(0.15 * h))
    return [testa, boch_e, boch_d]


def mascara_pele(roi_bgr):
    """Mascara simples de pele em YCrCb para descartar pixels nao-pele."""
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return m > 0


def extrair_rgb(caminho, salvar_preview=None):
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir {caminho}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    rgb, caixa_suave, n = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detecta em resolucao reduzida (mais rapido)
        esc = 0.5
        faces = detector.detectMultiScale(
            cv2.resize(cinza, None, fx=esc, fy=esc), 1.1, 5, minSize=(60, 60))
        if len(faces):
            f = max(faces, key=lambda r: r[2] * r[3]) / esc
            # suavizacao exponencial da caixa (reduz tremor da ROI)
            caixa_suave = f if caixa_suave is None else 0.85 * caixa_suave + 0.15 * f
        if caixa_suave is None:
            rgb.append([np.nan] * 3)
            continue

        x, y, w, h = caixa_suave.astype(int)
        pixels = []
        for (rx, ry, rw, rh) in rois_do_rosto(x, y, w, h):
            roi = frame[max(ry, 0):ry + rh, max(rx, 0):rx + rw]
            if roi.size == 0:
                continue
            m = mascara_pele(roi)
            pixels.append(roi[m] if m.sum() > 50 else roi.reshape(-1, 3))
        px = np.concatenate(pixels).astype(np.float64)
        b, g, r = px.mean(axis=0)
        rgb.append([r, g, b])

        if salvar_preview and n == 60:
            prev = frame.copy()
            cv2.rectangle(prev, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for (rx, ry, rw, rh) in rois_do_rosto(x, y, w, h):
                cv2.rectangle(prev, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.imwrite(salvar_preview, prev)
    cap.release()

    rgb = np.array(rgb)
    # preenche quadros sem deteccao por interpolacao
    for c in range(3):
        col = rgb[:, c]
        nan = np.isnan(col)
        if nan.all():
            raise RuntimeError("Nenhum rosto detectado no video.")
        col[nan] = np.interp(np.flatnonzero(nan), np.flatnonzero(~nan), col[~nan])
    return rgb, fps


# ----------------------------------------------------------------------------
# 3. Metodos de extracao do pulso
# ----------------------------------------------------------------------------
def metodo_green(rgb, fps):
    return rgb[:, 1] - rgb[:, 1].mean()


def metodo_chrom(rgb, fps):
    win = int(1.6 * fps)
    N = len(rgb)
    h = np.zeros(N)
    b_, a_ = signal.butter(3, [F_MIN, F_MAX], btype="band", fs=fps)
    for s in range(0, N - win, win // 2):
        C = rgb[s:s + win]
        Cn = C / C.mean(axis=0)
        X = 3 * Cn[:, 0] - 2 * Cn[:, 1]
        Y = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]
        Xf, Yf = signal.filtfilt(b_, a_, X, padlen=min(len(X) - 1, 9)), \
                 signal.filtfilt(b_, a_, Y, padlen=min(len(Y) - 1, 9))
        alpha = Xf.std() / (Yf.std() + 1e-9)
        S = (Xf - alpha * Yf) * signal.windows.hann(win)
        h[s:s + win] += S
    return h


def metodo_pos(rgb, fps):
    win = int(1.6 * fps)
    N = len(rgb)
    H = np.zeros(N)
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    for m in range(win, N + 1):
        s = m - win
        C = rgb[s:m].T                       # 3 x win
        Cn = C / C.mean(axis=1, keepdims=True)
        S = P @ Cn
        alpha = S[0].std() / (S[1].std() + 1e-9)
        h = S[0] + alpha * S[1]
        H[s:m] += h - h.mean()
    return H


METODOS = {"POS": metodo_pos, "CHROM": metodo_chrom, "GREEN": metodo_green}


# ----------------------------------------------------------------------------
# 4-5. Filtragem e estimativa de BPM
# ----------------------------------------------------------------------------
def filtrar(x, fps):
    b, a = signal.butter(3, [F_MIN, F_MAX], btype="band", fs=fps)
    x = signal.detrend(x)
    return signal.filtfilt(b, a, x)


def bpm_espectro(x, fps):
    nfft = max(4096, int(2 ** np.ceil(np.log2(len(x)))) * 4)
    f, p = signal.welch(x, fs=fps, nperseg=min(len(x), int(8 * fps)),
                        noverlap=None, nfft=nfft)
    banda = (f >= F_MIN) & (f <= F_MAX)
    f, p = f[banda], p[banda]
    return 60.0 * f[np.argmax(p)], f * 60, p


def bpm_janelas(x, fps):
    jan, passo = int(JANELA_S * fps), int(PASSO_S * fps)
    tempos, bpms = [], []
    for s in range(0, len(x) - jan + 1, passo):
        seg = x[s:s + jan]
        bpm, _, _ = bpm_espectro(seg, fps)
        tempos.append((s + jan / 2) / fps)
        bpms.append(bpm)
    return np.array(tempos), np.array(bpms)


# ----------------------------------------------------------------------------
def processar(caminho, pasta, referencia=None):
    nome = os.path.splitext(os.path.basename(caminho))[0]
    print(f"\n=== {nome} ===")
    rgb, fps = extrair_rgb(caminho, os.path.join(pasta, f"{nome}_roi.jpg"))
    rgb = rgb[int(DESCARTE_INICIO_S * fps):]
    print(f"  FPS: {fps:.2f} | quadros: {len(rgb)} | duracao: {len(rgb)/fps:.1f} s")

    resultados = {}
    for nm, fn in METODOS.items():
        sinal = filtrar(fn(rgb, fps), fps)
        bpm_global, fb, pb = bpm_espectro(sinal, fps)
        t, bpm_t = bpm_janelas(sinal, fps)
        resultados[nm] = dict(sinal=sinal, bpm=bpm_global, fb=fb, pb=pb, t=t, bpm_t=bpm_t)
        print(f"  {nm:6s}: BPM global = {bpm_global:6.1f} | janelas: "
              f"media {bpm_t.mean():.1f}  mediana {np.median(bpm_t):.1f}  "
              f"min {bpm_t.min():.1f}  max {bpm_t.max():.1f}")

    # ---- figuras (metodo principal: POS)
    r = resultados["POS"]
    tt = np.arange(len(r["sinal"])) / fps + DESCARTE_INICIO_S
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(tt, r["sinal"], lw=1)
    ax[0].set(title=f"{nome} - sinal rPPG (POS, filtrado 0,7-3,5 Hz)",
              xlabel="Tempo (s)", ylabel="Amplitude")
    ax[1].plot(r["fb"], r["pb"] / r["pb"].max(), lw=1.5)
    ax[1].axvline(r["bpm"], color="r", ls="--", label=f"pico = {r['bpm']:.1f} BPM")
    if referencia:
        ax[1].axvspan(*referencia, color="g", alpha=0.15,
                      label=f"referencia {referencia[0]}-{referencia[1]} BPM")
    ax[1].set(xlabel="Frequencia (BPM)", ylabel="Potencia (norm.)", xlim=(40, 200))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, f"{nome}_sinal.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for nm, rr in resultados.items():
        ax.plot(rr["t"] + DESCARTE_INICIO_S, rr["bpm_t"], marker="o", ms=3, label=nm)
    if referencia:
        ax.axhspan(*referencia, color="g", alpha=0.15, label="referencia (sensor)")
    ax.set(title=f"{nome} - BPM ao longo do tempo (janela {JANELA_S:.0f} s)",
           xlabel="Tempo (s)", ylabel="BPM", ylim=(40, 160))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, f"{nome}_bpm_tempo.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(pasta, f"{nome}_bpm.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tempo_s"] + [f"bpm_{m}" for m in resultados])
        for i, t in enumerate(r["t"]):
            w.writerow([f"{t + DESCARTE_INICIO_S:.1f}"] +
                       [f"{resultados[m]['bpm_t'][i]:.1f}" for m in resultados])
    return nome, resultados


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*",
                    default=[os.path.join(base, "SemMovimento.mp4"),
                             os.path.join(base, "ComMovimento.mp4")])
    ap.add_argument("--ref", action="append", default=[],
                    help="referencia: NOME=min-max, ex. SemMovimento=65-75")
    ap.add_argument("--saida", default=os.path.join(base, "resultados"))
    a = ap.parse_args()

    refs = dict(REFERENCIAS_PADRAO)
    for item in a.ref:
        k, v = item.split("=")
        lo, hi = v.split("-")
        refs[os.path.splitext(k)[0]] = (float(lo), float(hi))

    os.makedirs(a.saida, exist_ok=True)
    linhas = []
    for v in a.videos:
        nome = os.path.splitext(os.path.basename(v))[0]
        ref = refs.get(nome)
        nome, res = processar(v, a.saida, ref)
        for m, r in res.items():
            linha = dict(video=nome, metodo=m, bpm_global=round(r["bpm"], 1),
                         bpm_media=round(r["bpm_t"].mean(), 1),
                         bpm_min=round(r["bpm_t"].min(), 1),
                         bpm_max=round(r["bpm_t"].max(), 1),
                         ref_min=ref[0] if ref else "", ref_max=ref[1] if ref else "")
            if ref:
                centro = (ref[0] + ref[1]) / 2
                linha["erro_abs_vs_centro_ref"] = round(abs(r["bpm"] - centro), 1)
                linha["dentro_da_faixa"] = ref[0] <= r["bpm"] <= ref[1]
            linhas.append(linha)

    campos = ["video", "metodo", "bpm_global", "bpm_media", "bpm_min", "bpm_max",
              "ref_min", "ref_max", "erro_abs_vs_centro_ref", "dentro_da_faixa"]
    with open(os.path.join(a.saida, "resumo.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    print(f"\nResultados salvos em: {a.saida}")


if __name__ == "__main__":
    sys.exit(main())
