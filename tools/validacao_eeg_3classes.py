#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aquisição standalone de MI (Imaginação Motora) 3 classes: mão DIREITA, mão ESQUERDA e REPOUSO.
Validação de hardware g.Nautilus - Gate C do plano de validação.

Protocolo: MI-only (sem ME, sem câmeras, sem animação, sem Brunel).
- 32 canais, 500 Hz via g.Nautilus/gpype
- Fonte 'generator' para teste sem hardware (EEG sintético com classes conhecidas)
- 3 classes balanceadas, ordem randomizada por bloco
- Marcadores claros em CSV (coluna Marker) e JSON de eventos
- Verificação de impedância opcional (flag) sem quebrar fonte generator
- Encerramento seguro com Ctrl+C/ESC
- Modo headless para teste automatizado (sem janela Qt)

Uso típico (hardware real):
    python tools/validacao_eeg_3classes.py --source gnautilus --blocos 2 --trials-por-classe 15

Teste rápido sem hardware (generator):
    python tools/validacao_eeg_3classes.py --source generator --blocos 1 --trials-por-classe 2 \\
        --mi-sec 2 --repouso-sec 1 --cue-sec 1 --pasta-saida ./teste_sintetico --sem-janela
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# gpype imports
import gpype as gp
from gpype.common.constants import Constants
from gpype.backend.filters.bandpass import Bandpass
from gpype.backend.filters.bandstop import Bandstop
from gpype.backend.sinks.csv_writer import CsvWriter
from gpype.backend.sources.g_nautilus import GNautilus
from gpype.backend.sources.generator import Generator
from gpype.backend.sources.marker import Marker
from gpype.common._private import channels
from gpype.common.launch_config import LaunchConfig
from gpype.backend.core.io_node import IONode

# Tentativa de importar PySide6 para UI opcional
try:
    from PySide6.QtCore import QObject, QTimer, Qt
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False

# =============================================================================
# CONSTANTES DO PROTOCOLO DE VALIDAÇÃO (MI 3 classes)
# =============================================================================

FS = 500.0                 # Taxa de amostragem do g.Nautilus (Hz)
PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT
CHANNEL_COUNT = 32         # Canais de EEG
FRAME_SIZE = 1             # Amostras por frame (exigido pelo muxer)

# Códigos de marcador (compatível com BCI Competition IV 2a estendido)
CODE_SESSION_START = 32766
CODE_SESSION_END = 32765
CODE_BLOCK_START = 790
CODE_BLOCK_END = 791
CODE_CUE_LEFT = 769        # Cue mão ESQUERDA
CODE_CUE_RIGHT = 770       # Cue mão DIREITA
CODE_CUE_REST = 784        # Cue REPOUSO (baseline repouso ativo)
CODE_ME_START = 778        # Início da execução motora
CODE_ME_END = 779          # Fim da execução motora
CODE_MI_START = 780        # Início fase MI
CODE_MI_END = 781          # Fim fase MI
CODE_TRIAL_CANCEL = 796    # Trial cancelado
CODE_IMPEDANCE_CHECK = 897 # Verificação de impedância

# Nomes das classes
CLASS_NAMES = ["esquerda", "direita", "repouso"]
CLASS_CODES = {
    "esquerda": CODE_CUE_LEFT,
    "direita": CODE_CUE_RIGHT,
    "repouso": CODE_CUE_REST,
}

# Configuração do gerador sintético (para teste sem hardware)
GEN_DEFAULTS = {
    "sampling_rate": FS,
    "channel_count": CHANNEL_COUNT,
    "frame_size": FRAME_SIZE,
    "signal_frequency": 10.0,    # Alpha ritmo base
    "signal_shape": "sine",
    "signal_amplitude": 10.0,    # µV - amplitude do sinal base
    "noise_amplitude": 5.0,      # µV - ruído de fundo
}

# Padrões específicos por classe para dados sintéticos separáveis
# Cada classe tem assinatura espectral distinta nos canais motores (C3/C4)
SYNTHETIC_CLASS_PATTERNS = {
    "esquerda": {   # ERD em C4 (hemisfério direito) -> imaginação mão esquerda
        "mod_channels": [3, 4],      # Canais aproximados C4/CP4 (0-indexed)
        "mod_freq": 12.0,            # Beta ritmo
        "mod_depth": 0.6,            # Supressão 60% (ERD)
    },
    "direita": {    # ERD em C3 (hemisfério esquerdo) -> imaginação mão direita
        "mod_channels": [1, 2],      # Canais aproximados C3/CP3 (0-indexed)
        "mod_freq": 12.0,
        "mod_depth": 0.6,
    },
    "repouso": {    # Sem modulação ERD/ERS específica
        "mod_channels": [],
        "mod_freq": 0.0,
        "mod_depth": 0.0,
    },
}

# =============================================================================
# GERADOR DE EEG SINTÉTICO COM CLASSES CONHECIDAS
# =============================================================================

class SyntheticEEGGenerator:
    """
    Gera EEG sintético 32 canais com 3 classes MI separáveis.
    
    Baseado no generator do gpype, mas com modulação por classe para criar
    padrões ERD/ERS realistas nos canais motores (C3/C4).
    """
    
    def __init__(
        self,
        sampling_rate: float = FS,
        channel_count: int = CHANNEL_COUNT,
        frame_size: int = FRAME_SIZE,
        base_amplitude: float = 10.0,
        noise_amplitude: float = 5.0,
        seed: Optional[int] = None,
    ):
        self.fs = sampling_rate
        self.n_channels = channel_count
        self.frame_size = frame_size
        self.base_amplitude = base_amplitude
        self.noise_amplitude = noise_amplitude
        self.rng = np.random.default_rng(seed)
        self._time = 0.0
        
        # Montagem 10-20 aproximada (32 canais padrão g.Nautilus)
        self.channel_names = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
            "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
            "P8", "PO7", "PO3", "PO4", "PO8", "O1", "Oz", "O2"
        ]
        
        # Índices dos canais motores principais
        self.c3_idx = self.channel_names.index("C3") if "C3" in self.channel_names else 12
        self.c4_idx = self.channel_names.index("C4") if "C4" in self.channel_names else 14
        self.cz_idx = self.channel_names.index("Cz") if "Cz" in self.channel_names else 13
        
    def generate_block(
        self,
        n_samples: int,
        class_label: str,
        mi_active: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera um bloco de EEG sintético.
        
        Args:
            n_samples: Número de amostras
            class_label: 'esquerda', 'direita' ou 'repouso'
            mi_active: Se True, aplica modulação ERD/ERS durante MI
            
        Returns:
            (data, timestamps) - data shape (n_samples, n_channels), timestamps shape (n_samples,)
        """
        dt = 1.0 / self.fs
        t = np.linspace(self._time, self._time + (n_samples - 1) * dt, n_samples)
        self._time += n_samples * dt
        
        # Sinal base: ruído colorido + ritmo alpha (8-12 Hz) em todos canais
        data = np.zeros((n_samples, self.n_channels), dtype=np.float64)
        
        # Alpha global (8-12 Hz) - mais forte em occipital
        alpha_freq = 10.0
        for ch in range(self.n_channels):
            # Amplitude varia por região
            if ch >= 26:  # Occipital
                amp = self.base_amplitude * 2.0
            elif ch >= 20:  # Parietal
                amp = self.base_amplitude * 1.5
            elif ch >= 8:  # Central/Frontal
                amp = self.base_amplitude * 1.0
            else:  # Frontal
                amp = self.base_amplitude * 0.8
            data[:, ch] += amp * np.sin(2 * np.pi * alpha_freq * t + self.rng.uniform(0, 2*np.pi))
        
        # Ruído de fundo 1/f
        for ch in range(self.n_channels):
            noise = self.rng.standard_normal(n_samples) * self.noise_amplitude
            # Filtro simples 1/f via acumulação
            noise = np.cumsum(noise) * 0.01
            data[:, ch] += noise
        
        # Modulação específica da classe durante MI (ERD/ERS)
        if mi_active and class_label in SYNTHETIC_CLASS_PATTERNS:
            pattern = SYNTHETIC_CLASS_PATTERNS[class_label]
            for ch_idx in pattern["mod_channels"]:
                if ch_idx < self.n_channels:
                    mod_signal = pattern["mod_depth"] * np.sin(2 * np.pi * pattern["mod_freq"] * t)
                    # ERD = supressão de potência (multiplicativo < 1)
                    data[:, ch_idx] *= (1.0 - 0.5 * (1.0 + mod_signal))
        
        timestamps = t.copy()
        return data.astype(np.float32), timestamps.astype(np.float64)
    
    def reset_time(self):
        self._time = 0.0


# =============================================================================
# PIPELINE GPYPE PARA AQUISIÇÃO
# =============================================================================

class MarkerMuxer(IONode):
    """Acrescenta uma coluna Marker sincrona ao EEG (frame_size=1).

    O Marker do gpype e' assincrono e nao conecta direto no CsvWriter; este
    muxer recebe eventos de uma queue e fixa o codigo na amostra atual. E' o
    mesmo principio do RecordingMuxer do paradigma principal, sem movimento.
    """

    def __init__(self, marker_queue, hold_samples=1, **kwargs):
        super().__init__(**kwargs)
        self._queue = marker_queue
        self._hold = max(1, int(hold_samples))
        self._marker = 0
        self._remaining = 0
        self._n_in = None

    def setup(self, data, port_context_in):
        ctx = port_context_in[PORT_IN]
        self._n_in = int(ctx[Constants.Keys.CHANNEL_COUNT])
        out = super().setup(data, port_context_in)
        for context in out.values():
            context[Constants.Keys.CHANNEL_COUNT] = self._n_in + 1
        return out

    def step(self, data):
        block = data.get(PORT_IN)
        if block is None or len(block) == 0:
            return None
        n = block.shape[0]
        try:
            while True:
                ev = self._queue.get_nowait()
                self._marker = int(ev["code"])
                self._remaining = self._hold
        except queue.Empty:
            pass
        col = np.zeros((n, 1), dtype=block.dtype)
        for i in range(n):
            if self._remaining > 0:
                col[i, 0] = self._marker
                self._remaining -= 1
                if self._remaining == 0:
                    self._marker = 0
        return {PORT_OUT: np.column_stack((block, col))}


class AcquisitionPipeline:
    """Gerencia pipeline gpype para aquisição EEG + marcadores -> CSV."""
    
    def __init__(
        self,
        source: str,
        output_csv: Path,
        channel_count: int = CHANNEL_COUNT,
        sampling_rate: float = FS,
        frame_size: int = FRAME_SIZE,
        notch_freq: float = 60.0,
        bp_low: float = 0.5,
        bp_high: float = 40.0,
        enable_oscar: bool = False,
    ):
        self.source_type = source
        self.output_csv = output_csv
        self.channel_count = channel_count
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.notch_freq = notch_freq
        self.bp_low = bp_low
        self.bp_high = bp_high
        self.enable_oscar = enable_oscar
        
        self.pipeline: Optional[gp.Pipeline] = None
        self.source_node = None
        self.marker_node = None
        self.csv_writer = None
        self._running = False
        self._synthetic_gen = SyntheticEEGGenerator(
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
        ) if source == "generator" else None
        
    def build(self):
        """Constrói o pipeline gpype."""
        self.pipeline = gp.Pipeline()
        
        # --- Source (g.Nautilus ou Generator) ---
        if self.source_type == "gnautilus":
            self.source_node = GNautilus(
                sampling_rate=self.sampling_rate,
                channel_count=self.channel_count,
                frame_size=self.frame_size,
                enable_oscar=self.enable_oscar,
            )
        elif self.source_type == "generator":
            self.source_node = Generator(
                sampling_rate=self.sampling_rate,
                channel_count=self.channel_count,
                frame_size=self.frame_size,
                signal_frequency=GEN_DEFAULTS["signal_frequency"],
                signal_shape=GEN_DEFAULTS["signal_shape"],
                signal_amplitude=GEN_DEFAULTS["signal_amplitude"],
                noise_amplitude=GEN_DEFAULTS["noise_amplitude"],
            )
        else:
            raise ValueError(f"Fonte desconhecida: {self.source_type}")
        
        # --- Filtros ---
        # Bandpass 0.5-40 Hz
        bandpass = Bandpass(f_lo=self.bp_low, f_hi=self.bp_high, order=4, phase="causal")
        # Notch 50/60 Hz
        notch = Bandstop(f_lo=self.notch_freq - 1, f_hi=self.notch_freq + 1, order=4, phase="causal")
        
        # --- Marker queue/muxer (sincrono com EEG) ---
        self.marker_codes = {
            "session_start": CODE_SESSION_START,
            "session_end": CODE_SESSION_END,
            "block_start": CODE_BLOCK_START,
            "block_end": CODE_BLOCK_END,
            "cue_left": CODE_CUE_LEFT,
            "cue_right": CODE_CUE_RIGHT,
            "cue_rest": CODE_CUE_REST,
            "me_start": CODE_ME_START,
            "me_end": CODE_ME_END,
            "mi_start": CODE_MI_START,
            "mi_end": CODE_MI_END,
            "trial_cancel": CODE_TRIAL_CANCEL,
            "impedance_check": CODE_IMPEDANCE_CHECK,
        }
        self._marker_queue = queue.Queue()
        self.marker_node = MarkerMuxer(self._marker_queue,
                                       hold_samples=1)
        self.csv_writer = CsvWriter(file_name=str(self.output_csv))
        
        # --- Conexões ---
        # Source -> Bandpass -> Notch -> MarkerMuxer -> CsvWriter
        self.pipeline.connect(self.source_node, bandpass)
        self.pipeline.connect(bandpass, notch)
        self.pipeline.connect(notch, self.marker_node)
        self.pipeline.connect(self.marker_node, self.csv_writer)
        
        return self.pipeline
    
    def start(self):
        """Inicia o pipeline."""
        if self.pipeline is None:
            self.build()
        self.pipeline.start()
        self._running = True
        
        # Se generator, inicia thread de injeção de dados sintéticos
        if self.source_type == "generator" and self._synthetic_gen:
            self._start_synthetic_injection()
    
    def _start_synthetic_injection(self):
        """Thread para injetar dados sintéticos no generator via callback."""
        # O generator do gpype roda no próprio pipeline; para dados customizados
        # por classe, precisamos substituir o callback interno.
        # Como o Generator é um chain, acessamos o core via internal_nodes.
        # Para simplificar, usamos o generator padrão e modulamos via marker timing
        # na análise offline. OU: sobrescrevemos o step do core.
        pass  # O generator padrão já produz ruído; a separação virá na análise
    
    def emit_marker(self, code_name: str):
        """Emite um marcador pelo nome."""
        if self._running:
            code = self.marker_codes[code_name]
            self._marker_queue.put({"code": code, "nome": code_name})
    
    def emit_marker_code(self, code: int):
        """Emite um marcador pelo código numérico."""
        if self._running:
            self._marker_queue.put({"code": int(code), "nome": str(code)})
    
    def stop(self, timeout: float = 5.0):
        """Para o pipeline graciosamente."""
        self._running = False
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"[AVISO] Erro ao parar pipeline: {e}")
            # Aguarda finalização
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                # Pipeline não tem atributo state público; apenas aguarda
                time.sleep(0.05)


# =============================================================================
# CONTROLADOR DO PARADIGMA MI 3 CLASSES
# =============================================================================

class MI3ClassParadigm:
    """
    Controla o paradigma MI 3 classes (esquerda, direita, repouso).
    
    Sequência por trial:
    1. CUE visual (0.5-2s): mostra direção (esquerda/direita) ou "REPOUSO"
    2. MI / REPOUSO (configurável, ex: 4-8s): participante imagina movimento ou repousa
    3. Pausa curta (ITI, 0.5-1.5s): tela neutra
    
    Blocos: ordem das classes randomizada por bloco, balanceada.
    """
    
    def __init__(
        self,
        pipeline: AcquisitionPipeline,
        n_blocks: int = 2,
        trials_per_class: int = 15,
        mi_sec: float = 4.0,
        me_sec: float = 4.0,
        transicao_sec: float = 1.0,
        rest_sec: float = 2.0,
        cue_sec: float = 1.0,
        iti_min_sec: float = 0.5,
        iti_max_sec: float = 1.5,
        output_dir: Path = Path("."),
        subject_id: str = "S01",
        session_id: str = "01",
        headless: bool = False,
        check_impedance: bool = False,
        impedance_limit_kohm: float = 50.0,
    ):
        self.pipeline = pipeline
        self.n_blocks = n_blocks
        self.trials_per_class = trials_per_class
        self.mi_sec = mi_sec
        self.me_sec = float(me_sec)
        self.transicao_sec = float(transicao_sec)
        self.rest_sec = rest_sec
        self.cue_sec = cue_sec
        self.iti_min_sec = iti_min_sec
        self.iti_max_sec = iti_max_sec
        self.output_dir = Path(output_dir)
        self.subject_id = subject_id
        self.session_id = session_id
        self.headless = headless
        self.check_impedance = check_impedance
        self.impedance_limit_kohm = impedance_limit_kohm
        
        self._running = False
        self._aborted = False
        self._paused = False
        self._current_trial = 0
        self._total_trials = n_blocks * trials_per_class * 3
        
        # Dados para JSON de eventos
        self.events = []
        self.block_orders = []  # Ordem das classes por bloco
        
        # UI opcional
        self.app = None
        self.window = None
        self.label = None
        self.timer = None
        
        # Thread de controle
        self._control_thread = None
        
        # Callbacks de hotkey
        self._hotkey_abort = False
    
    def _setup_hotkeys(self):
        """Configura hotkeys globais (ESC para abortar)."""
        # Implementação simplificada sem hotkeys globais complexas
        # Em headless, usamos apenas Ctrl+C via signal
        pass
    
    def _signal_handler(self, signum, frame):
        """Handler para Ctrl+C."""
        print("\n[SINAL] Ctrl+C recebido - abortando graciosamente...")
        self._aborted = True
        self._hotkey_abort = True
    
    def run(self) -> Dict[str, Any]:
        """Executa o paradigma completo."""
        # Configura handlers de sinal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Gera ordens dos blocos (randomizado por bloco, balanceado)
        self._generate_block_orders()
        
        # Inicia pipeline
        print("[INFO] Iniciando pipeline de aquisição...")
        self.pipeline.start()
        time.sleep(1.0)  # Estabilização
        
        # Marca início da sessão
        self.pipeline.emit_marker("session_start")
        self._log_event("session_start", {"subject": self.subject_id, "session": self.session_id})
        
        # Verificação de impedância opcional (apenas g.Nautilus real)
        if self.check_impedance and self.pipeline.source_type == "gnautilus":
            self._run_impedance_check()
        
        # UI se não headless
        if not self.headless and PYSIDE6_AVAILABLE:
            self._setup_ui()
        
        # Loop principal em thread separada (para não bloquear UI)
        self._running = True
        self._control_thread = threading.Thread(target=self._run_blocks, daemon=True)
        self._control_thread.start()
        
        # Loop de eventos (UI ou wait)
        if not self.headless and PYSIDE6_AVAILABLE and self.app:
            self.app.exec()
        else:
            # Headless: aguarda thread terminar
            while self._running and not self._aborted:
                time.sleep(0.1)
        
        # Finalização
        return self._finalize()
    
    def _generate_block_orders(self):
        """Gera ordem aleatória das classes para cada bloco (balanceado)."""
        classes = CLASS_NAMES.copy()
        for b in range(self.n_blocks):
            random.shuffle(classes)
            block_trials = []
            for cls in classes:
                block_trials.extend([cls] * self.trials_per_class)
            random.shuffle(block_trials)  # Embaralha dentro do bloco
            self.block_orders.append(block_trials)
    
    def _run_impedance_check(self):
        """Executa verificação de impedância (apenas hardware real)."""
        print("[IMPEDÂNCIA] Verificando impedâncias dos eletrodos...")
        self.pipeline.emit_marker("impedance_check")
        self._log_event("impedance_check", {})
        
        # TODO: Integrar com gtec_gds para leitura real
        # Por enquanto, apenas simula
        time.sleep(2.0)
        print("[IMPEDÂNCIA] Verificação concluída (simulada).")
    
    def _setup_ui(self):
        """Configura interface visual simples (Qt)."""
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle(f"Validação EEG 3 Classes - {self.subject_id}_s{self.session_id}")
        self.window.resize(800, 600)
        
        central = QWidget()
        layout = QVBoxLayout(central)
        
        self.label = QLabel("PREPARANDO...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 48px; font-weight: bold;")
        layout.addWidget(self.label)
        
        self.window.setCentralWidget(central)
        self.window.showFullScreen()
        
        # Timer para atualização da UI
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_ui)
        self.timer.start(50)  # 20 Hz
    
    def _update_ui(self):
        """Atualiza a UI (chamado pelo timer)."""
        if not self._running or self._aborted:
            if self.app:
                self.app.quit()
    
    def _run_blocks(self):
        """Loop principal dos blocos (roda em thread separada)."""
        try:
            # Contagem inicial para o participante se acomodar (sem tecla).
            if not self.headless:
                for n in (5, 4, 3, 2, 1):
                    self._show_text(f"PREPARANDO\n\n{n}", duration=1.0)
                    if self._aborted:
                        break
            for block_idx in range(self.n_blocks):
                if self._aborted:
                    break
                
                print(f"\n=== BLOCO {block_idx + 1}/{self.n_blocks} ===")
                self.pipeline.emit_marker("block_start")
                self._log_event("block_start", {"block": block_idx + 1, "order": self.block_orders[block_idx]})
                
                if not self.headless:
                    self._show_text(f"BLOCO {block_idx + 1}/{self.n_blocks}\nIniciando...", duration=1.0)
                
                # Trials do bloco
                block_trials = self.block_orders[block_idx]
                for trial_idx, class_label in enumerate(block_trials):
                    if self._aborted:
                        break
                    
                    self._current_trial += 1
                    self._run_trial(block_idx, trial_idx, class_label, block_trials)
                    
                    # ITI entre trials
                    if not self._aborted and trial_idx < len(block_trials) - 1:
                        iti = random.uniform(self.iti_min_sec, self.iti_max_sec)
                        self._wait_with_abort_check(iti)
                
                # Fim do bloco
                self.pipeline.emit_marker("block_end")
                self._log_event("block_end", {"block": block_idx + 1})
                
                # Pausa entre blocos (exceto último)
                if block_idx < self.n_blocks - 1 and not self._aborted:
                    pause_sec = 10.0  # pausa curta, sem tecla; ESC/Ctrl+C aborta
                    self._show_text(f"FIM DO BLOCO {block_idx + 1}\n"
                                    f"Pausa de {pause_sec:.0f}s", duration=pause_sec)
            
        except Exception as e:
            print(f"[ERRO] Exceção no loop de blocos: {e}")
            import traceback
            traceback.print_exc()
            self._aborted = True
        finally:
            self._running = False
    
    def _run_trial(self, block_idx: int, trial_idx: int, class_label: str, block_trials: List[str]):
        """Executa um único trial."""
        trial_num = self._current_trial
        cue_code = CLASS_CODES[class_label]
        
        print(f"  Trial {trial_num}/{self._total_trials}: {class_label.upper()} (código {cue_code})")
        
        # Log do trial
        trial_info = {
            "trial": trial_num,
            "block": block_idx + 1,
            "trial_in_block": trial_idx + 1,
            "class": class_label,
            "cue_code": cue_code,
            "mi_sec": self.mi_sec,
            "me_sec": self.me_sec,
            "transicao_sec": self.transicao_sec,
            "rest_sec": self.rest_sec,
            "cue_sec": self.cue_sec,
        }
        self._log_event("trial_start", trial_info)
        
        # 1. CUE: o marcador marca o ONSET do cue; depois a tela fica pelo
        # tempo do cue. Antes o marcador era emitido DEPOIS do cue, o que
        # desalinhava a analise em relacao ao estimulo visual.
        self.pipeline.emit_marker_code(cue_code)
        self._log_event("cue", {"class": class_label, "code": cue_code})
        if not self.headless:
            cue_text = self._get_cue_text(class_label)
            self._show_text(cue_text, duration=self.cue_sec)
        else:
            self._wait_with_abort_check(self.cue_sec)
        
        # 2. ME (execução motora) / REPOUSO
        if class_label == "repouso":
            texto_me = "REPOUSO\n\nNão se mova"
        else:
            texto_me = "EXECUTE\n\nMão " + class_label.upper()
        self.pipeline.emit_marker("me_start")
        self._log_event("me_start", {"class": class_label})
        if not self.headless:
            self._show_text(texto_me, duration=self.me_sec)
        else:
            self._wait_with_abort_check(self.me_sec)
        self.pipeline.emit_marker("me_end")
        self._log_event("me_end", {"class": class_label})
        
        # 3. Transição curta (neutra)
        if not self.headless:
            self._show_text("+", duration=self.transicao_sec)
        else:
            self._wait_with_abort_check(self.transicao_sec)
        
        # 4. MI / REPOUSO
        if class_label == "repouso":
            texto_mi = "REPOUSO\n\nContinue parado"
        else:
            texto_mi = "IMAGINE\n\nMão " + class_label.upper()
        self.pipeline.emit_marker("mi_start")
        self._log_event("mi_start", {"class": class_label})
        if not self.headless:
            self._show_text(texto_mi, duration=self.mi_sec)
        else:
            self._wait_with_abort_check(self.mi_sec)
        self.pipeline.emit_marker("mi_end")
        self._log_event("mi_end", {"class": class_label})
        
        # 5. Pausa curta (ITI) - tela neutra
        if not self.headless:
            self._show_text("+", duration=0.5)
        
        self._log_event("trial_end", {"trial": trial_num, "class": class_label})
    
    def _get_cue_text(self, class_label: str) -> str:
        """Retorna texto do cue visual."""
        if class_label == "esquerda":
            return "◀  MÃO ESQUERDA  ▶"
        elif class_label == "direita":
            return "◀  MÃO DIREITA  ▶"
        else:  # repouso
            return "○  REPOUSO  ○"
    
    def _show_text(self, text: str, duration: Optional[float] = None, wait_key: bool = False):
        """Mostra texto na UI (ou aguarda se headless)."""
        if self.headless or not self.label:
            if duration:
                self._wait_with_abort_check(duration)
            return
        
        self.label.setText(text)
        self.app.processEvents()
        
        if wait_key:
            # Aguarda tecla (simplificado - em produção usaria event filter)
            print(f"[UI] {text} - aguardando tecla...")
            while self._running and not self._aborted:
                self.app.processEvents()
                time.sleep(0.05)
        elif duration:
            self._wait_with_abort_check(duration)
    
    def _wait_with_abort_check(self, seconds: float):
        """Aguarda com verificação de abort."""
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time and not self._aborted:
            time.sleep(0.05)
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Registra evento para JSON."""
        self.events.append({
            "timestamp": time.time(),
            "monotonic": time.monotonic(),
            "type": event_type,
            "data": data,
        })
    
    def _finalize(self) -> Dict[str, Any]:
        """Finaliza aquisição e salva metadados."""
        print("\n[INFO] Finalizando aquisição...")
        
        # Marca fim da sessão
        self.pipeline.emit_marker("session_end")
        self._log_event("session_end", {})
        
        # Para pipeline
        self.pipeline.stop()
        
        # Salva JSON de eventos
        json_path = self.output_dir / f"eventos_{self.subject_id}_s{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        metadata = {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "date": datetime.now().isoformat(),
            "protocol": "MI_3classes_validation",
            "version": "1.0",
            "source": self.pipeline.source_type,
            "fs": FS,
            "channels": CHANNEL_COUNT,
            "channel_names": SyntheticEEGGenerator().channel_names,
            "n_blocks": self.n_blocks,
            "trials_per_class": self.trials_per_class,
            "total_trials": self._total_trials,
            "mi_sec": self.mi_sec,
            "me_sec": self.me_sec,
            "transicao_sec": self.transicao_sec,
            "rest_sec": self.rest_sec,
            "cue_sec": self.cue_sec,
            "iti_range": [self.iti_min_sec, self.iti_max_sec],
            "classes": CLASS_NAMES,
            "class_codes": CLASS_CODES,
            "marker_codes": {
                "session_start": CODE_SESSION_START,
                "session_end": CODE_SESSION_END,
                "block_start": CODE_BLOCK_START,
                "block_end": CODE_BLOCK_END,
                "cue_left": CODE_CUE_LEFT,
                "cue_right": CODE_CUE_RIGHT,
                "cue_rest": CODE_CUE_REST,
                "me_start": CODE_ME_START,
                "me_end": CODE_ME_END,
                "mi_start": CODE_MI_START,
                "mi_end": CODE_MI_END,
                "trial_cancel": CODE_TRIAL_CANCEL,
                "impedance_check": CODE_IMPEDANCE_CHECK,
            },
            "block_orders": self.block_orders,
            "events": self.events,
            "aborted": self._aborted,
            "completed_trials": self._current_trial,
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] JSON de eventos salvo em: {json_path}")
        print(f"[OK] CSV de EEG salvo em: {self.pipeline.output_csv}")
        print(f"[RESUMO] Trials completados: {self._current_trial}/{self._total_trials}")
        
        return {
            "csv_path": str(self.pipeline.output_csv),
            "json_path": str(json_path),
            "completed_trials": self._current_trial,
            "total_trials": self._total_trials,
            "aborted": self._aborted,
        }


# =============================================================================
# MAIN - ENTRY POINT
# =============================================================================

# =============================================================================
# MODO GENERATOR DIRETO (offline, sem gpype)
# =============================================================================

def _sync_generator_header(n_channels: int) -> list:
    nomes = SyntheticEEGGenerator().channel_names
    if len(nomes) < n_channels:
        nomes += [f"E{i+1}" for i in range(len(nomes), n_channels)]
    return nomes[:n_channels]


def run_generator_offline(args) -> int:
    """Gera um dataset sintético completo, com marcadores no CSV, sem hardware.

    O classificador espera: time, EEG0..EEG31, Marker. Gerar direto evita o
    problema do CsvWriter do gpype (timestamp no fim e marcador fora do CSV).
    """
    import csv
    random.seed(42)
    np.random.seed(42)
    out = Path(args.pasta_saida)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out / f"eeg_mi3c_{args.sujeito}_s{args.sessao}_{stamp}.csv"
    json_path = out / f"eventos_{args.sujeito}_s{args.sessao}_{stamp}.json"

    nomes = _sync_generator_header(CHANNEL_COUNT)
    gen = SyntheticEEGGenerator(sampling_rate=FS, channel_count=CHANNEL_COUNT,
                                frame_size=1, seed=42)
    classes = CLASS_NAMES.copy()
    bloco_ordens = []
    for _ in range(args.blocos):
        ordem = []
        for cls in classes:
            ordem.extend([cls] * args.trials_por_classe)
        random.shuffle(ordem)
        bloco_ordens.append(ordem)

    linhas = []
    eventos = []
    t = 0.0
    sample = 0

    def escreve_bloco(dados, marcador):
        nonlocal t, sample
        for i, amostra in enumerate(dados):
            linhas.append((f"{t:.6f}", *[f"{v:.6f}" for v in amostra], marcador if i == 0 else 0))
            t += 1.0 / FS
            sample += 1

    # sessão e blocos
    escreve_bloco(np.zeros((1, CHANNEL_COUNT)), CODE_SESSION_START)
    eventos.append({"type": "session_start", "sample": 0, "code": CODE_SESSION_START})
    for bi, ordem in enumerate(bloco_ordens):
        escreve_bloco(np.zeros((1, CHANNEL_COUNT)), CODE_BLOCK_START)
        eventos.append({"type": "block_start", "block": bi + 1, "sample": sample - 1,
                        "code": CODE_BLOCK_START})
        for cls in ordem:
            # cue
            cue = CLASS_CODES[cls]
            escreve_bloco(np.zeros((int(args.cue_sec * FS), CHANNEL_COUNT)), cue)
            eventos.append({"type": "cue", "class": cls, "code": cue,
                            "sample": sample - int(args.cue_sec * FS)})
            # ME (execução) — mesma assinatura sintética, fase separada
            n_me = 0 if getattr(args, "sem_me", False) else int(args.me_sec * FS)
            n_mi = int(args.mi_sec * FS)
            tt_me = np.arange(n_me) / FS
            c3 = gen.channel_names.index("C3")
            c4 = gen.channel_names.index("C4")
            oz = gen.channel_names.index("Oz")
            if n_me > 0:
                dados_me, _ = gen.generate_block(n_me, cls, mi_active=True)
                if cls == "direita":
                    dados_me[:, c3] += 40.0 * np.sin(2 * np.pi * 12.0 * tt_me)
                    dados_me[:, c4] += 5.0 * np.sin(2 * np.pi * 12.0 * tt_me)
                elif cls == "esquerda":
                    dados_me[:, c4] += 40.0 * np.sin(2 * np.pi * 12.0 * tt_me)
                    dados_me[:, c3] += 5.0 * np.sin(2 * np.pi * 12.0 * tt_me)
                else:
                    dados_me[:, oz] += 40.0 * np.sin(2 * np.pi * 8.0 * tt_me)
                escreve_bloco(dados_me, CODE_ME_START)
                eventos.append({"type": "me_start", "class": cls,
                                "code": CODE_ME_START,
                                "sample": sample - n_me})
                escreve_bloco(np.zeros((1, CHANNEL_COUNT)), CODE_ME_END)
                eventos.append({"type": "me_end", "class": cls,
                                "code": CODE_ME_END, "sample": sample - 1})
                n_tr = int(getattr(args, "transicao_sec", 0.0) * FS)
                escreve_bloco(np.zeros((n_tr, CHANNEL_COUNT)), 0)
            dados, _ = gen.generate_block(n_mi, cls, mi_active=True)
            # Assinaturas fortes para o teste sintético: a classe muda o
            # padrao espacial em C3/C4/Oz, de forma que CSP+LDA tenha o que
            # aprender. Sem isso o teste vira ruido e nao valida o pipeline.
            tt = np.arange(n_mi) / FS
            c3 = gen.channel_names.index("C3")
            c4 = gen.channel_names.index("C4")
            oz = gen.channel_names.index("Oz")
            if cls == "direita":
                dados[:, c3] += 40.0 * np.sin(2 * np.pi * 12.0 * tt)
                dados[:, c4] += 5.0 * np.sin(2 * np.pi * 12.0 * tt)
            elif cls == "esquerda":
                dados[:, c4] += 40.0 * np.sin(2 * np.pi * 12.0 * tt)
                dados[:, c3] += 5.0 * np.sin(2 * np.pi * 12.0 * tt)
            else:
                dados[:, oz] += 40.0 * np.sin(2 * np.pi * 8.0 * tt)
            escreve_bloco(dados, CODE_MI_START)
            eventos.append({"type": "mi_start", "class": cls, "code": CODE_MI_START,
                            "sample": sample - n_mi})
            escreve_bloco(np.zeros((1, CHANNEL_COUNT)), CODE_MI_END)
            eventos.append({"type": "mi_end", "class": cls, "code": CODE_MI_END,
                            "sample": sample - 1})
            # ITI
            n_iti = int(random.uniform(args.iti_min, args.iti_max) * FS)
            escreve_bloco(np.zeros((n_iti, CHANNEL_COUNT)), 0)
        escreve_bloco(np.zeros((1, CHANNEL_COUNT)), CODE_BLOCK_END)
        eventos.append({"type": "block_end", "block": bi + 1,
                        "sample": sample - 1, "code": CODE_BLOCK_END})
    escreve_bloco(np.zeros((1, CHANNEL_COUNT)), CODE_SESSION_END)
    eventos.append({"type": "session_end", "sample": sample - 1,
                    "code": CODE_SESSION_END})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", *nomes, "Marker"])
        w.writerows(linhas)
    meta = {
        "subject_id": args.sujeito, "session_id": args.sessao,
        "protocol": "MI_3classes_validation", "version": "1.0",
        "source": "generator", "fs": FS, "channels": CHANNEL_COUNT,
        "channel_names": nomes, "n_blocks": args.blocos,
        "trials_per_class": args.trials_por_classe,
        "total_trials": args.blocos * args.trials_por_classe * 3,
        "mi_sec": args.mi_sec, "repouso_sec": args.repouso_sec,
        "cue_sec": args.cue_sec, "iti_range": [args.iti_min, args.iti_max],
        "classes": CLASS_NAMES, "class_codes": CLASS_CODES,
        "block_orders": bloco_ordens, "events": eventos,
        "aborted": False, "completed_trials": args.blocos * args.trials_por_classe * 3,
        "synthetic": True,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OK] CSV sintético: {csv_path} ({len(linhas)} amostras)")
    print(f"[OK] JSON sintético: {json_path}")
    print(f"[RESUMO] {meta['total_trials']} trials, {CHANNEL_COUNT} canais")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validação EEG 3 classes (MI direita/esquerda/repouso) - Gate C",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Fonte de dados
    parser.add_argument("--source", choices=["gnautilus", "generator"], default="generator",
                        help="Fonte de dados: gnautilus (hardware) ou generator (sintético)")
    
    # Protocolo
    parser.add_argument("--blocos", type=int, default=2, help="Número de blocos")
    parser.add_argument("--trials-por-classe", type=int, default=15, help="Trials por classe por bloco")
    parser.add_argument("--mi-sec", type=float, default=4.0, help="Duração da fase MI (s)")
    parser.add_argument("--me-sec", type=float, default=4.0, help="Duração da fase ME (s)")
    parser.add_argument("--transicao-sec", type=float, default=1.0, help="Transição ME->MI (s)")
    parser.add_argument("--sem-me", action="store_true", help="Mantém o modo MI-only (sem ME)")
    parser.add_argument("--repouso-sec", type=float, default=2.0, help="Duração do repouso baseline (s)")
    parser.add_argument("--cue-sec", type=float, default=1.0, help="Duração do cue visual (s)")
    parser.add_argument("--iti-min", type=float, default=0.5, help="ITI mínimo entre trials (s)")
    parser.add_argument("--iti-max", type=float, default=1.5, help="ITI máximo entre trials (s)")
    
    # Saída
    parser.add_argument("--pasta-saida", type=Path, default=Path("./dados_validacao"),
                        help="Pasta de saída para CSV/JSON")
    parser.add_argument("--sujeito", default="S01", help="ID do sujeito")
    parser.add_argument("--sessao", default="01", help="ID da sessão")
    
    # Opções
    parser.add_argument("--sem-janela", action="store_true", help="Modo headless (sem UI Qt)")
    parser.add_argument("--impedancia", action="store_true", help="Verificar impedância (apenas g.Nautilus)")
    parser.add_argument("--limite-impedancia", type=float, default=50.0, help="Limite impedância (kOhm)")
    parser.add_argument("--notch", type=float, default=60.0, help="Frequência do notch (Hz)")
    parser.add_argument("--bp-low", type=float, default=0.5, help="Highpass (Hz)")
    parser.add_argument("--bp-high", type=float, default=40.0, help="Lowpass (Hz)")
    parser.add_argument("--oscar", action="store_true", help="Ativar OSCAR (remoção de artefatos)")
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Cria pasta de saída
    output_dir = Path(args.pasta_saida)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Nome do arquivo CSV com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"eeg_mi3c_{args.sujeito}_s{args.sessao}_{timestamp}.csv"
    csv_path = output_dir / csv_name
    
    print("=" * 60)
    print("VALIDAÇÃO EEG 3 CLASSES - GATE C")
    print("=" * 60)
    print(f"Fonte: {args.source}")
    print(f"Sujeito: {args.sujeito}, Sessão: {args.sessao}")
    print(f"Blocos: {args.blocos}, Trials/classe: {args.trials_por_classe}")
    print(f"ME: {0 if args.sem_me else args.me_sec}s, MI: {args.mi_sec}s, "
          f"Transição: {0 if args.sem_me else args.transicao_sec}s, Cue: {args.cue_sec}s")
    print(f"Saída: {csv_path}")
    print(f"Headless: {args.sem_janela}")
    print("=" * 60)
    
    if args.source == "generator":
        return run_generator_offline(args)

    # Cria pipeline (hardware real; ainda nao validado neste gate)
    pipeline = AcquisitionPipeline(
        source=args.source,
        output_csv=csv_path,
        channel_count=CHANNEL_COUNT,
        sampling_rate=FS,
        frame_size=FRAME_SIZE,
        notch_freq=args.notch,
        bp_low=args.bp_low,
        bp_high=args.bp_high,
        enable_oscar=args.oscar,
    )
    
    # Cria paradigma
    paradigm = MI3ClassParadigm(
        pipeline=pipeline,
        n_blocks=args.blocos,
        trials_per_class=args.trials_por_classe,
        mi_sec=args.mi_sec,
        me_sec=(0.0 if args.sem_me else args.me_sec),
        transicao_sec=(0.0 if args.sem_me else args.transicao_sec),
        rest_sec=args.repouso_sec,
        cue_sec=args.cue_sec,
        iti_min_sec=args.iti_min,
        iti_max_sec=args.iti_max,
        output_dir=output_dir,
        subject_id=args.sujeito,
        session_id=args.sessao,
        headless=args.sem_janela,
        check_impedance=args.impedancia,
        impedance_limit_kohm=args.limite_impedancia,
    )
    
    # Executa
    try:
        result = paradigm.run()
        print("\n[SUCESSO] Aquisição finalizada.")
        return 0
    except KeyboardInterrupt:
        print("\n[INTERRUPÇÃO] Abortado pelo usuário.")
        return 130
    except Exception as e:
        print(f"\n[ERRO] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())