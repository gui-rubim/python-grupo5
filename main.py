
import csv
import time
import tkinter as tk

from datetime import datetime
from pathlib import Path

import cv2

from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        "Dependências não encontradas.\n"
        "Execute no terminal:\n"
        "python -m pip install ultralytics opencv-python pillow"
    )


class YOLOApp(tk.Tk):

    def __init__(self):
        super().__init__()

        # ==========================================
        # CONFIGURAÇÕES DA JANELA
        # ==========================================

        self.title("YOLO - Detecção de Pessoas e Objetos")
        self.geometry("1100x760")
        self.minsize(900, 650)
        self.configure(bg="#101820")

        # ==========================================
        # VARIÁVEIS DO SISTEMA
        # ==========================================

        self.model = None
        self.cap = None
        self.running = False

        self.frame_count = 0
        self.detected_objects = {}

        self.confidence = tk.DoubleVar(value=0.35)

        self.after_id = None
        self.latest_annotated_frame = None
        self.latest_frame_time = None

        self.fps_started_at = None

        # Diretório principal do projeto
        self.project_dir = Path(__file__).resolve().parent

        # ==========================================
        # INTERFACE PRINCIPAL
        # ==========================================

        self.create_interface()

        # Carrega o modelo depois de criar a interface
        self.load_model()

        # Evento de fechamento da janela
        self.protocol("WM_DELETE_WINDOW", self.close)

    # ==============================================
    # CRIAÇÃO DA INTERFACE
    # ==============================================

    def create_interface(self):

        self.top_bar = tk.Frame(
            self,
            bg="#1d2733",
            padx=18,
            pady=12
        )

        self.top_bar.pack(fill="x")

        self.title_label = tk.Label(
            self.top_bar,
            text="Detecção de Pessoas e Objetos com YOLO",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#1d2733"
        )

        self.title_label.pack(side="left")

        self.status_var = tk.StringVar(value="Carregando...")

        self.status_label = tk.Label(
            self.top_bar,
            textvariable=self.status_var,
            font=("Arial", 11, "bold"),
            fg="#a8f0c0",
            bg="#1d2733"
        )

        self.status_label.pack(side="right")

        # Área principal

        self.main_frame = tk.Frame(
            self,
            bg="#101820"
        )

        self.main_frame.pack(
            fill="both",
            expand=True,
            padx=16,
            pady=16
        )

        # Painel de vídeo

        self.video_panel = tk.Label(
            self.main_frame,
            bg="#1b2430",
            bd=2,
            relief="solid"
        )

        self.video_panel.pack(
            side="left",
            fill="both",
            expand=True
        )

        # Painel lateral

        self.side_panel = tk.Frame(
            self.main_frame,
            bg="#111b22",
            width=280,
            padx=14,
            pady=14
        )

        self.side_panel.pack(
            side="right",
            fill="y"
        )

        self.side_panel.pack_propagate(False)

        # ==========================================
        # CONTROLES
        # ==========================================

        self.controls = tk.Frame(
            self.side_panel,
            bg="#111b22"
        )

        self.controls.pack(
            fill="x",
            pady=(0, 18)
        )

        # Botão iniciar

        self.start_btn = tk.Button(
            self.controls,
            text="Iniciar Câmera",
            command=self.start_camera,
            bg="#2ec27e",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=14,
            pady=10,
            relief="flat",
            cursor="hand2"
        )

        self.start_btn.pack(
            fill="x",
            pady=(0, 10)
        )

        # Botão parar

        self.stop_btn = tk.Button(
            self.controls,
            text="Parar Câmera",
            command=self.stop_camera,
            bg="#d65d5d",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=14,
            pady=10,
            relief="flat",
            cursor="hand2",
            state="disabled"
        )

        self.stop_btn.pack(fill="x")

        # Controle de confiança

        self.confidence_label = tk.Label(
            self.controls,
            text="Confiança mínima: 0.35",
            bg="#111b22",
            fg="white",
            font=("Arial", 10)
        )

        self.confidence_label.pack(
            anchor="w",
            pady=(14, 0)
        )

        self.confidence_scale = tk.Scale(
            self.controls,
            from_=0.1,
            to=0.9,
            resolution=0.05,
            orient="horizontal",
            variable=self.confidence,
            command=self.update_confidence_label,
            bg="#111b22",
            fg="white",
            highlightthickness=0,
            troughcolor="#2b4052"
        )

        self.confidence_scale.pack(fill="x")

        # Botão salvar imagem

        self.save_image_btn = tk.Button(
            self.controls,
            text="Salvar captura",
            command=self.save_capture,
            bg="#3776ab",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=8,
            relief="flat",
            cursor="hand2"
        )

        self.save_image_btn.pack(
            fill="x",
            pady=(10, 8)
        )

        # Botão CSV

        self.save_csv_btn = tk.Button(
            self.controls,
            text="Registrar resumo CSV",
            command=self.save_detection_csv,
            bg="#3776ab",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=8,
            relief="flat",
            cursor="hand2"
        )

        self.save_csv_btn.pack(fill="x")

        # Botão sair

        self.exit_btn = tk.Button(
            self.controls,
            text="Sair",
            command=self.close,
            bg="#465361",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=8,
            relief="flat",
            cursor="hand2"
        )

        self.exit_btn.pack(
            fill="x",
            pady=(8, 0)
        )

        # ==========================================
        # PAINEL DE ESTATÍSTICAS
        # ==========================================

        self.metrics_frame = tk.LabelFrame(
            self.side_panel,
            text="Resumo",
            bg="#111b22",
            fg="#e8f3ff",
            font=("Arial", 11, "bold")
        )

        self.metrics_frame.pack(
            fill="x",
            pady=(0, 14)
        )

        self.obj_count_var = tk.StringVar(
            value="0 detecções | Pessoas: 0"
        )

        self.fps_var = tk.StringVar(
            value="FPS: 0"
        )

        self.last_detect_var = tk.StringVar(
            value="Nenhuma detecção"
        )

        tk.Label(
            self.metrics_frame,
            textvariable=self.obj_count_var,
            bg="#111b22",
            fg="white",
            font=("Arial", 11)
        ).pack(
            anchor="w",
            padx=10,
            pady=4
        )

        tk.Label(
            self.metrics_frame,
            textvariable=self.fps_var,
            bg="#111b22",
            fg="white",
            font=("Arial", 11)
        ).pack(
            anchor="w",
            padx=10,
            pady=4
        )

        tk.Label(
            self.metrics_frame,
            textvariable=self.last_detect_var,
            bg="#111b22",
            fg="white",
            font=("Arial", 11),
            wraplength=220,
            justify="left"
        ).pack(
            anchor="w",
            padx=10,
            pady=(4, 10)
        )

        # ==========================================
        # HISTÓRICO DE DETECÇÕES
        # ==========================================

        self.log_frame = tk.LabelFrame(
            self.side_panel,
            text="Classes detectadas",
            bg="#111b22",
            fg="#e8f3ff",
            font=("Arial", 11, "bold")
        )

        self.log_frame.pack(
            fill="both",
            expand=True
        )

        self.log_text = tk.Text(
            self.log_frame,
            height=16,
            bg="#0f1720",
            fg="#dfeefc",
            font=("Consolas", 10)
        )

        self.log_text.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        self.log_text.config(state="disabled")

        # Informação do modelo

        tk.Label(
            self.side_panel,
            text=(
                "Modelo pré-treinado: YOLOv8n. "
                "Detecta pessoas e objetos das classes COCO. "
                "Não identifica quem é a pessoa."
            ),
            bg="#111b22",
            fg="#aab8c5",
            font=("Arial", 9),
            wraplength=250,
            justify="left"
        ).pack(
            anchor="w",
            pady=(10, 0)
        )

        # Imagem inicial

        self.empty_frame = Image.new(
            "RGB",
            (800, 600),
            color=(15, 24, 32)
        )

        self.empty_photo = ImageTk.PhotoImage(
            self.empty_frame
        )

        self.video_panel.configure(
            image=self.empty_photo
        )

        self.video_panel.image = self.empty_photo

    # ==============================================
    # CARREGAMENTO DO MODELO YOLO
    # ==============================================

    def load_model(self):

        try:

            self.status_var.set("Carregando modelo YOLO...")

            self.update_idletasks()

            self.model = YOLO("yolov8n.pt")

            self.status_var.set("Modelo carregado - Pronto")

            self.log_message(
                "Modelo YOLOv8n carregado com sucesso."
            )

        except Exception as error:

            self.model = None

            self.status_var.set("Erro ao carregar modelo")

            self.log_message(
                f"Erro no carregamento do YOLO: {error}"
            )

    # ==============================================
    # ATUALIZAÇÃO DA CONFIANÇA
    # ==============================================

    def update_confidence_label(self, value):

        self.confidence_label.config(
            text=f"Confiança mínima: {float(value):.2f}"
        )

    # ==============================================
    # ABRIR CÂMERA
    # ==============================================

    def open_camera(self):

        if self.cap is not None:
            return self.cap.isOpened()

        try:

            self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():

                self.cap.release()
                self.cap = None

                self.status_var.set("Erro na câmera")

                self.log_message(
                    "Não foi possível abrir a câmera."
                )

                return False

            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                1280
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                720
            )

            return True

        except cv2.error as error:

            self.log_message(
                f"Erro ao acessar câmera: {error}"
            )

            self.cap = None

            return False

    # ==============================================
    # INICIAR CÂMERA
    # ==============================================

    def start_camera(self):

        if self.running:
            return

        if self.model is None:

            self.status_var.set(
                "Modelo YOLO indisponível"
            )

            self.log_message(
                "Não foi possível iniciar: modelo não carregado."
            )

            return

        if not self.open_camera():
            return

        self.running = True

        self.frame_count = 0

        self.fps_started_at = time.perf_counter()

        self.status_var.set("Executando")

        self.start_btn.config(state="disabled")

        self.stop_btn.config(state="normal")

        self.process_frame()

    # ==============================================
    # PARAR CÂMERA
    # ==============================================

    def stop_camera(self, status="Parado"):

        self.running = False

        self.status_var.set(status)

        if self.after_id is not None:

            try:
                self.after_cancel(self.after_id)

            except tk.TclError:
                pass

            self.after_id = None

        if self.cap is not None:

            self.cap.release()

            self.cap = None

        if hasattr(self, "start_btn"):

            self.start_btn.config(state="normal")

        if hasattr(self, "stop_btn"):

            self.stop_btn.config(state="disabled")

    # ==============================================
    # FECHAR PROGRAMA
    # ==============================================

    def close(self):

        self.stop_camera("Fechado")

        self.destroy()

    # ==============================================
    # REGISTRAR MENSAGENS
    # ==============================================

    def log_message(self, message):

        self.log_text.config(state="normal")

        self.log_text.insert(
            tk.END,
            message + "\n"
        )

        self.log_text.see(tk.END)

        self.log_text.config(state="disabled")

    # ==============================================
    # SALVAR CAPTURA
    # ==============================================

    def save_capture(self):

        if self.latest_annotated_frame is None:

            self.status_var.set(
                "Nenhuma imagem disponível"
            )

            self.log_message(
                "Inicie a câmera antes de salvar uma captura."
            )

            return

        captures_dir = self.project_dir / "capturas"

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        image_path = (
            captures_dir / f"deteccao_{timestamp}.jpg"
        )

        try:

            captures_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            success = cv2.imwrite(
                str(image_path),
                self.latest_annotated_frame
            )

            if not success:

                raise OSError(
                    "OpenCV não conseguiu gravar a imagem."
                )

        except OSError as error:

            self.status_var.set(
                "Erro ao salvar captura"
            )

            self.log_message(
                f"Erro ao salvar imagem: {error}"
            )

            return

        self.status_var.set("Captura salva")

        self.log_message(
            f"Imagem salva: {image_path}"
        )

    # ==============================================
    # REGISTRAR DETECÇÕES NO CSV
    # ==============================================

    def save_detection_csv(self):

        if self.latest_frame_time is None:

            self.status_var.set(
                "Nenhuma detecção disponível"
            )

            self.log_message(
                "Inicie a câmera antes de registrar o CSV."
            )

            return

        csv_dir = self.project_dir / "registros"

        csv_path = csv_dir / "deteccoes.csv"

        timestamp = self.latest_frame_time.isoformat(
            timespec="seconds"
        )

        rows = (
            sorted(self.detected_objects.items())
            if self.detected_objects
            else [("", 0)]
        )

        try:

            csv_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            needs_header = (
                not csv_path.exists()
                or csv_path.stat().st_size == 0
            )

            with csv_path.open(
                "a",
                newline="",
                encoding="utf-8-sig"
            ) as file:

                writer = csv.writer(file)

                if needs_header:

                    writer.writerow([
                        "data_hora",
                        "confianca_minima",
                        "classe",
                        "quantidade"
                    ])

                for class_name, count in rows:

                    writer.writerow([
                        timestamp,
                        f"{self.confidence.get():.2f}",
                        class_name,
                        count
                    ])

        except OSError as error:

            self.status_var.set(
                "Erro ao registrar CSV"
            )

            self.log_message(
                f"Erro ao gravar CSV: {error}"
            )

            return

        self.status_var.set(
            "Resumo registrado no CSV"
        )

        self.log_message(
            f"Relatório salvo: {csv_path}"
        )

    # ==============================================
    # ATUALIZAR HISTÓRICO DE DETECÇÕES
    # ==============================================

    def update_detection_log(self):

        self.log_text.config(state="normal")

        self.log_text.delete(
            "1.0",
            tk.END
        )

        if self.detected_objects:

            ordered_objects = sorted(
                self.detected_objects.items(),
                key=lambda item: (-item[1], item[0])
            )

            for name, count in ordered_objects:

                display_name = (
                    "pessoas"
                    if name == "person"
                    else name
                )

                self.log_text.insert(
                    tk.END,
                    f"{display_name}: {count}\n"
                )

        else:

            self.log_text.insert(
                tk.END,
                "Nenhum objeto detectado.\n"
            )

        self.log_text.config(state="disabled")

    # ==============================================
    # PROCESSAMENTO DAS DETECÇÕES
    # ==============================================

    def process_detections(self, results):

        self.detected_objects = {}

        for box in results[0].boxes:

            class_id = int(
                box.cls.item()
            )

            class_name = self.model.names[class_id]

            self.detected_objects[class_name] = (
                self.detected_objects.get(class_name, 0) + 1
            )

        total_objects = sum(
            self.detected_objects.values()
        )

        people_count = self.detected_objects.get(
            "person",
            0
        )

        self.obj_count_var.set(
            f"{total_objects} detecções | Pessoas: {people_count}"
        )

        if self.detected_objects:

            summary = ", ".join(
                f"{'pessoas' if name == 'person' else name}: {count}"
                for name, count in sorted(
                    self.detected_objects.items()
                )
            )

        else:

            summary = "Nenhuma detecção"

        self.last_detect_var.set(summary)

        self.update_detection_log()

    # ==============================================
    # ATUALIZAÇÃO DO FPS
    # ==============================================

    def update_fps(self):

        if self.fps_started_at is None:
            return

        elapsed = (
            time.perf_counter() - self.fps_started_at
        )

        if elapsed > 0:

            fps = self.frame_count / elapsed

            self.fps_var.set(
                f"FPS: {fps:.1f}"
            )

    # ==============================================
    # EXIBIR IMAGEM NA INTERFACE
    # ==============================================

    def display_frame(self, frame):

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            frame_rgb
        )

        resample = getattr(
            Image,
            "Resampling",
            Image
        ).LANCZOS

        # Mantém a proporção original da imagem
        image.thumbnail(
            (800, 600),
            resample
        )

        # Fundo para centralizar o vídeo
        background = Image.new(
            "RGB",
            (800, 600),
            color=(15, 24, 32)
        )

        x = (800 - image.width) // 2
        y = (600 - image.height) // 2

        background.paste(
            image,
            (x, y)
        )

        photo = ImageTk.PhotoImage(
            image=background
        )

        self.video_panel.configure(
            image=photo
        )

        # Evita que o Python apague a imagem
        self.video_panel.image = photo

    # ==============================================
    # PROCESSAMENTO PRINCIPAL DA CÂMERA
    # ==============================================

    def process_frame(self):

        self.after_id = None

        if not self.running:
            return

        if self.cap is None or not self.cap.isOpened():

            self.stop_camera(
                "Câmera indisponível"
            )

            self.log_message(
                "A câmera foi desconectada."
            )

            return

        try:

            # Captura imagem da câmera
            success, frame = self.cap.read()

            if not success:

                self.log_message(
                    "Erro ao capturar imagem da câmera."
                )

                self.stop_camera(
                    "Falha na captura"
                )

                return

            self.frame_count += 1

            self.latest_frame_time = datetime.now()

            # Executa a inteligência artificial
            results = self.model(
                frame,
                conf=self.confidence.get(),
                imgsz=640,
                iou=0.45,
                verbose=False
            )

            # Desenha as caixas de identificação
            annotated_frame = results[0].plot()

            # Guarda a imagem para salvar posteriormente
            self.latest_annotated_frame = (
                annotated_frame.copy()
            )

            # Atualiza estatísticas
            self.update_fps()

            # Atualiza contagem de objetos
            self.process_detections(results)

            # Atualiza a interface
            self.display_frame(
                annotated_frame
            )

            # Agenda o próximo processamento
            if self.running:

                self.after_id = self.after(
                    10,
                    self.process_frame
                )

        except Exception as error:

            self.log_message(
                f"Erro durante o processamento: {error}"
            )

            self.stop_camera(
                "Erro no processamento"
            )


# ==============================================
# EXECUÇÃO DO PROGRAMA
# ==============================================

def main():
    print("1 - Iniciando programa...")

    app = YOLOApp()

    print("2 - Janela criada!")

    app.mainloop()

    print("3 - Programa encerrado!")


if __name__ == "__main__":
    main()

