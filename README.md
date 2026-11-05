# python-grupo5

Detecção de pessoas e objetos em tempo real pela webcam, usando YOLOv8 e uma interface em Tkinter. O modelo detecta as classes do conjunto COCO e não identifica quem é a pessoa.

## O que precisa para rodar

Além do Python 3 (testado no 3.12):

**Bibliotecas Python**

| Pacote | Para quê |
|---|---|
| `ultralytics` | Biblioteca do YOLO |
| `torch` e `torchvision` | Executam a rede neural (instalados junto com o `ultralytics`) |
| `opencv-python` | Câmera e processamento de imagem |
| `pillow` | Mostra o vídeo na janela |

**No computador**

- Tkinter: já vem com o Python no Windows; no Ubuntu/Debian, instale com `sudo apt install python3-tk`
- Webcam com a lente livre
- Ambiente gráfico, porque o app abre uma janela
- Internet na primeira execução, para baixar o modelo `yolov8n.pt` (6,2 MB)

GPU não é obrigatória: em um notebook, rodando só no processador, o app ficou em cerca de 9 FPS.

## Como rodar

Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install ultralytics opencv-python pillow
.venv/bin/python main.py
```

Sem GPU NVIDIA, instale antes o PyTorch para CPU. O download fica bem menor:

```bash
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Arquivos gerados

- `capturas/`: imagens salvas pelo botão "Salvar captura"
- `registros/deteccoes.csv`: resumos salvos pelo botão "Registrar resumo CSV"

Essas pastas ficam fora do repositório, assim como o modelo e o ambiente virtual.
