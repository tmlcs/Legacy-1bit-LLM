# Instrucciones para tu LLM 1-bit (1.58)

Este documento te guía sobre cómo utilizar y entrenar tu propio Modelo de Lenguaje Grande (LLM) de 1.58 bits, diseñado desde cero. Hemos creado la arquitectura, el proceso de tokenización, y el script de entrenamiento con gestión de checkpoints, optimizado tanto para ejecución local como en Google Colab.

---

## 🚀 Estructura del Proyecto

Los archivos clave de tu proyecto local son:

*   **`mi_modelo.py`**: Contiene la definición de la arquitectura de tu LLM (clases `BitLinear`, `AtencionMultiCabeza`, `BloqueTransformer`, `MiLLM`).
*   **`preparar_datos.py`**: Script para descargar el dataset WikiText-2 y entrenar/guardar el tokenizador (ya no se usa directamente para el entrenamiento, su lógica está integrada).
*   **`train.py`**: El script principal de entrenamiento. Utiliza las clases de `mi_modelo.py` y los datos preparados para entrenar tu LLM. Incluye lógica de carga y guardado de checkpoints.
*   **`tokenizer.json`**: El archivo generado que contiene tu tokenizador entrenado. Es esencial para que el modelo entienda el texto.
*   **`mi_llm_checkpoint.pth`**: Archivo de checkpoint. Aquí se guarda el estado del modelo y del optimizador durante el entrenamiento, permitiendo retomar el proceso.
*   **`mi_llm_epoch_X.pth`**: Copias de seguridad de los checkpoints al finalizar cada época (opcional).

---

## 💻 Uso Local (en tu PC)

Tu PC local es ideal para el desarrollo y pruebas pequeñas. Para el entrenamiento completo, Google Colab es más adecuado.

### Paso 1: Configuración del Entorno Local

Abre una terminal en el directorio raíz de tu proyecto y ejecuta los siguientes comandos para instalar las dependencias necesarias:

```bash
pip install torch torchvision torchaudio datasets tokenizers
```

### Paso 2: Entrenar el Modelo

Ejecuta el script de entrenamiento. El script detectará automáticamente si tienes una GPU local (compatible con CUDA) y la usará; de lo contrario, usará la CPU (lo cual será lento pero optimizado para `num_workers`).

```bash
python train.py
```

**Notas importantes:**
*   El script `train.py` ahora contiene la lógica para entrenar o cargar un tokenizador si no existe `tokenizer.json`.
*   El sistema de checkpoints (`mi_llm_checkpoint.pth`) permite retomar el entrenamiento en caso de interrupciones, guardando el progreso cada 1000 lotes.
*   **Rendimiento:** El entrenamiento en un PC portátil (especialmente en CPU) es un proceso muy lento y puede generar un sobrecalentamiento considerable. Para un entrenamiento eficiente y completo, se recomienda encarecidamente usar Google Colab.

---

## ☁️ Uso en Google Colab (Recomendado para Entrenamiento)

Google Colab te proporciona acceso gratuito a GPUs, lo cual acelera el entrenamiento exponencialmente.

### Paso 1: Abrir Colab y Habilitar la GPU

1.  Abre un nuevo notebook en [Google Colab](https://colab.research.google.com/).
2.  Ve al menú **Entorno de ejecución** (Runtime).
3.  Haz clic en **Cambiar tipo de entorno de ejecución** (Change runtime type).
4.  En el menú desplegable "Acelerador por hardware", selecciona **T4 GPU** (o la GPU que esté disponible, como P100).
5.  Haz clic en **Guardar**.

### Paso 2: Celda de Instalación de Dependencias

Crea una celda nueva, pega la siguiente línea y ejecútala para instalar las librerías necesarias en tu sesión de Colab:

```python
!pip install datasets tokenizers torch torchvision torchaudio
```

### Paso 3: Celda de Carga y Ejecución del Script de Entrenamiento

En la siguiente celda, pega y ejecuta **todo el contenido de tu archivo local `train.py`**. Como `mi_modelo.py` es un archivo separado en tu sistema local, para que Colab lo vea, tendrás que cargarlo.

**Opción A: Subida manual de archivos (requiere `mi_modelo.py` y `train.py`)**

1.  Usa el panel lateral de Colab (el icono de la carpeta 🗂️) para **arrastrar y soltar** los archivos `mi_modelo.py` y `train.py` (y `tokenizer.json` si ya lo tienes) directamente a la raíz de tu sesión de Colab.
2.  Crea una nueva celda y ejecuta:
    ```
    !python train.py
    ```

**Opción B: Consolidar todo en una celda (no requiere subir archivos)**

Si prefieres la opción que no requiere subir archivos manualmente, aquí tienes el script consolidado que incluye la definición del modelo. Pega **TODO este bloque de código** en una única celda de Colab y ejecútala.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# --- 1. Definición de la Arquitectura del Modelo ---
class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=torch.math.sqrt(5))
    def forward(self, x):
        scaling_factor = self.weight.abs().mean()
        quantized_weight = torch.round(self.weight / scaling_factor).clamp(-1, 1)
        ste_weight = (quantized_weight - self.weight).detach() + self.weight
        return F.linear(x, ste_weight) * scaling_factor

class AtencionMultiCabeza(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AtencionMultiCabeza, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads, self.head_dim = embed_dim, num_heads, embed_dim // num_heads
        self.q_proj = BitLinear(embed_dim, embed_dim)
        self.k_proj = BitLinear(embed_dim, embed_dim)
        self.v_proj = BitLinear(embed_dim, embed_dim)
        self.out_proj = BitLinear(embed_dim, embed_dim)
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.math.sqrt(self.head_dim)
        if mask is not None: scores = scores.masked_fill(mask, float('-inf'))
        context = torch.matmul(F.softmax(scores, dim=-1), v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        return self.out_proj(context)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.w_1, self.w_2 = BitLinear(embed_dim, ff_dim), BitLinear(ff_dim, embed_dim)
        self.activation = nn.ReLU()
    def forward(self, x): return self.w_2(self.activation(self.w_1(x)))

class BloqueTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(BloqueTransformer, self).__init__()
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.attn = AtencionMultiCabeza(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim)
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

class MiLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len):
        super(MiLLM, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([BloqueTransformer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_layer = BitLinear(embed_dim, vocab_size)
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        for layer in self.layers: x = layer(x, mask=mask)
        return self.output_layer(self.norm(x))

# --- 2. Preparación de Datos y Tokenizador ---
def preparar_tokenizer(vocab_size=30000, tokenizer_path="tokenizer.json"):
    if os.path.exists(tokenizer_path):
        print(f"Cargando tokenizador existente de {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)
    print("Entrenando tokenizador nuevo...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    def text_iterator():
        for i in range(len(dataset)):
            if dataset[i]['text'].strip(): yield dataset[i]['text']
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]","[PAD]","[CLS]","[SEP]","[MASK]"])
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(tokenizer_path)
    return tokenizer

# --- 3. Script de Entrenamiento con Checkpoints ---
CONFIG = {"vocab_size":30000, "max_seq_len":512, "embed_dim":512, "num_heads":8, "ff_dim":2048, "num_layers":6, "batch_size":16, "learning_rate":1e-4, "num_epochs":3, "checkpoint_path":"mi_llm_checkpoint.pth"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
tokenizer = preparar_tokenizer(CONFIG["vocab_size"])
model = MiLLM(CONFIG["vocab_size"], CONFIG["embed_dim"], CONFIG["num_heads"], CONFIG["ff_dim"], CONFIG["num_layers"], CONFIG["max_seq_len"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
loss_fn = torch.nn.CrossEntropyLoss()
start_epoch, start_batch = 0, 0
if os.path.exists(CONFIG["checkpoint_path"]):
    print(f"Cargando checkpoint desde {CONFIG['checkpoint_path']}")
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    start_batch = checkpoint.get('batch', 0) + 1
    print(f"Continuando desde la época {start_epoch + 1}, lote {start_batch}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Modelo creado con {total_params/1e6:.2f}M de parámetros entrenables.")
class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer, self.max_seq_len = tokenizer, max_seq_len
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        print("Tokenizando el dataset completo...")
        self.token_ids = [id for text in dataset['text'] if text.strip() for id in self.tokenizer.encode(text).ids]
        print(f"Dataset tokenizado. Total de tokens: {len(self.token_ids)}")
    def __len__(self): return len(self.token_ids) - self.max_seq_len
    def __getitem__(self, idx):
        input_chunk = self.token_ids[idx : idx + self.max_seq_len]
        target_chunk = self.token_ids[idx + 1 : idx + self.max_seq_len + 1]
        return torch.tensor(input_chunk), torch.tensor(target_chunk)
print("Creando Dataset y DataLoader...")
train_dataset = WikiTextDataset(tokenizer, CONFIG["max_seq_len"])
train_dataloader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=os.cpu_count())
print("Iniciando el entrenamiento...")
for epoch in range(start_epoch, CONFIG["num_epochs"]):
    model.train()
    if epoch > start_epoch: start_batch = 0
    for i, (input_batch, target_batch) in enumerate(train_dataloader):
        if i < start_batch: continue
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        optimizer.zero_grad()
        logits = model(input_batch)
        loss = loss_fn(logits.view(-1, CONFIG["vocab_size"]), target_batch.view(-1))
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Lote [{i+1}/{len(train_dataloader)}], Pérdida: {loss.item():.4f}")
        if (i + 1) % 1000 == 0:
            print(f"--- Guardando checkpoint en lote {i+1} ---")
            torch.save({'epoch':epoch, 'batch':i, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, CONFIG["checkpoint_path"])
    start_batch = 0
    print(f"--- Fin de la Epoch {epoch+1} ---")
    torch.save({'epoch':epoch + 1, 'batch':0, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, CONFIG["checkpoint_path"])
print(f"Entrenamiento completado. Modelo final guardado en '{CONFIG['checkpoint_path']}'")

```

---

Espero que esta guía te sea de gran utilidad para crear y entrenar tu propio LLM 1-bit. ¡Mucho éxito!
