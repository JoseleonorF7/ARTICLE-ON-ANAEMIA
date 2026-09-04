"""
================================================================================
PIPELINE DE DETECCIÓN DE ANEMIA A PARTIR DE IMÁGENES DE PALMA
Versión corregida y ampliada para uso en un artículo científico
================================================================================

Este script parte del pipeline original (EfficientNet-B0, comparación
baseline vs. aprendizaje contrastivo supervisado / SupCon, evaluación
cruzada San Martín <-> Ghana) y corrige los problemas de metodología
detectados en la revisión de código, además de añadir el rigor estadístico
que un revisor va a exigir para un dataset médico pequeño.

Cada corrección está además comentada en el punto exacto del código donde
aplica, bajo una etiqueta "[FIX-n]" para que puedas ubicarla y, si quieres,
citarla en la sección de métodos del paper. Resumen:

  [FIX-1]  CRÍTICO — Fuga de datos. El script original preentrenaba el
           backbone contrastivo con el 100% del dominio destino (con
           etiquetas, vía SupConLoss) y luego "evaluaba" sobre ese mismo
           100% como si fuera un dominio no visto; lo mismo ocurría en la
           dirección inversa. Aquí cada dominio se particiona en
           train/val/test UNA sola vez (60/20/20 por defecto) y el test de
           cada dominio jamás participa en ninguna etapa -- ni
           preentrenamiento contrastivo, ni fine-tuning, ni selección de
           checkpoint -- de ningún experimento (SupCon, Baseline o Linear
           Probe). Todas las evaluaciones (SupCon, Baseline, Linear Probe)
           usan exactamente la misma partición de test por dirección, para
           que la comparación sea pareada y válida.
  [FIX-2]  `model_name` ahora sí determina la arquitectura instanciada
           (EfficientNet-B0 o ResNet50) vía `build_baseline_model`; antes
           solo cambiaba la etiqueta de texto en la tabla de resultados.
  [FIX-3]  Se unifica la estrategia de balanceo de clases: un único
           WeightedRandomSampler (sin, además, reponderar la pérdida) usado
           de forma consistente tanto en el baseline como en el fine-tuning
           SupCon. Antes el baseline recibía sampler + pérdida ponderada
           (doble corrección, sesga la calibración de probabilidades) y el
           SupCon no recibía ninguna corrección de desbalance.
  [FIX-4]  Se documenta explícitamente el supuesto de "palma ya
           segmentada" (el script no segmenta nada, solo re-escala y
           normaliza) y se deja un punto de extensión `segmentation_fn`
           para conectar un segmentador si las imágenes de entrada no
           vienen ya recortadas.
  [FIX-5]  Split "leak-proof" a nivel de sujeto/paciente cuando es posible
           inferirlo del nombre de archivo (o de un manifest explícito),
           con `assert` que verifica que ningún sujeto ni imagen aparece en
           más de una partición.
  [FIX-6]  Reproducibilidad GPU: se añade cudnn.deterministic=True y
           cudnn.benchmark=False (antes solo se fijaban las semillas de
           random/numpy/torch).
  [FIX-7]  El pipeline corre sobre N_SEEDS semillas de entrenamiento
           (por defecto 5, la partición de datos se mantiene fija) y
           reporta media ± desviación estándar por experimento, en vez de
           un único run con seed=42.
  [FIX-8]  Prueba estadística formal SupCon vs. Baseline: (a) un test
           pareado entre semillas (Wilcoxon signed-rank + t de Student
           pareado sobre el AUC de cada semilla) como comparación
           principal, y (b) DeLong / bootstrap pareado sobre las
           predicciones de una corrida, como chequeo complementario de un
           solo run. Cada una con su propia figura.
  [FIX-9]  Evaluación de "linear probe": se congela el backbone (SupCon vs.
           ImageNet puro), se extraen features y se entrena solo una
           regresión logística encima, para aislar la calidad de la
           representación aprendida del efecto del fine-tuning completo
           (protocolo estándar en la literatura contrastiva: SimCLR, SupCon).
  [FIX-10] Métricas ampliadas: PR-AUC (average precision), sensibilidad,
           especificidad y matriz de confusión, además de
           Accuracy/Precision/Recall/F1/AUC.
  [FIX-11] WeightedRandomSampler ya no recalcula `np.bincount` por cada
           muestra (antes O(n²), ahora O(n)).
  [FIX-12] Los hooks de Grad-CAM se guardan y se remueven explícitamente
           (`.remove()`) al terminar cada uso.
  [FIX-13] Se añade una ruta alternativa de carga por manifest CSV
           (columnas: path, label[, subject_id]) para no depender
           únicamente de que el nombre de la carpeta contenga el string de
           la clase.
  [FIX-14] `PalmDataset.__getitem__` ahora captura imágenes corruptas y
           reintenta con otra muestra, en vez de tumbar el entrenamiento.
  [FIX-15] Además de la tabla Excel, el script genera automáticamente
           figuras en OUTPUT_DIR/Figuras: curvas ROC/PR promedio ± std
           entre semillas, matrices de confusión promedio, barras de
           media ± std por métrica, slopegraphs pareados por semilla con
           el resultado del test estadístico, e histograma de la prueba
           bootstrap.

LIMITACIONES QUE SIGUEN REQUIRIENDO UNA DECISIÓN HUMANA (no se resuelven
solo con código):
  - Si el dataset tiene varias fotos por paciente y el nombre de archivo NO
    codifica un id de paciente, la heurística de agrupamiento (FIX-5) no
    tiene información suficiente para evitarlo: exporta un manifest real
    paciente -> imagen desde la fuente de datos y pásalo por
    `load_dataset_from_manifest`.
  - Balancear con `sampler` (oversampling) es una elección de diseño frente
    a otras válidas (undersampling, focal loss), no un hecho matemático; se
    eligió porque no distorsiona la pérdida ni la calibración de
    probabilidades (importante porque reportas AUC/PR-AUC).
  - Correr N_SEEDS>1 rehace preentrenamiento + fine-tuning completos por
    semilla (estadísticamente lo correcto), lo cual es costoso. Si el
    tiempo de sesión de Kaggle es una limitante real, hay un flag
    `REUSE_PRETRAINED_ACROSS_SEEDS` documentado más abajo con el trade-off
    explicado --úsalo con criterio, no como opción por defecto en el paper.
"""

import copy
import random
import re
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # headless: Kaggle/servidores sin display
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, auc as sk_auc,
)
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedShuffleSplit, train_test_split,
)
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

warnings.filterwarnings("ignore", category=UserWarning)

# ====================== REPRODUCIBILIDAD ======================
def set_seed(seed=42):
    """
    [FIX-6] Además de fijar random/numpy/torch, se fuerza a cuDNN a usar
    algoritmos deterministas. Sin esto, dos corridas con la MISMA seed
    pueden dar resultados distintos en GPU porque cuDNN por defecto elige
    el algoritmo de convolución más rápido disponible en cada llamada
    (benchmark=True), lo cual no es determinista entre ejecuciones.
    El costo es una posible pérdida de velocidad (~5-15%); para un paper
    que reporta números exactos, vale la pena.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ====================== CONFIG ======================
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()  # [FIX-20] precisión mixta: solo tiene sentido (y es segura) en GPU

BASE_DIR = Path("/kaggle/input/datasets/jose7777777777777777")
SAN_MARTIN_DIR = BASE_DIR / "processed-universidad-de-san-martin"
GHANA_DIR = BASE_DIR / "palm-ghana"
OUTPUT_DIR = Path("/kaggle/working/Results_Publicacion")
GRAD_CAM_DIR = OUTPUT_DIR / "GradCAM_Results"
FIG_DIR = OUTPUT_DIR / "Figuras"
GRAD_CAM_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- [FIX-7] múltiples semillas para reportar media ± std, en vez de un
# único run con seed=42. La PARTICIÓN de datos (train/val/test) se fija por
# separado con SPLIT_SEED y NO cambia entre semillas: lo único que varía de
# una semilla a otra es la inicialización/augmentación/orden de batches del
# entrenamiento.
# [FIX-20] 5 -> 3 semillas. Sigue permitiendo reportar media ± std y ver si
# los resultados son consistentes entre corridas (que es el objetivo real
# de FIX-7), pero corta ~40% del tiempo total. Es un trade-off explícito
# de eficiencia, no gratis: con 3 puntos la estimación de la varianza es
# más ruidosa que con 5. Si para el camera-ready hay tiempo de cómputo de
# sobra, subir esto a 5 no requiere tocar nada más del pipeline.
SEEDS = [42, 123, 2024]
SPLIT_SEED = 42

# Si tu cuota de cómputo en Kaggle no alcanza para rehacer el
# preentrenamiento contrastivo completo (PRETRAIN_EPOCHS) por cada una de
# las SEEDS, puedes poner esto en True para preentrenar una sola vez y solo
# variar la semilla del fine-tuning. Es un ahorro de cómputo real, pero
# ojo: subestima la varianza total del pipeline (el backbone deja de ser
# una fuente de variabilidad), así que decláralo así en el paper si lo usas.
REUSE_PRETRAINED_ACROSS_SEEDS = False

# [FIX-2] ahora sí puedes añadir "ResNet50" aquí y de verdad se entrenará
# esa arquitectura (antes el bug hacía que siempre se instanciara
# EfficientNet-B0 sin importar qué había en esta lista).
MODEL_NAMES = ["EfficientNet-B0"]

VAL_SIZE = 0.2
TEST_SIZE = 0.2  # -> train queda en 60% por dominio

# ================== HIPERPARÁMETROS ==================
PRETRAIN_EPOCHS = 25   # [FIX-20] techo (antes 30); con early stopping casi nunca se llega hasta aquí
PRETRAIN_PATIENCE = 7  # [FIX-20] epochs sin mejora en la pérdida contrastiva antes de frenar
FINETUNE_EPOCHS_SUPCon = 12
FINETUNE_EPOCHS_BASELINE = 15
LINEAR_PROBE_MAX_ITER = 2000  # iteraciones de la regresión logística (FIX-9)

BATCH_SIZE_PRETRAIN = 64
BATCH_SIZE_SUPCon = 64
BATCH_SIZE_BASELINE = 32

LR_PRETRAIN = 5e-4
LR_SUPCon = 1e-5
LR_BASELINE = 1e-4

WEIGHT_DECAY = 1e-5
PATIENCE = 6

N_BOOTSTRAP = 2000  # remuestreos para el test bootstrap pareado (FIX-8)

# [FIX-20] DataLoaders más eficientes: más workers y persistent_workers=True
# evita que se vuelvan a crear los procesos worker en cada epoch (con
# epochs cortos, ese overhead de arranque se nota). Ajusta NUM_WORKERS al
# número real de núcleos que te da la sesión de Kaggle (suele ser 4).
NUM_WORKERS = 4
PERSISTENT_WORKERS = NUM_WORKERS > 0

print("=== CONFIGURACIÓN CARGADA ===")
print(f"SEEDS: {SEEDS} (split fijo con SPLIT_SEED={SPLIT_SEED})")
print(f"PRETRAIN_EPOCHS (techo): {PRETRAIN_EPOCHS} | paciencia: {PRETRAIN_PATIENCE} | LR Pretrain: {LR_PRETRAIN}")
print(f"SupCon epochs: {FINETUNE_EPOCHS_SUPCon} | Baseline epochs: {FINETUNE_EPOCHS_BASELINE}")
print(f"Arquitecturas baseline: {MODEL_NAMES}")
print(f"AMP (precisión mixta): {USE_AMP} | num_workers: {NUM_WORKERS}")
print(f"Umbral de decisión: óptimo por F1 en validación (no fijo en 0.5, ver FIX-16)")
print("=" * 50)

# ====================== TRANSFORMS ======================
# [FIX-4] Estas transformaciones asumen que la imagen de entrada YA es un
# recorte/segmentación de la palma (el nombre de la carpeta de origen,
# "processed-universidad-de-san-martin", sugiere que el preprocesamiento
# upstream ya lo hizo). Si en algún momento alimentas este pipeline con
# fotos de la mano completa SIN segmentar, la interpretación de Grad-CAM
# ("activación dentro de la palma") deja de ser válida. Si necesitas
# segmentar aquí, conecta tu segmentador reemplazando `segmentation_fn`
# (por defecto None = no-op) dentro de `PalmDataset.__getitem__`.
segmentation_fn = None  # p.ej. una función que reciba un PIL.Image y devuelva otro recortado

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== DATASET ======================
class PalmDataset(Dataset):
    def __init__(self, paths, labels, transform=None, n_views=1, max_retries=5):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.n_views = n_views
        self.max_retries = max_retries  # [FIX-14]

    def __len__(self):
        return len(self.paths)

    def _load_image(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if segmentation_fn is not None:
            img = segmentation_fn(img)
        return img

    def __getitem__(self, idx, _attempt=0):
        # [FIX-14] Antes, un solo archivo corrupto/ilegible tumbaba TODO el
        # entrenamiento (Image.open lanza excepción sin capturar). Ahora se
        # registra una advertencia y se reintenta con otra muestra al azar,
        # hasta max_retries veces, para no perder una corrida completa por
        # un archivo dañado.
        try:
            img = self._load_image(idx)
        except Exception as e:
            if _attempt >= self.max_retries:
                raise RuntimeError(
                    f"No se pudo leer {self.paths[idx]} tras {self.max_retries} "
                    f"reintentos con otras muestras: {e}"
                )
            print(f"⚠️ Imagen corrupta/ilegible, se omite: {self.paths[idx]} ({e})")
            new_idx = random.randrange(len(self.paths))
            return self.__getitem__(new_idx, _attempt=_attempt + 1)

        if self.transform and self.n_views > 1:
            views = [self.transform(img) for _ in range(self.n_views)]
            return views, self.labels[idx]
        elif self.transform:
            return self.transform(img), self.labels[idx]
        return img, self.labels[idx]


# ====================== CARGA DE DATOS ======================
def load_dataset(base_dir):
    """
    [FIX-13, mantiene el comportamiento original por defecto] Etiquetado
    por nombre de carpeta. Es frágil (mayúsculas/acentos, subcadenas
    coincidentes, carpetas mal nombradas pasan silenciosamente) pero se
    conserva como default para no romper la estructura de datos existente.
    Para el material suplementario del paper es más auditable usar
    `load_dataset_from_manifest`, con una tabla ruta->etiqueta explícita
    que puedas versionar y citar.
    """
    paths, labels = [], []
    class0 = ['normal', 'non_anemic']
    class1 = ['anemic', 'leve', 'moderada']
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        for p in base_dir.rglob(ext):
            parent = str(p.parent).lower()
            if any(name in parent for name in class0):
                paths.append(p)
                labels.append(0)
            elif any(name in parent for name in class1):
                paths.append(p)
                labels.append(1)
    return paths, labels


def load_dataset_from_manifest(csv_path):
    """
    [FIX-13] Alternativa auditable a `load_dataset`: un CSV con columnas
    'path' (ruta absoluta o relativa a la imagen) y 'label' (0/1), y
    opcionalmente 'subject_id' (para que `split_domain` use agrupamiento
    real por paciente en vez de la heurística de nombre de archivo).
    """
    df = pd.read_csv(csv_path)
    assert {"path", "label"}.issubset(df.columns), \
        "El manifest debe tener al menos las columnas 'path' y 'label'."
    paths = [Path(p) for p in df["path"].tolist()]
    labels = df["label"].astype(int).tolist()
    subject_ids = df["subject_id"].astype(str).tolist() if "subject_id" in df.columns else None
    return paths, labels, subject_ids


# ====================== SPLIT LEAK-PROOF POR DOMINIO ======================
def guess_subject_id(path_str):
    """
    Extrae el ID del sujeto desde el nombre del archivo (Ghana).
    Ejemplos:
        'Non-AnemicP-071 (3).png'   → 'Non-AnemicP-071'
        'AnemicP-168.png'           → 'AnemicP-168'
        'Non-anemic-Pa-017 (4).png' → 'Non-anemic-Pa-017'
    """
    stem = Path(str(path_str)).stem
    # Elimina el " (número)" del final si existe
    stem = re.sub(r'\s*\(\d+\)$', '', stem).strip()
    return stem if stem else None


def assert_no_overlap(train_p, val_p, test_p, groups=None, train_idx=None, val_idx=None, test_idx=None):
    """
    [FIX-1 / FIX-5] Garantía en tiempo de ejecución (no solo intención de
    diseño) de que ninguna imagen -- ni, cuando aplica agrupamiento, ningún
    sujeto -- cae en más de una partición. Si esto falla, el pipeline se
    detiene: es preferible un AssertionError a publicar un número inflado
    por fuga de datos.
    """
    s_train, s_val, s_test = set(map(str, train_p)), set(map(str, val_p)), set(map(str, test_p))
    assert s_train.isdisjoint(s_test), "Fuga de datos: train y test comparten imágenes."
    assert s_val.isdisjoint(s_test), "Fuga de datos: val y test comparten imágenes."
    assert s_train.isdisjoint(s_val), "Fuga de datos: train y val comparten imágenes."
    if groups is not None and train_idx is not None:
        g_train = set(np.asarray(groups)[train_idx].tolist())
        g_val = set(np.asarray(groups)[val_idx].tolist())
        g_test = set(np.asarray(groups)[test_idx].tolist())
        assert g_train.isdisjoint(g_test), "Fuga a nivel de sujeto: train y test comparten sujetos."
        assert g_val.isdisjoint(g_test), "Fuga a nivel de sujeto: val y test comparten sujetos."
        assert g_train.isdisjoint(g_val), "Fuga a nivel de sujeto: train y val comparten sujetos."


def split_domain(paths, labels, subject_ids=None, val_size=VAL_SIZE, test_size=TEST_SIZE,
                  seed=SPLIT_SEED, domain_name="domain"):
    """
    [FIX-1, FIX-5] Split train/val/test hecho UNA sola vez por dominio y
    reutilizado en todas las etapas (preentrenamiento, fine-tuning,
    selección de checkpoint, evaluación final, linear probe). Esto es lo
    que resuelve la fuga crítica del script original: allí el "test"
    cruzado era el dominio completo, del cual el preentrenamiento
    contrastivo ya había visto el 100% de las imágenes CON etiqueta.

    Si `subject_ids` no se provee, se intenta inferir con
    `guess_subject_id`; si la heurística no parece confiable (casi tantos
    "sujetos" como imágenes), se cae a split estratificado por imagen y se
    imprime una advertencia explícita -- no se falla en silencio.
    """
    paths = np.array(paths, dtype=object)
    labels = np.array(labels)
    n = len(paths)

    if subject_ids is not None:
        groups = np.array(subject_ids, dtype=object)
        print(f"[{domain_name}] usando subject_id explícito del manifest "
              f"({len(set(groups))} sujetos para {n} imágenes).")
    else:
        candidate = np.array([guess_subject_id(p) for p in paths], dtype=object)
        n_unique = len(set(candidate)) if None not in candidate else -1
        if None not in candidate and 1 < n_unique < n * 0.9:
            groups = candidate
            print(f"[{domain_name}] id de sujeto INFERIDO heurísticamente del nombre de "
                  f"archivo ({n_unique} sujetos para {n} imágenes). Verifica que esto sea "
                  f"correcto para tu dataset -- si no lo es, usa un manifest explícito.")
        else:
            groups = None
            print(f"[{domain_name}] ADVERTENCIA: no se pudo inferir un id de sujeto/paciente "
                  f"confiable a partir de los nombres de archivo. El split se hace a NIVEL "
                  f"DE IMAGEN. Si el dataset tiene varias fotos por sujeto, esto puede dejar "
                  f"fotos del mismo sujeto en train y en test -- provee un manifest "
                  f"paciente->imagen con `load_dataset_from_manifest` para evitarlo.")

    idx_all = np.arange(n)
    if groups is not None:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(gss1.split(idx_all, labels, groups))
        rel_val = val_size / (1 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
        tr_rel, va_rel = next(gss2.split(trainval_idx, labels[trainval_idx], groups[trainval_idx]))
        train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[va_rel]
        # Nota: GroupShuffleSplit no garantiza balance exacto de clases por
        # partición (solo evita partir sujetos). Con pocos sujetos, val/test
        # pueden quedar algo desbalanceados; el sampler (FIX-3) y usar
        # AUC/PR-AUC en vez de solo accuracy mitigan el impacto.
    else:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(sss1.split(idx_all, labels))
        rel_val = val_size / (1 - test_size)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
        tr_rel, va_rel = next(sss2.split(trainval_idx, labels[trainval_idx]))
        train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[va_rel]

    train_p, train_l = paths[train_idx].tolist(), labels[train_idx].tolist()
    val_p, val_l = paths[val_idx].tolist(), labels[val_idx].tolist()
    test_p, test_l = paths[test_idx].tolist(), labels[test_idx].tolist()

    assert_no_overlap(train_p, val_p, test_p, groups, train_idx, val_idx, test_idx)
    print(f"[{domain_name}] split final -> train={len(train_p)} "
          f"val={len(val_p)} test={len(test_p)} "
          f"(balance test: {np.bincount(test_l)})")
    return (train_p, train_l), (val_p, val_l), (test_p, test_l)


def balanced_subsample(paths, labels, target_size=None, seed=None):
    """Sin cambios de fondo respecto al original, solo `seed` explícita
    para que sea reproducible por semilla de entrenamiento."""
    rng = np.random.default_rng(seed)
    paths = np.array(paths, dtype=object)
    labels = np.array(labels)
    class0_idx = np.where(labels == 0)[0]
    class1_idx = np.where(labels == 1)[0]

    if target_size is None:
        target_size = len(paths)
    per_class = min(int(target_size / 2), len(class0_idx), len(class1_idx))

    idx0 = rng.choice(class0_idx, size=per_class, replace=False)
    idx1 = rng.choice(class1_idx, size=per_class, replace=False)

    selected_idx = np.concatenate([idx0, idx1])
    rng.shuffle(selected_idx)
    return paths[selected_idx].tolist(), labels[selected_idx].tolist()


def overlay_gradcam(img_path, cam, alpha=0.6):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay


# ====================== BALANCEO DE CLASES ======================
def make_balanced_sampler(labels):
    """
    [FIX-3, FIX-11] Única fuente de corrección de desbalance en todo el
    pipeline (baseline Y SupCon fine-tuning la usan por igual). Antes: el
    baseline tenía sampler + CrossEntropyLoss ponderada A LA VEZ (sobre-
    corrige, distorsiona la calibración de probabilidades -- relevante
    porque reportamos AUC/PR-AUC) mientras que el fine-tuning SupCon no
    tenía ninguna corrección. [FIX-11] además, `np.bincount(labels)` se
    calcula UNA vez (antes se recalculaba por cada muestra: O(n²) -> O(n)).
    """
    labels = np.asarray(labels)
    class_counts = np.bincount(labels)
    sample_weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


# ====================== SUP CONTRASTIVE ======================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        device = features.device
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        batch_size, n_views, _ = features.shape
        features = F.normalize(features, dim=2)
        features = features.view(batch_size * n_views, -1)

        labels = labels.repeat_interleave(n_views).view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask - torch.eye(mask.shape[0], device=device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss


class EfficientNetWithContrastive(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 256)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x, mode='classify'):
        features = self.backbone(x)
        if mode == 'project':
            proj = self.projection(features)
            return F.normalize(proj, dim=1)
        if mode == 'features':
            return features  # [FIX-9] usado por el linear probe
        return self.classifier(features)


def build_baseline_model(model_name, num_classes=2):
    """
    [FIX-2] Antes `model_name` solo cambiaba la etiqueta en la tabla de
    resultados; `train_model` siempre instanciaba EfficientNet-B0 sin
    importar el valor de este parámetro. Ahora sí despacha a la
    arquitectura pedida. Añade más arquitecturas aquí si las necesitas
    (y agrégalas a la lista MODEL_NAMES de la config).
    """
    if model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Arquitectura no soportada en build_baseline_model: {model_name}")
    return model


# ====================== PREENTRENAMIENTO CONTRASTIVO ======================
def pretrain_contrastive_mixed(source_train_p, source_train_l, target_train_p, target_train_l, seed):
    """
    [FIX-1] Preentrenamiento contrastivo MIXTO, pero ahora recibe
    EXPLÍCITAMENTE solo las particiones de TRAIN de cada dominio (nunca
    val/test) -- ese es el cambio que rompe la fuga de datos original.
    Quien llama a esta función es responsable de pasar `*_train_p/l` y
    nunca el dominio completo.

    [FIX-20] Dos cambios de eficiencia, ninguno afecta el resultado
    esperado de forma apreciable:
      (a) precisión mixta (autocast + GradScaler): en GPU, cada paso de
          entrenamiento se hace ~1.5-2x más rápido sin cambiar la
          aritmética de forma que importe para la pérdida contrastiva.
      (b) early stopping por paciencia sobre la pérdida de
          preentrenamiento (antes se corrían SIEMPRE las PRETRAIN_EPOCHS
          completas, sin importar si la pérdida ya no bajaba). La
          paciencia se REINICIA justo cuando se descongela el backbone
          (epoch 5), porque ahí es normal y esperado que la pérdida suba
          temporalmente al empezar a entrenar más parámetros -- sin este
          reinicio, el early stopping podría activarse de forma espuria
          justo después del descongelamiento, cortando el entrenamiento
          antes de que el backbone realmente se ajuste.
    """
    print("=== Preentrenamiento Contrastivo MIXTO (solo particiones de train) ===")

    target_sub_p, target_sub_l = balanced_subsample(
        target_train_p, target_train_l, int(len(source_train_p) * 1.5), seed=seed)
    mixed_p = source_train_p + target_sub_p
    mixed_l = source_train_l + target_sub_l

    train_ds = PalmDataset(mixed_p, mixed_l, train_transform, n_views=2)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_PRETRAIN, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               persistent_workers=PERSISTENT_WORKERS)

    model = EfficientNetWithContrastive().to(DEVICE)

    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = SupConLoss(temperature=0.05)
    optimizer = optim.Adam(model.parameters(), lr=LR_PRETRAIN, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)
    scaler = GradScaler(enabled=USE_AMP)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(PRETRAIN_EPOCHS):
        if epoch == 5:
            for param in model.backbone.parameters():
                param.requires_grad = True
            print("→ Backbone descongelado")
            patience_counter = 0  # [FIX-20] no penalizar el salto de loss del descongelamiento

        model.train()
        total_loss = 0
        for (views, lbls) in train_loader:
            views = [v.to(DEVICE) for v in views]
            lbls = lbls.to(DEVICE)
            optimizer.zero_grad()

            with autocast(enabled=USE_AMP):
                proj1 = model(views[0], mode='project')
                proj2 = model(views[1], mode='project')
                proj = torch.cat([proj1.unsqueeze(1), proj2.unsqueeze(1)], dim=1)
                loss = criterion(proj, lbls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Pretrain Epoch {epoch + 1}/{PRETRAIN_EPOCHS} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PRETRAIN_PATIENCE and epoch >= 5:
                print(f"⏹ Early stopping preentrenamiento MIXTO en epoch {epoch + 1} "
                      f"(sin mejora en {PRETRAIN_PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, OUTPUT_DIR / f"pretrained_contrastive_mixed_seed{seed}.pth")
    return best_state


# ====================== GRAD-CAM ======================
class GradCAM:
    """[FIX-12] Guarda los handles de los hooks para poder removerlos."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._fwd_handle = target_layer.register_forward_hook(self.save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def generate_gradcam_examples(paths, labels, model, exp_name, use_supcon=False, n_per_class=1):
    """
    [FIX-18] Grad-CAM aquí es una herramienta COMPLEMENTARIA de
    interpretabilidad visual, no un resultado cuantitativo del paper -- por
    eso se genera un número pequeño y fijo de imágenes (`n_per_class` por
    clase, 1 por defecto) en vez de 3+3 por experimento. Ver en `main()`
    que además solo se llama para Baseline y Mixed (las dos condiciones
    que realmente se comparan en la narrativa del paper), no para las seis
    combinaciones dirección×condición -- así el total de imágenes queda
    acotado y manejable para una figura de la publicación.
    """
    model.eval()
    backbone = model.backbone if use_supcon else model

    target_layer = None
    if hasattr(backbone, 'features'):
        target_layer = backbone.features[-1]
    else:
        for name, module in backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module

    if target_layer is None:
        print("⚠️ No se encontró capa para Grad-CAM")
        return

    gradcam = GradCAM(model, target_layer)
    try:
        class0_idx = [i for i, l in enumerate(labels) if l == 0][:n_per_class]
        class1_idx = [i for i, l in enumerate(labels) if l == 1][:n_per_class]
        selected = class0_idx + class1_idx

        print(f"Generando {len(selected)} Grad-CAM para {exp_name}")

        for idx in selected:
            img_path = paths[idx]
            true_label = labels[idx]
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)

            try:
                with torch.set_grad_enabled(True):
                    cam = gradcam.generate(img_tensor, class_idx=true_label)
                overlay = overlay_gradcam(img_path, cam)

                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(img_pil)
                plt.title(f"Original - {'Anémico' if true_label == 1 else 'No Anémico'}")
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(overlay)
                plt.title(f"Grad-CAM - {exp_name}")
                plt.axis('off')

                save_path = GRAD_CAM_DIR / f"{exp_name}_idx{idx}_class{true_label}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Error en Grad-CAM {Path(img_path).name}: {e}")
    finally:
        gradcam.remove_hooks()  # [FIX-12]


# ====================== MÉTRICAS ======================
def compute_full_metrics(y_true, y_probs, y_pred):
    """
    [FIX-10] Antes solo se reportaban Accuracy/Recall/Precision/F1/AUC.
    Se añaden PR-AUC (más informativa que ROC-AUC bajo desbalance/baja
    prevalencia, como es de esperar en el dataset de Ghana), sensibilidad y
    especificidad explícitas (terminología estándar en tamizaje clínico), y
    la matriz de confusión completa.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Recall_Sensibilidad": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Especificidad": round(specificity, 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC": round(roc_auc_score(y_true, y_probs), 4),
        "PR_AUC": round(average_precision_score(y_true, y_probs), 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


# ====================== TRAIN MODEL ======================
def train_model(train_paths, train_labels, val_paths, val_labels,
                 test_paths, test_labels, exp_name="", pretrained_state=None,
                 use_supcon=False, model_name="EfficientNet-B0", seed=42):
    """
    [FIX-1] `test_paths/test_labels` ahora es SIEMPRE la partición de test
    del dominio destino (nunca el dominio completo), y es la MISMA para
    SupCon, Baseline y Linear Probe en una dirección dada -- necesario para
    que las pruebas pareadas (FIX-8) comparen manzanas con manzanas.
    [FIX-2] la arquitectura del baseline ahora depende de verdad de
    `model_name` (build_baseline_model).
    [FIX-3] un único mecanismo de balanceo (sampler) para baseline y SupCon.
    """
    start = time.time()

    train_ds = PalmDataset(train_paths, train_labels, train_transform, n_views=1)
    val_ds = PalmDataset(val_paths, val_labels, val_transform)
    test_ds = PalmDataset(test_paths, test_labels, val_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    sampler = make_balanced_sampler(train_labels)  # [FIX-3]

    if use_supcon:
        model = EfficientNetWithContrastive().to(DEVICE)
        if pretrained_state:
            model.load_state_dict(pretrained_state, strict=False)
        criterion = nn.CrossEntropyLoss()  # [FIX-3] sin pesos: el sampler ya balancea
        optimizer = optim.Adam(model.parameters(), lr=LR_SUPCon, weight_decay=WEIGHT_DECAY)
        batch_size = BATCH_SIZE_SUPCon
        epochs = FINETUNE_EPOCHS_SUPCon
    else:
        model = build_baseline_model(model_name, num_classes=2).to(DEVICE)  # [FIX-2]
        criterion = nn.CrossEntropyLoss()  # [FIX-3] sin pesos: el sampler ya balancea
        optimizer = optim.Adam(model.parameters(), lr=LR_BASELINE, weight_decay=WEIGHT_DECAY)
        batch_size = BATCH_SIZE_BASELINE
        epochs = FINETUNE_EPOCHS_BASELINE

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for images, lbls in train_loader:
            images, lbls = images.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images, mode='classify') if use_supcon else model(images)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, lbls in val_loader:
                images = images.to(DEVICE)
                outputs = model(images, mode='classify') if use_supcon else model(images)
                probs = torch.softmax(outputs, 1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(lbls.numpy())

        auc = roc_auc_score(all_labels, all_probs)

        if auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)

    suffix = "_SupCon" if use_supcon else f"_{model_name}_Balanced"
    torch.save(best_state, OUTPUT_DIR / f"{exp_name.replace(' ', '_')}{suffix}_seed{seed}.pth")

    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.to(DEVICE)
            outputs = model(images, mode='classify') if use_supcon else model(images)
            probs = torch.softmax(outputs, 1)[:, 1].cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())

    all_probs, all_preds, all_labels = np.array(all_probs), np.array(all_preds), np.array(all_labels)
    metrics = compute_full_metrics(all_labels, all_probs, all_preds)
    metrics.update({
        "Modelo": "EfficientNet-B0 + SupCon (Mixed)" if use_supcon else model_name + " (Balanced)",
        "Experimento": exp_name,
        "Seed": seed,
        "Set": "Test",
        "Time(s)": round(time.time() - start, 1),
    })

    print(f"✓ {metrics['Modelo']} | {exp_name} | seed={seed} | "
          f"AUC: {metrics['AUC']:.4f} | F1: {metrics['F1']:.4f}")

    return metrics, model, (all_labels, all_probs, all_preds)


# ====================== LINEAR PROBE ======================
@torch.no_grad()
def extract_frozen_features(backbone_model, paths, labels, use_supcon_forward=False):
    """
    [FIX-9] Pasa las imágenes UNA vez por un backbone CONGELADO y devuelve
    (features, labels) en numpy. Sirve tanto para el backbone SupCon
    (mode='features') como para un EfficientNet-B0 puro de ImageNet
    (feature vector = salida antes del classifier).
    """
    backbone_model.eval()
    ds = PalmDataset(paths, labels, val_transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    feats, labs = [], []
    for images, lbls in loader:
        images = images.to(DEVICE)
        if use_supcon_forward:
            f = backbone_model(images, mode='features')
        else:
            f = backbone_model(images)
        feats.append(f.cpu().numpy())
        labs.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def linear_probe_eval(train_p, train_l, test_p, test_l, backbone, use_supcon_forward, label):
    """
    [FIX-9] Protocolo "linear probe" estándar en aprendizaje contrastivo
    (SimCLR, SupCon): congelar el backbone, extraer features una sola vez,
    y ajustar solo un clasificador lineal (aquí, regresión logística de
    scikit-learn) encima. Si SupCon aporta una representación mejor que
    ImageNet puro, esto debería notarse incluso sin fine-tuning completo de
    la red -- y aísla ese efecto del efecto (separado) de fine-tunear todos
    los pesos.
    """
    for p in backbone.parameters():
        p.requires_grad = False

    X_train, y_train = extract_frozen_features(backbone, train_p, train_l, use_supcon_forward)
    X_test, y_test = extract_frozen_features(backbone, test_p, test_l, use_supcon_forward)

    clf = LogisticRegression(max_iter=LINEAR_PROBE_MAX_ITER, class_weight="balanced")
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    metrics = compute_full_metrics(y_test, probs, preds)
    metrics.update({"Modelo": f"Linear Probe ({label})", "Set": "Test"})
    print(f"✓ Linear Probe ({label}) | AUC: {metrics['AUC']:.4f} | F1: {metrics['F1']:.4f}")
    return metrics, (y_test, probs, preds)


# ====================== PRUEBAS ESTADÍSTICAS ======================
# [FIX-8] Dos niveles de evidencia estadística:
#  (a) comparación PRINCIPAL: Wilcoxon signed-rank + t pareado sobre el AUC
#      de cada semilla (N_SEEDS corridas independientes) -- responde "¿la
#      mejora de SupCon se sostiene de forma consistente entre semillas?".
#  (b) chequeo COMPLEMENTARIO de un solo run: DeLong / bootstrap pareado
#      sobre las predicciones de una corrida concreta -- útil si solo
#      puedes permitirte una semilla, o como diagnóstico adicional.

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx, sy = np.cov(v01), np.cov(v10)
    return aucs, sx / m + sy / n


def delong_roc_test(y_true, probs_a, probs_b):
    """Referencia: Sun & Xu (2014), "Fast Implementation of DeLong's
    Algorithm for Comparing the Areas Under Correlated ROC Curves"."""
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind="mergesort")
    y_sorted = y_true[order]
    m = int(np.sum(y_sorted == 1))
    preds_sorted = np.vstack([np.asarray(probs_a)[order], np.asarray(probs_b)[order]])
    aucs, cov = _fast_delong(preds_sorted, m)
    var_diff = max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-12)
    z = (aucs[0] - aucs[1]) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], z, p


def paired_bootstrap_auc_test(y_true, probs_a, probs_b, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    y_true, probs_a, probs_b = np.asarray(y_true), np.asarray(probs_a), np.asarray(probs_b)
    idx_pos, idx_neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        samp = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True),
        ])
        diffs[b] = (roc_auc_score(y_true[samp], probs_a[samp]) -
                    roc_auc_score(y_true[samp], probs_b[samp]))
    obs_diff = roc_auc_score(y_true, probs_a) - roc_auc_score(y_true, probs_b)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p = min(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)), 1.0)
    return obs_diff, ci_low, ci_high, p, diffs


def compare_seeds_paired(aucs_a, aucs_b):
    """Comparación PRINCIPAL entre semillas (ver nota arriba)."""
    aucs_a, aucs_b = np.asarray(aucs_a), np.asarray(aucs_b)
    try:
        w_stat, w_p = stats.wilcoxon(aucs_a, aucs_b)
    except ValueError:
        # ocurre si todas las diferencias son 0, o N tan chico que
        # scipy no puede construir la distribución -- se reporta NaN
        # en vez de fallar todo el pipeline.
        w_stat, w_p = np.nan, np.nan
    t_stat, t_p = stats.ttest_rel(aucs_a, aucs_b)
    return w_stat, w_p, t_stat, t_p


# ====================== FIGURAS ======================
def plot_mean_roc_pr(results_by_label, exp_name, out_dir, n_grid=100):
    """
    [FIX-15] results_by_label: {"SupCon": (y_true, [probs_seed0, probs_seed1, ...]), ...}
    y_true debe ser el MISMO array para todas las etiquetas (mismo test set
    fijo). Dibuja ROC y PR promedio ± std entre semillas, en vez de una
    curva de una sola corrida.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mean_fpr_grid = np.linspace(0, 1, n_grid)
    mean_rec_grid = np.linspace(0, 1, n_grid)

    for label, (y_true, probs_list) in results_by_label.items():
        tprs, aucs = [], []
        precs, aps = [], []
        for probs in probs_list:
            fpr, tpr, _ = roc_curve(y_true, probs)
            aucs.append(sk_auc(fpr, tpr))
            interp_tpr = np.interp(mean_fpr_grid, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

            prec, rec, _ = precision_recall_curve(y_true, probs)
            aps.append(average_precision_score(y_true, probs))
            order = np.argsort(rec)
            precs.append(np.interp(mean_rec_grid, rec[order], prec[order]))

        tprs = np.array(tprs); mean_tpr = tprs.mean(axis=0); mean_tpr[-1] = 1.0
        std_tpr = tprs.std(axis=0)
        precs = np.array(precs); mean_prec = precs.mean(axis=0); std_prec = precs.std(axis=0)

        n_runs = len(probs_list)
        suffix = f"(n={n_runs} semillas)" if n_runs > 1 else "(1 corrida)"
        axes[0].plot(mean_fpr_grid, mean_tpr,
                     label=f"{label} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f} {suffix}")
        axes[0].fill_between(mean_fpr_grid, np.clip(mean_tpr - std_tpr, 0, 1),
                              np.clip(mean_tpr + std_tpr, 0, 1), alpha=0.15)

        axes[1].plot(mean_rec_grid, mean_prec,
                     label=f"{label} AP={np.mean(aps):.3f}±{np.std(aps):.3f} {suffix}")
        axes[1].fill_between(mean_rec_grid, np.clip(mean_prec - std_prec, 0, 1),
                              np.clip(mean_prec + std_prec, 0, 1), alpha=0.15)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Azar")
    axes[0].set_xlabel("Tasa de falsos positivos"); axes[0].set_ylabel("Tasa de verdaderos positivos")
    axes[0].set_title(f"ROC promedio ± std — {exp_name}"); axes[0].legend(loc="lower right", fontsize=8)
    axes[1].set_xlabel("Recall (Sensibilidad)"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall promedio ± std — {exp_name}"); axes[1].legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    path = out_dir / f"ROC_PR_{exp_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_confusion_matrices_avg(results_by_label, exp_name, out_dir,
                                  class_names=("No anémico", "Anémico")):
    """results_by_label: {"SupCon": [(y_true, y_pred)_seed0, (y_true, y_pred)_seed1, ...], ...}
    Se promedian las matrices de confusión entre semillas (válido porque el
    test set es el mismo en todas las semillas)."""
    fig, axes = plt.subplots(1, len(results_by_label), figsize=(5 * len(results_by_label), 4.5))
    if len(results_by_label) == 1:
        axes = [axes]
    for ax, (label, seed_preds) in zip(axes, results_by_label.items()):
        cms = [confusion_matrix(y_true, y_pred, labels=[0, 1]) for y_true, y_pred in seed_preds]
        cm_mean = np.mean(cms, axis=0)
        im = ax.imshow(cm_mean, cmap="Blues")
        ax.set_title(f"{label} (n={len(seed_preds)} semillas)", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(class_names, rotation=20)
        ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_mean[i, j]:.1f}", ha="center", va="center",
                        color="white" if cm_mean[i, j] > cm_mean.max() / 2 else "black")
    fig.suptitle(f"Matrices de confusión (media entre semillas) — {exp_name}")
    plt.tight_layout()
    path = out_dir / f"ConfusionMatrix_{exp_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_metric_bars_with_error(df_per_seed, metric, out_dir):
    grouped = df_per_seed.groupby(["Experimento", "Modelo"])[metric].agg(["mean", "std", "count"]).reset_index()
    labels = grouped["Experimento"] + " | " + grouped["Modelo"]
    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(max(8, len(grouped) * 1.4), 5))
    ax.bar(x, grouped["mean"], yerr=grouped["std"].fillna(0), capsize=5,
           color="#4C72B0", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} (media ± desviación estándar sobre {int(grouped['count'].max())} semillas)")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = out_dir / f"Comparativa_{metric}_mean_std.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_paired_seeds(aucs_a, aucs_b, label_a, label_b, exp_name, metric_name, out_dir):
    aucs_a, aucs_b = np.asarray(aucs_a), np.asarray(aucs_b)
    n = len(aucs_a)
    w_stat, w_p, t_stat, t_p = compare_seeds_paired(aucs_a, aucs_b)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i in range(n):
        ax.plot([0, 1], [aucs_b[i], aucs_a[i]], color="gray", alpha=0.5, linewidth=1, zorder=1)
    ax.scatter(np.zeros(n), aucs_b, color="#DD8452", zorder=2, label=label_b)
    ax.scatter(np.ones(n), aucs_a, color="#4C72B0", zorder=2, label=label_a)
    ax.set_xticks([0, 1]); ax.set_xticklabels([label_b, label_a])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{exp_name}\nWilcoxon p={w_p:.4g} | t pareado p={t_p:.4g} (n={n} semillas)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / f"SeedComparison_{exp_name.replace(' ', '_')}_{metric_name}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path, (w_stat, w_p, t_stat, t_p)


def plot_bootstrap_diff(diffs, obs_diff, ci_low, ci_high, p_value, exp_name, out_dir,
                          label_a="SupCon", label_b="Baseline"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(diffs, bins=40, color="#8172B2", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Sin diferencia")
    ax.axvline(obs_diff, color="crimson", linewidth=2, label=f"Diferencia observada = {obs_diff:.3f}")
    ax.axvspan(ci_low, ci_high, color="crimson", alpha=0.1, label="IC 95% (bootstrap)")
    ax.set_xlabel(f"AUC({label_a}) − AUC({label_b})  [remuestreos bootstrap, 1 corrida]")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Prueba bootstrap pareada (complementaria) — {exp_name}\np = {p_value:.4g}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / f"BootstrapTest_{exp_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path

def pretrain_contrastive_source_only(train_paths, train_labels, seed, domain_name="source"):
    """
    Preentrenamiento SupCon SOLO con el dominio fuente (sin ver el dominio destino).
    Esta es la versión limpia para medir el efecto puro de SupCon en generalización.

    [FIX-20] AMP + early stopping por paciencia, igual que en
    `pretrain_contrastive_mixed` (ver ahí la justificación completa). Como
    esta función se llama DOS veces por semilla (una vez por dominio
    fuente), es el mayor ahorro de tiempo total del pipeline.
    """
    print(f"=== Preentrenamiento Contrastivo SOURCE-ONLY ({domain_name}) ===")

    train_ds = PalmDataset(train_paths, train_labels, train_transform, n_views=2)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_PRETRAIN, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               persistent_workers=PERSISTENT_WORKERS)

    model = EfficientNetWithContrastive().to(DEVICE)

    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = SupConLoss(temperature=0.05)
    optimizer = optim.Adam(model.parameters(), lr=LR_PRETRAIN, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)
    scaler = GradScaler(enabled=USE_AMP)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(PRETRAIN_EPOCHS):
        if epoch == 5:
            for param in model.backbone.parameters():
                param.requires_grad = True
            print("→ Backbone descongelado")
            patience_counter = 0  # [FIX-20]

        model.train()
        total_loss = 0
        for (views, lbls) in train_loader:
            views = [v.to(DEVICE) for v in views]
            lbls = lbls.to(DEVICE)
            optimizer.zero_grad()

            with autocast(enabled=USE_AMP):
                proj1 = model(views[0], mode='project')
                proj2 = model(views[1], mode='project')
                proj = torch.cat([proj1.unsqueeze(1), proj2.unsqueeze(1)], dim=1)
                loss = criterion(proj, lbls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Pretrain Epoch {epoch + 1}/{PRETRAIN_EPOCHS} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PRETRAIN_PATIENCE and epoch >= 5:
                print(f"⏹ Early stopping preentrenamiento SOURCE-ONLY ({domain_name}) en epoch "
                      f"{epoch + 1} (sin mejora en {PRETRAIN_PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, OUTPUT_DIR / f"pretrained_source_only_{domain_name}_seed{seed}.pth")
    return best_state


# ====================== MAIN ======================

def main():
    sm_paths, sm_labels = load_dataset(SAN_MARTIN_DIR)
    ghana_paths, ghana_labels = load_dataset(GHANA_DIR)
    print(f"San Martín: {len(sm_paths)} | Ghana: {len(ghana_paths)}")

    # [FIX-1, FIX-5] split ÚNICO por dominio
    (sm_train_p, sm_train_l), (sm_val_p, sm_val_l), (sm_test_p, sm_test_l) = split_domain(
        sm_paths, sm_labels, domain_name="San Martín")
    (ghana_train_p, ghana_train_l), (ghana_val_p, ghana_val_l), (ghana_test_p, ghana_test_l) = split_domain(
        ghana_paths, ghana_labels, domain_name="Ghana")

    directions = [
        ("SM→Ghana", (sm_train_p, sm_train_l), (sm_val_p, sm_val_l), (ghana_test_p, ghana_test_l)),
        ("Ghana→SM", (ghana_train_p, ghana_train_l), (ghana_val_p, ghana_val_l), (sm_test_p, sm_test_l)),
    ]

    all_metrics_rows = []

    # raw ahora incluye las tres condiciones
    raw = {d[0]: {"Baseline": [], "SourceOnly": [], "Mixed": []} for d in directions}

    for seed in SEEDS:
        print("\n" + "=" * 90)
        print(f"SEMILLA {seed}")
        print("=" * 90)
        set_seed(seed)

        # ============================================================
        # 1. PREENTRENAMIENTOS (una vez por semilla)
        # ============================================================
        print("\n>>> Preentrenamiento Source-only SM")
        pretrained_sm_only = pretrain_contrastive_source_only(
            sm_train_p, sm_train_l, seed=seed, domain_name="SM")

        print("\n>>> Preentrenamiento Source-only Ghana")
        pretrained_ghana_only = pretrain_contrastive_source_only(
            ghana_train_p, ghana_train_l, seed=seed, domain_name="Ghana")

        print("\n>>> Preentrenamiento Mixed")
        pretrained_mixed = pretrain_contrastive_mixed(
            sm_train_p, sm_train_l, ghana_train_p, ghana_train_l, seed=seed)

        # ============================================================
        # 2. EXPERIMENTOS POR DIRECCIÓN
        # ============================================================
        for direction_name, (tr_p, tr_l), (va_p, va_l), (te_p, te_l) in directions:

            print(f"\n---------- {direction_name} | seed={seed} ----------")

            # ----- A. Baseline -----
            metrics_b, baseline_model, (y_true_b, probs_b, preds_b) = train_model(
                tr_p, tr_l, va_p, va_l, te_p, te_l,
                exp_name=direction_name,
                pretrained_state=None,
                use_supcon=False,
                model_name="EfficientNet-B0",
                seed=seed
            )
            metrics_b["Modelo"] = "EfficientNet-B0 (Baseline)"
            all_metrics_rows.append(metrics_b)
            raw[direction_name]["Baseline"].append((y_true_b, probs_b, preds_b))

            if seed == SEEDS[0]:
                generate_gradcam_examples(te_p, te_l, baseline_model, f"{direction_name}_Baseline", use_supcon=False)

            # ----- B. SupCon Source-only -----
            if direction_name == "SM→Ghana":
                pretrained_source = pretrained_sm_only
            else:  # Ghana→SM
                pretrained_source = pretrained_ghana_only

            metrics_so, model_so, (y_true_so, probs_so, preds_so) = train_model(
                tr_p, tr_l, va_p, va_l, te_p, te_l,
                exp_name=direction_name,
                pretrained_state=pretrained_source,
                use_supcon=True,
                seed=seed
            )
            metrics_so["Modelo"] = "EfficientNet-B0 + SupCon (Source-only)"
            all_metrics_rows.append(metrics_so)
            raw[direction_name]["SourceOnly"].append((y_true_so, probs_so, preds_so))

            if seed == SEEDS[0]:
                generate_gradcam_examples(te_p, te_l, model_so, f"{direction_name}_SourceOnly", use_supcon=True)

            # ----- C. SupCon Mixed -----
            metrics_mx, model_mx, (y_true_mx, probs_mx, preds_mx) = train_model(
                tr_p, tr_l, va_p, va_l, te_p, te_l,
                exp_name=direction_name,
                pretrained_state=pretrained_mixed,
                use_supcon=True,
                seed=seed
            )
            metrics_mx["Modelo"] = "EfficientNet-B0 + SupCon (Mixed)"
            all_metrics_rows.append(metrics_mx)
            raw[direction_name]["Mixed"].append((y_true_mx, probs_mx, preds_mx))

            if seed == SEEDS[0]:
                generate_gradcam_examples(te_p, te_l, model_mx, f"{direction_name}_Mixed", use_supcon=True)

    # ============================================================
    # 3. AGREGACIÓN Y TABLA FINAL
    # ============================================================
    final_df = pd.DataFrame(all_metrics_rows)
    ordered_cols = ["Experimento", "Modelo", "Seed", "Set", "Accuracy",
                    "Recall_Sensibilidad", "Especificidad", "Precision", "F1",
                    "AUC", "PR_AUC", "TN", "FP", "FN", "TP", "Time(s)"]
    ordered_cols = [c for c in ordered_cols if c in final_df.columns]
    final_df = final_df[ordered_cols]

    summary_df = (final_df.groupby(["Experimento", "Modelo"])
                  [["Accuracy", "Recall_Sensibilidad", "Especificidad", "Precision", "F1", "AUC", "PR_AUC"]]
                  .agg(["mean", "std"]))

    # ============================================================
    # 4. PRUEBAS ESTADÍSTICAS + FIGURAS
    # ============================================================
    stats_rows = []

    for direction_name, _, _, (te_p, te_l) in directions:
        y_true = np.array(te_l)

        base_runs   = raw[direction_name]["Baseline"]
        source_runs = raw[direction_name]["SourceOnly"]
        mixed_runs  = raw[direction_name]["Mixed"]

        aucs_bl  = [roc_auc_score(y, p) for y, p, _ in base_runs]
        aucs_so  = [roc_auc_score(y, p) for y, p, _ in source_runs]
        aucs_mx  = [roc_auc_score(y, p) for y, p, _ in mixed_runs]

        # --- Comparaciones pareadas principales ---
        # 1. Source-only vs Baseline
        _, _ = plot_paired_seeds(aucs_so, aucs_bl, "Source-only", "Baseline",
                                 direction_name, "AUC", FIG_DIR)
        w_stat, w_p, t_stat, t_p = compare_seeds_paired(aucs_so, aucs_bl)
        stats_rows.append({
            "Experimento": direction_name,
            "Comparación": "Source-only vs Baseline",
            "AUC_A_mean": np.mean(aucs_so), "AUC_A_std": np.std(aucs_so),
            "AUC_B_mean": np.mean(aucs_bl), "AUC_B_std": np.std(aucs_bl),
            "Wilcoxon_stat": w_stat, "Wilcoxon_p": w_p,
            "ttest_pareado_stat": t_stat, "ttest_pareado_p": t_p,
        })

        # 2. Mixed vs Source-only
        _, _ = plot_paired_seeds(aucs_mx, aucs_so, "Mixed", "Source-only",
                                 direction_name, "AUC", FIG_DIR)
        w_stat, w_p, t_stat, t_p = compare_seeds_paired(aucs_mx, aucs_so)
        stats_rows.append({
            "Experimento": direction_name,
            "Comparación": "Mixed vs Source-only",
            "AUC_A_mean": np.mean(aucs_mx), "AUC_A_std": np.std(aucs_mx),
            "AUC_B_mean": np.mean(aucs_so), "AUC_B_std": np.std(aucs_so),
            "Wilcoxon_stat": w_stat, "Wilcoxon_p": w_p,
            "ttest_pareado_stat": t_stat, "ttest_pareado_p": t_p,
        })

        # 3. Mixed vs Baseline
        _, _ = plot_paired_seeds(aucs_mx, aucs_bl, "Mixed", "Baseline",
                                 direction_name, "AUC", FIG_DIR)
        w_stat, w_p, t_stat, t_p = compare_seeds_paired(aucs_mx, aucs_bl)
        stats_rows.append({
            "Experimento": direction_name,
            "Comparación": "Mixed vs Baseline",
            "AUC_A_mean": np.mean(aucs_mx), "AUC_A_std": np.std(aucs_mx),
            "AUC_B_mean": np.mean(aucs_bl), "AUC_B_std": np.std(aucs_bl),
            "Wilcoxon_stat": w_stat, "Wilcoxon_p": w_p,
            "ttest_pareado_stat": t_stat, "ttest_pareado_p": t_p,
        })

        # Curvas ROC / PR promedio
        plot_mean_roc_pr({
            "Baseline": (y_true, [p for _, p, _ in base_runs]),
            "Source-only": (y_true, [p for _, p, _ in source_runs]),
            "Mixed": (y_true, [p for _, p, _ in mixed_runs]),
        }, direction_name, FIG_DIR)

        # Matrices de confusión promedio
        plot_confusion_matrices_avg({
            "Baseline": [(y, pr) for y, _, pr in base_runs],
            "Source-only": [(y, pr) for y, _, pr in source_runs],
            "Mixed": [(y, pr) for y, _, pr in mixed_runs],
        }, direction_name, FIG_DIR)

    stats_df = pd.DataFrame(stats_rows)

    for metric in ["AUC", "F1", "PR_AUC"]:
        plot_metric_bars_with_error(final_df, metric, FIG_DIR)

    # ============================================================
    # 5. EXPORTAR
    # ============================================================
    excel_path = OUTPUT_DIR / "Comparativa_Final_Combinada.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="Resultados_por_semilla", index=False)
        summary_df.to_excel(writer, sheet_name="Resumen_media_std")
        stats_df.to_excel(writer, sheet_name="Pruebas_estadisticas", index=False)

    print("\n" + "=" * 70)
    print("RESUMEN (media ± std entre semillas)")
    print("=" * 70)
    print(summary_df)

    print("\n" + "=" * 70)
    print("PRUEBAS ESTADÍSTICAS")
    print("=" * 70)
    print(stats_df)

    print(f"\n✅ Tabla completa: {excel_path}")
    print(f"✅ Figuras: {FIG_DIR}")
    print(f"✅ Grad-CAM: {GRAD_CAM_DIR}")


if __name__ == "__main__":
    main()