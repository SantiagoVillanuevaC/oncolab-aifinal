
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report
)


_BASE        = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE, "Cancer_Data.csv")
SCALER_PATH  = os.path.join(_BASE, "scaler.pkl")
MODEL_PATH   = os.path.join(_BASE, "model_rf.pkl")
RANDOM_STATE = 42
TEST_SIZE    = 0.2  

FEATURES = [
    "perimeter_worst",
    "area_worst",
    "radius_worst",
    "concave points_mean",   
    "concave points_worst",   
    "concavity_mean",
    "perimeter_mean",
    "area_mean",
]


print(f"\n{'='*60}")
print("  ONCOLAB AI — Entrenamiento Random Forest")
print(f"{'='*60}\n")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"No se encontró '{DATASET_PATH}'")

df = pd.read_csv(DATASET_PATH)


cols_drop = [c for c in df.columns if c.lower() in ('id', 'unnamed: 32')]
df.drop(columns=cols_drop, inplace=True)
print(f" Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

X = df[FEATURES].values
y = (df['diagnosis'] == 'M').astype(int).values

print(f" Features seleccionadas: {len(FEATURES)}")
print(f"   → Benignos (0): {(y==0).sum()} muestras")
print(f"   → Malignos (1): {(y==1).sum()} muestras")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n División del dataset:")
print(f"   → Entrenamiento: {len(X_train)} muestras (80%)")
print(f"   → Prueba:        {len(X_test)} muestras (20%)")


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"\n MinMaxScaler entrenado → rango [0, 1]")


modelo = RandomForestClassifier(
    n_estimators=100,     
    max_depth=None,       
    random_state=RANDOM_STATE,
    n_jobs=-1            
)
modelo.fit(X_train_scaled, y_train)
print(f"\n Random Forest entrenado ({modelo.n_estimators} árboles)")

y_pred = modelo.predict(X_test_scaled)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)

print(f"\n{'─'*50}")
print("  MÉTRICAS DE EVALUACIÓN (sobre datos de prueba)")
print(f"{'─'*50}")
print(f"  Accuracy  (Exactitud):   {accuracy*100:.2f}%")
print(f"  Precision (Precisión):   {precision*100:.2f}%")
print(f"  Recall    (Sensibilidad):{recall*100:.2f}%")
print(f"  F1-Score:                {f1*100:.2f}%")
print(f"\n  Matriz de Confusión:")
print(f"                  Pred.Benigno  Pred.Maligno")
print(f"  Real Benigno:   {cm[0][0]:^12}  {cm[0][1]:^12}")
print(f"  Real Maligno:   {cm[1][0]:^12}  {cm[1][1]:^12}")
print(f"\n  Importancia de Features:")

importancias = modelo.feature_importances_
for feat, imp in sorted(zip(FEATURES, importancias), key=lambda x: -x[1]):
    barra = '█' * int(imp * 40)
    print(f"  {feat:<30} {imp:.4f}  {barra}")


joblib.dump(scaler, SCALER_PATH)
joblib.dump(modelo, MODEL_PATH)

print(f"\n{'='*60}")
print("  ARCHIVOS EXPORTADOS")
print(f"{'='*60}")
print(f"  scaler.pkl    ({os.path.getsize(SCALER_PATH)/1024:.1f} KB)")
print(f"  model_rf.pkl  ({os.path.getsize(MODEL_PATH)/1024:.1f} KB)")
print(f"\n  Proceso completado. Accuracy final: {accuracy*100:.2f}%")
print(f"  Copia ambos .pkl a la raíz del proyecto Django")
print(f"{'='*60}\n")
