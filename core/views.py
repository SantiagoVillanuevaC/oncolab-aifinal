import os, logging
import numpy as np
import joblib
from django.shortcuts import render, redirect
from django.utils import timezone
from django.conf import settings

logger = logging.getLogger(__name__)

_SCALER_PATH = os.path.join(settings.BASE_DIR, "scaler.pkl")
_MODEL_PATH  = os.path.join(settings.BASE_DIR, "model_rf.pkl")  # ← ahora usa Random Forest

try:
    SCALER = joblib.load(_SCALER_PATH)
    MODELO = joblib.load(_MODEL_PATH)
    MODELOS_CARGADOS = True
    print("Random Forest cargado correctamente.")
except FileNotFoundError as e:
    SCALER = None; MODELO = None; MODELOS_CARGADOS = False
    print(f"Modelos NO encontrados: {e}")


def _calcular_stats(historial):
    total    = len(historial)
    benignos = sum(1 for d in historial if d.get('resultado') == 'Benigno')
    malignos = total - benignos
    pct_b = round(benignos / total * 100) if total > 0 else 0
    pct_m = round(malignos / total * 100) if total > 0 else 0
    return total, benignos, malignos, pct_b, pct_m


def dashboard(request):
    historial = request.session.get('historial_diagnosticos', [])
    total, benignos, malignos, pct_b, pct_m = _calcular_stats(historial)
    return render(request, 'core/index.html', {
        'ultimos_diagnosticos': list(reversed(historial))[:5],
        'total_diagnosticos':   total,
        'total_benignos':       benignos,
        'total_malignos':       malignos,
        'modelo_activo':        MODELOS_CARGADOS,
    })


def nuevo_diagnostico(request):
    if not MODELOS_CARGADOS:
        return render(request, 'core/diagnostico_form.html', {
            'form': {}, 'error': 'El modelo no está cargado. Ejecuta train_and_export.py primero.'
        })

    if request.method == 'POST':
        try:
            pw  = float(request.POST.get("perimeter_worst", 0))
            aw  = float(request.POST.get("area_worst", 0))
            rw  = float(request.POST.get("radius_worst", 0))
            cpm = float(request.POST.get("concave_points_mean", 0))
            cpw = float(request.POST.get("concave_points_worst", 0))
            cm  = float(request.POST.get("concavity_mean", 0))
            pm  = float(request.POST.get("perimeter_mean", 0))
            am  = float(request.POST.get("area_mean", 0))

            X        = np.array([[pw, aw, rw, cpm, cpw, cm, pm, am]])
            X_scaled = SCALER.transform(X)

           
            cluster      = int(MODELO.predict(X_scaled)[0])
            probabilidad = MODELO.predict_proba(X_scaled)[0]

            
            resultado     = "Maligno" if cluster == 1 else "Benigno"
            prob_benigno  = round(probabilidad[0] * 100, 1)
            prob_maligno  = round(probabilidad[1] * 100, 1)

            print(f"  Predicción: {resultado} | Prob. Benigno: {prob_benigno}% | Prob. Maligno: {prob_maligno}%")

            id_paciente  = request.POST.get('id_paciente', 'SIN-ID').strip() or 'SIN-ID'
            fecha_actual = timezone.now().strftime('%d/%m/%Y %H:%M')

            request.session['resultado_diagnostico'] = {
                'resultado':    resultado,
                'cluster':      cluster,
                'id_paciente':  id_paciente,
                'fecha':        fecha_actual,
                'prob_benigno': prob_benigno,
                'prob_maligno': prob_maligno,
                'datos': {
                    'perimeter_worst':      str(pw),
                    'area_worst':           str(aw),
                    'radius_worst':         str(rw),
                    'concave_points_mean':  str(cpm),
                    'concave_points_worst': str(cpw),
                    'concavity_mean':       str(cm),
                    'perimeter_mean':       str(pm),
                    'area_mean':            str(am),
                }
            }

            historial = request.session.get('historial_diagnosticos', [])
            historial.append({
                'id': len(historial)+1,
                'id_paciente': id_paciente,
                'fecha': fecha_actual,
                'resultado': resultado,
                'prob_benigno': prob_benigno,
                'prob_maligno': prob_maligno,
            })
            request.session['historial_diagnosticos'] = historial
            request.session.modified = True
            return redirect('resultado_diagnostico')

        except Exception as e:
            print(f"Error: {e}")
            return render(request, 'core/diagnostico_form.html', {
                'form': request.POST, 'error': f'Error al procesar: {str(e)}'
            })

    return render(request, 'core/diagnostico_form.html', {'form': {}})


def resultado_diagnostico(request):
    datos = request.session.get('resultado_diagnostico', None)
    if not datos:
        return redirect('nuevo_diagnostico')
    context = {
        'resultado':    datos.get('resultado'),
        'cluster':      datos.get('cluster'),
        'id_paciente':  datos.get('id_paciente'),
        'fecha':        datos.get('fecha'),
        'prob_benigno': datos.get('prob_benigno', 50),
        'prob_maligno': datos.get('prob_maligno', 50),
        'datos':        datos.get('datos', {}),
    }
    del request.session['resultado_diagnostico']
    request.session.modified = True
    return render(request, 'core/resultado.html', context)


def historial(request):
    todos = list(reversed(request.session.get('historial_diagnosticos', [])))
    total, benignos, malignos, pct_b, pct_m = _calcular_stats(todos)
    return render(request, 'core/historial.html', {
        'diagnosticos':   todos,
        'total':          total,
        'total_benignos': benignos,
        'total_malignos': malignos,
        'pct_benigno':    pct_b,
        'pct_maligno':    pct_m,
    })


def ver_resultado(request, id):
    historial_sesion = request.session.get('historial_diagnosticos', [])
    diag = next((d for d in historial_sesion if d.get('id') == id), None)
    context = {
        'resultado':    diag.get('resultado', 'No disponible') if diag else 'No disponible',
        'cluster':      '1' if (diag and diag.get('resultado') == 'Maligno') else '0',
        'id_paciente':  diag.get('id_paciente', 'N/A') if diag else f'#{id}',
        'fecha':        diag.get('fecha', '--') if diag else '--',
        'prob_benigno': diag.get('prob_benigno', 50) if diag else 50,
        'prob_maligno': diag.get('prob_maligno', 50) if diag else 50,
        'datos':        {f: '0' for f in ['perimeter_worst','area_worst','radius_worst',
                        'concave_points_mean','concave_points_worst','concavity_mean',
                        'perimeter_mean','area_mean']},
    }
    return render(request, 'core/resultado.html', context)
