"""Inspección IA - UI del modelo comedores_v2 (YOLOv8n deteccion, 52 epochs
efectivos por EarlyStopping, dataset Roboflow: 222 fotos reales etiquetadas a
mano x3 aumentadas = 666 imagenes, 600 train / 66 val).

Rediseño visual (ver documentacion DamageAI_UI_UX.md) sobre el mismo flujo
funcional del POC original (FASE4/prueba_100/ui_streamlit.py): apunta a
runs/comedores_v2 y a las 3 clases reales (desprendimiento/rayado_o_golpe/
rotura). No cambia lógica de inferencia ni métricas -- solo presentación.

Métricas reales de este modelo (ver documentacion/ESTADO_PROYECTO.md y memoria
del sub-proyecto): mAP50-95 global 0.158 -- mejora ~5.5x sobre comedores_v1
(0.0286), sigue siendo demo, no produccion.

Nota de diseño: el mock de referencia (IADetection.png) dibuja el area de
dano como una elipse a mano. Eso NO es lo que el modelo entrega -- YOLO
devuelve cajas rectangulares (bounding boxes). El overlay de abajo dibuja el
rectangulo real (bbox_xyxy), no una elipse, para no prometer visualmente algo
que la deteccion no produce.

2026-09-21: la demo dejo de cargar el modelo (.pt) en el mismo proceso -- ahora
llama al servicio real (FASE4/api_ec2, endpoint /predict) por HTTP, igual que
cualquier otro consumidor de la API. Antes de este cambio, la demo y la API
eran 2 caminos de inferencia duplicados que nunca se probaban juntos.
"""
import glob
import io
import os

# pyrefly: ignore [missing-import]
import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont

API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.set_page_config(
    page_title='Inspección IA',
    layout='wide',
    initial_sidebar_state='expanded',
)

CLASES = ['desprendimiento', 'rayado_o_golpe', 'rotura']
NOMBRE_VISIBLE = {
    'desprendimiento': 'Desprendimiento',
    'rayado_o_golpe': 'Rayado o golpe',
    'rotura': 'Rotura',
}
MODEL_NAME = 'comedores_v2'
MODEL_EPOCHS = 52
MODEL_IMAGES = 666
MODEL_MAP = '0.158'

# ---------------------------------------------------------------------------
# CSS -- paleta y componentes segun documentacion/DamageAI_UI_UX.md
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg-main: #080F1C;
        --bg-sidebar: #0B1424;
        --surface-card: #0E1727;
        --surface-secondary: #141D2D;
        --control-bg: #101827;
        --primary: #526BFF;
        --primary-bright: #6678FF;
        --purple: #7567FF;
        --text-primary: #F4F7FC;
        --text-secondary: #A9B4C8;
        --text-muted: #718098;
        --border: #26334A;
        --border-subtle: rgba(148,163,184,0.10);
        --damage: #FF555C;
        --success: #37D391;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--bg-main) !important;
        color: var(--text-primary);
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; z-index: 999999; }
    [data-testid="stToolbar"] { z-index: 1000000; }
    #MainMenu, footer { visibility: hidden; }

    /* Fix: a 75-100 de zoom el sidebar y el contenido principal necesitaban
       scroll para verse completos. Se compactan paddings/margenes/tamanos de
       ambos paneles para que quepan en una sola pantalla sin scrollear; el
       overflow-y:auto queda solo como red de seguridad (por ejemplo si salen
       muchas detecciones a la vez), nunca como el comportamiento esperado.
       El header fijo de Streamlit (boton "Deploy") ya vive en stHeader/
       stToolbar con z-index alto, siempre por encima. */
    [data-testid="stAppViewContainer"] { overflow: hidden; }
    [data-testid="stSidebar"] {
        height: 100vh;
        overflow-y: auto;
        overflow-x: hidden;
    }
    [data-testid="stMain"] {
        height: 100vh;
        overflow-y: auto;
    }
    /* padding-top suficiente para no quedar debajo de la barra fija de
       Streamlit (boton "Deploy" incluido) -- si se baja mas de esto el
       header propio queda tapado por ella. */
    [data-testid="stMain"] .block-container {
        padding-top: 2.8rem !important;
        padding-bottom: 1rem !important;
        max-width: 100%;
    }
    [data-testid="stVerticalBlock"] { gap: 0.6rem !important; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.45rem !important; }
    [data-testid="stSidebar"] .block-container { padding-top: 0.2rem !important; padding-bottom: 0.5rem !important; }
    [data-testid="stSidebar"] [data-testid="stElementContainer"],
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0 !important;
    }
    /* Separa visualmente "Modelo activo" de "Detalles tecnicos" -- quedaban
       pegados el uno al otro. */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        margin-top: 10px !important;
    }
    /* El path completo del modelo (FASE4/.../best.pt) desborda el ancho del
       sidebar y generaba scroll horizontal -- se trunca con "..." en vez de
       permitir scroll lateral. */
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1424 0%, #091322 100%) !important;
        border-right: 1px solid rgba(148,163,184,0.08);
    }
    [data-testid="stSidebar"] > div { padding-top: 0; }

    .dai-logo-row { display:flex; align-items:center; gap:10px; padding: 0 4px 10px 4px; margin-top:-8px; }
    .dai-logo-box {
        width:36px; height:36px; border-radius:10px;
        background: rgba(82,107,255,0.12); border:1px solid rgba(82,107,255,0.35);
        display:flex; align-items:center; justify-content:center;
        font-size:13px; font-weight:700; color:#8DA0FF; letter-spacing:0.5px;
    }
    .dai-logo-name { font-size:17px; font-weight:700; color:var(--text-primary); line-height:1.1; }
    .dai-logo-name span { color:#6375FF; }
    .dai-logo-sub { font-size:10px; color:#8190AA; margin-top:1px; }

    .dai-nav-item {
        display:flex; align-items:center; gap:12px; height:34px; padding:0 12px;
        border-radius:7px; font-size:13px; color:#B1BCD0; margin-bottom:4px;
    }
    .dai-nav-item.active {
        background: rgba(82,107,255,0.35);
        border:1px solid rgba(104,123,255,0.20);
        color:#FFFFFF; font-weight:600;
    }

    .dai-sidebar-label {
        font-size:11px; font-weight:600; letter-spacing:0.6px; text-transform:uppercase;
        color:var(--text-muted); margin: 16px 0 6px 4px;
    }

    .dai-model-card {
        background: linear-gradient(145deg, rgba(20,34,57,0.90), rgba(12,24,43,0.95));
        border:1px solid rgba(82,107,255,0.25); border-radius:10px; padding:12px 14px; margin-top:16px;
    }
    .dai-model-card .k { font-size:10px; letter-spacing:0.6px; color:#7F8DA5; text-transform:uppercase; }
    .dai-model-card .v { font-size:14px; font-weight:700; color:var(--text-primary); margin:2px 0 1px 0; }
    .dai-model-card .d { font-size:11px; color:#8B98AF; }

    .dai-header {
        display:flex; justify-content:flex-end; align-items:center; gap:18px;
        padding: 0 0 8px 0; border-bottom:1px solid rgba(148,163,184,0.08); margin-bottom:10px;
    }
    .dai-status { display:flex; align-items:center; gap:8px; font-size:12px; color:var(--text-secondary); }
    .dai-status .dot { width:7px; height:7px; border-radius:50%; background:var(--success); display:inline-block; }
    .dai-avatar {
        width:38px; height:38px; border-radius:50%;
        background: linear-gradient(135deg, #526BFF, #6E5FFF);
        color:white; font-size:12px; font-weight:700;
        display:flex; align-items:center; justify-content:center;
    }

    .dai-eyebrow { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
    .dai-eyebrow .bar { width:28px; height:3px; border-radius:999px; background:var(--primary); }
    .dai-eyebrow span { font-size:10px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:#6D7FFF; }

    .dai-h1 { font-size:24px; font-weight:700; line-height:1.2; letter-spacing:-0.5px; color:var(--text-primary); max-width:720px; margin:0; }
    .dai-h1 .hl { background:linear-gradient(90deg,#5D72FF,#7868FF); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .dai-sub { font-size:13px; line-height:1.45; color:var(--text-secondary); max-width:650px; margin-top:6px; margin-bottom:22px; }

    .dai-card {
        background: var(--surface-card);
        border:1px solid rgba(148,163,184,0.10);
        border-radius:11px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.16);
        padding:12px 14px;
        margin-bottom:10px;
    }
    .dai-card-title { display:flex; align-items:center; gap:8px; font-size:14px; font-weight:600; color:var(--text-primary); margin-bottom:8px; }

    .dai-result-alert {
        background: rgba(255,85,92,0.09); border:1px solid rgba(255,85,92,0.48);
        border-radius:10px; padding:10px 12px; display:flex; gap:10px; align-items:flex-start;
    }
    .dai-result-alert.ok { background: rgba(55,211,145,0.09); border:1px solid rgba(55,211,145,0.48); }
    .dai-result-icon {
        width:38px; height:38px; min-width:38px; border-radius:50%; background:rgba(255,85,92,0.85);
        display:flex; align-items:center; justify-content:center; font-size:15px; font-weight:700; color:white;
    }
    .dai-result-icon.ok { background: rgba(55,211,145,0.85); }
    .dai-result-name { font-size:15px; font-weight:700; color:#FFFFFF; }
    .dai-result-desc { font-size:11.5px; color:#C0C8D6; line-height:1.35; margin-top:2px; }
    .dai-badge {
        background: rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.10);
        border-radius:999px; padding:4px 9px; font-size:11.5px; font-weight:700; color:#FFFFFF; white-space:nowrap;
    }

    .dai-meta-row { display:flex; justify-content:space-between; align-items:center; min-height:26px; border-bottom:1px solid rgba(148,163,184,0.08); }
    .dai-meta-row:last-child { border-bottom:none; }
    .dai-meta-label { font-size:11px; color:#7F8DA5; }
    .dai-meta-value { font-size:12px; color:#E5EAF3; font-weight:600; }

    .dai-analysis-text { font-size:11.5px; color:#9DA9BC; line-height:1.4; }

    .dai-empty { text-align:center; padding: 30px 10px; color:var(--text-secondary); }
    .dai-empty .ic { font-size:40px; margin-bottom:10px; }

    .dai-filename { display:flex; justify-content:space-between; font-size:12px; color:#A4B1C5; margin-top:10px; }

    div.stButton > button {
        background: linear-gradient(135deg, #526BFF, #675EFF) !important;
        color: white !important; border:none !important; border-radius:9px !important;
        font-weight:650 !important; font-size:14px !important; height:46px;
    }
    div.stButton > button:hover { transform: translateY(-1px); box-shadow:0 8px 24px rgba(82,107,255,0.22); }

    [data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.015) !important;
        border:1px dashed rgba(120,140,175,0.45) !important;
        border-radius:8px !important;
    }

    .stSlider label, .stSelectbox label, .stRadio label { color: var(--text-secondary) !important; font-size:13px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API de inferencia (FASE4/api_ec2) -- la demo ya no carga el modelo local
# ---------------------------------------------------------------------------
def api_health(api_url):
    try:
        r = requests.get(f'{api_url}/health', timeout=5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def api_predict(api_url, img, conf):
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='JPEG')
    buf.seek(0)
    r = requests.post(
        f'{api_url}/predict',
        params={'conf': conf},
        files={'file': ('imagen.jpg', buf, 'image/jpeg')},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="dai-logo-row">
            <div class="dai-logo-box">IA</div>
            <div>
                <div class="dai-logo-name">Inspección <span>IA</span></div>
                <div class="dai-logo-sub">Detección de daños en imágenes</div>
            </div>
        </div>
        <div class="dai-nav-item active">Inicio</div>
        <div class="dai-nav-item">Historial</div>
        <div class="dai-nav-item">Configuración</div>
        <div class="dai-nav-item">Ayuda</div>
        <div class="dai-sidebar-label">Configuración del modelo</div>
        """,
        unsafe_allow_html=True,
    )
    api_url = st.text_input('URL de la API', API_URL)
    conf = st.slider('Confianza mínima', 0.01, 0.95, 0.05, 0.01)
    st.markdown(
        f"""
        <div class="dai-model-card">
            <div class="k">Modelo activo</div>
            <div class="v">{MODEL_NAME}</div>
            <div class="d">{MODEL_EPOCHS} epochs · {MODEL_IMAGES} imágenes</div>
            <div class="d">mAP50-95: {MODEL_MAP}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander('Detalles técnicos'):
        st.caption(
            'Modelo entrenado con 222 fotos reales etiquetadas a mano (x3 aumentadas = 666), '
            '52 epochs (EarlyStopping, mejor resultado en epoch 37). '
            'mAP50-95 global 0.158 (desprendimiento 0.224, rotura 0.165, rayado_o_golpe 0.085) — '
            'todavía demo, no apto para producción. Umbral bajo por defecto para poder ver detecciones.'
        )

api_disponible = api_health(api_url)
if not api_disponible:
    st.sidebar.markdown(
        f'<div class="dai-model-card" style="border-color:#FF555C">'
        f'<div class="k">API no disponible</div>'
        f'<div class="d">No se pudo conectar a {api_url}. '
        f'Levanta FASE4/api_ec2 (uvicorn/docker) antes de analizar imagenes.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dai-header">
        <div class="dai-status"><span class="dot"></span>Aplicación local en ejecución (localhost:8501)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dai-eyebrow"><span class="bar"></span><span>DETECCIÓN DE DAÑOS</span></div>
    <h1 class="dai-h1">Analiza tus imágenes y detecta daños con <span class="hl">inteligencia artificial</span></h1>
    <div class="dai-sub">Sube una imagen y nuestro modelo analizará el contenido para identificar
    y clasificar el tipo de daño, con el nivel de confianza real que entrega el modelo.</div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers de overlay / metadata (sobre datos reales del modelo, nada inventado)
# ---------------------------------------------------------------------------
REGIONES_X = ['izquierda', 'central', 'derecha']
REGIONES_Y = ['superior', '', 'inferior']


def _region_de_caja(cx_ratio, cy_ratio):
    col = 0 if cx_ratio < 1 / 3 else (1 if cx_ratio < 2 / 3 else 2)
    fila = 0 if cy_ratio < 1 / 3 else (1 if cy_ratio < 2 / 3 else 2)
    partes = [p for p in (REGIONES_Y[fila], REGIONES_X[col]) if p]
    return 'Área ' + ' '.join(partes) if partes else 'Área central'


def _cargar_fuente(size):
    for ruta in ('C:/Windows/Fonts/segoeuib.ttf', 'C:/Windows/Fonts/arialbd.ttf'):
        try:
            return ImageFont.truetype(ruta, size)
        except Exception:
            continue
    return ImageFont.load_default()


def dibujar_overlay(img, api_detecciones):
    """Dibuja el bounding box REAL entregado por la API (rectángulo), no una
    elipse decorativa -- esto es justo lo que el endpoint /predict produce.
    api_detecciones: lista de {'clase', 'confianza', 'bbox_xyxy'} tal como
    la devuelve FASE4/api_ec2/server.py."""
    anotada = img.convert('RGB').copy()
    draw = ImageDraw.Draw(anotada)
    fuente = _cargar_fuente(16)
    detecciones = []
    w, h = anotada.size
    for det in api_detecciones:
        x1, y1, x2, y2 = [float(v) for v in det['bbox_xyxy']]
        clase = det['clase']
        score = float(det['confianza'])
        nombre = NOMBRE_VISIBLE.get(clase, clase)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 85, 92), width=3)
        etiqueta = nombre
        tw = draw.textlength(etiqueta, font=fuente)
        pad = 6
        ly1 = max(0, y1 - 26)
        draw.rectangle([x1, ly1, x1 + tw + pad * 2, ly1 + 24], fill=(255, 85, 92))
        draw.text((x1 + pad, ly1 + 3), etiqueta, fill=(255, 255, 255), font=fuente)
        area_pct = ((x2 - x1) * (y2 - y1)) / (w * h) * 100
        region = _region_de_caja(((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h)
        detecciones.append({
            'clase': clase,
            'nombre': nombre,
            'confianza': score,
            'region': region,
            'area_pct': area_pct,
        })
    detecciones.sort(key=lambda d: d['confianza'], reverse=True)
    return anotada, detecciones


# ---------------------------------------------------------------------------
# Estado / reset
# ---------------------------------------------------------------------------
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0


def _analizar_otra_imagen():
    st.session_state.uploader_key += 1


# ---------------------------------------------------------------------------
# Layout principal: 1.5fr / 1fr
# ---------------------------------------------------------------------------
col_izq, col_der = st.columns([1.5, 1], gap='large')

with col_izq:
    with st.container(border=True):
        st.markdown('<div class="dai-card-title">Cargar imagen</div>', unsafe_allow_html=True)
        fuente = st.radio(
            'Fuente de imagen', ['Subir imagen', 'Imagen del set de validación'],
            horizontal=True, label_visibility='collapsed',
        )

        img = None
        nombre_archivo = None
        tamano_archivo = None

        if fuente == 'Subir imagen':
            up = st.file_uploader(
                'Arrastra y suelta tu imagen aquí (JPG, PNG · Máx. 10 MB)',
                type=['jpg', 'jpeg', 'png'],
                key=f'uploader_{st.session_state.uploader_key}',
            )
            if up:
                img = Image.open(up)
                nombre_archivo = up.name
                tamano_archivo = f'{up.size / 1024 / 1024:.1f} MB'
        else:
            val_imgs = sorted(glob.glob('FASE4/comedores_v1/dataset_yolo_v2/val/images/*'))
            if val_imgs:
                elegida = st.selectbox('Elegir imagen', val_imgs, format_func=os.path.basename)
                img = Image.open(elegida)
                nombre_archivo = os.path.basename(elegida)
                tamano_archivo = f'{os.path.getsize(elegida) / 1024 / 1024:.1f} MB'
            else:
                st.info('No hay imágenes en dataset_yolo_v2/val/images/.')

    detecciones = []
    anotada = None

    if img is not None:
        if not api_disponible:
            st.error(f'No se pudo conectar a la API ({api_url}). Verifica que este corriendo.')
        else:
            with st.spinner('Analizando imagen...'):
                try:
                    resultado_api = api_predict(api_url, img, conf)
                except requests.exceptions.RequestException as e:
                    st.error(f'Error llamando a la API: {e}')
                    resultado_api = {'detecciones': []}
            if resultado_api['detecciones']:
                anotada, detecciones = dibujar_overlay(img, resultado_api['detecciones'])
            else:
                anotada = img

        with st.container(border=True):
            st.markdown('<div class="dai-card-title">Vista previa</div>', unsafe_allow_html=True)
            st.image(anotada, use_container_width=True)
            st.markdown(
                f'<div class="dai-filename"><span>{nombre_archivo}</span><span>{tamano_archivo}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        with st.container(border=True):
            st.markdown(
                """
                <div class="dai-empty">
                    <div style="font-weight:600; color:var(--text-primary); margin-bottom:4px;">Sube una imagen para comenzar</div>
                    <div style="font-size:13px;">El modelo analizará la superficie y detectará posibles daños.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with col_der:
    with st.container(border=True):
        st.markdown('<div class="dai-card-title">Resultado del análisis</div>', unsafe_allow_html=True)

        if img is None:
            st.markdown(
                '<div class="dai-analysis-text">Aún no se ha analizado ninguna imagen.</div>',
                unsafe_allow_html=True,
            )
        elif not detecciones:
            st.markdown(
                """
                <div class="dai-result-alert ok">
                    <div class="dai-result-icon ok">OK</div>
                    <div>
                        <div class="dai-result-name">Sin daño detectado</div>
                        <div class="dai-result-desc">El modelo no encontró regiones por encima del umbral de confianza configurado.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            principal = detecciones[0]
            if len(detecciones) > 1:
                st.markdown(
                    f'<div style="font-size:13px; color:var(--text-secondary); margin-bottom:10px;">{len(detecciones)} daños detectados</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"""
                <div class="dai-result-alert">
                    <div class="dai-result-icon">!</div>
                    <div style="flex:1;">
                        <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                            <div class="dai-result-name">{principal['nombre']}</div>
                            <div class="dai-badge">{principal['confianza']*100:.1f}%</div>
                        </div>
                        <div class="dai-result-desc">El modelo ha detectado {principal['nombre'].lower()} en la superficie de la imagen.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            filas = ''.join(
                f"""
                <div class="dai-meta-row"><span class="dai-meta-label">{d['nombre']}</span><span class="dai-meta-value">{d['confianza']*100:.1f}%</span></div>
                """
                for d in detecciones
            ) if len(detecciones) > 1 else f"""
                <div class="dai-meta-row"><span class="dai-meta-label">Tipo de daño</span><span class="dai-meta-value">{principal['nombre']}</span></div>
                <div class="dai-meta-row"><span class="dai-meta-label">Confianza</span><span class="dai-meta-value">{principal['confianza']*100:.1f}%</span></div>
                <div class="dai-meta-row"><span class="dai-meta-label">Región detectada</span><span class="dai-meta-value">{principal['region']}</span></div>
                <div class="dai-meta-row"><span class="dai-meta-label">Tamaño del daño</span><span class="dai-meta-value">~{principal['area_pct']:.1f}% del área</span></div>
            """
            st.markdown(filas, unsafe_allow_html=True)

            st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
            clases_texto = ', '.join(sorted({d['nombre'].lower() for d in detecciones}))
            st.markdown(
                f"""
                <div class="dai-card-title" style="font-size:14px;">Análisis del modelo</div>
                <div class="dai-analysis-text">
                El modelo evaluó la imagen completa y encontró {len(detecciones)} región(es) con
                patrones visuales asociados a: {clases_texto}. La confianza mostrada corresponde
                directamente a la salida del modelo (comedores_v2, YOLOv8n) — no es un valor
                editado ni una estimación manual.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.button('Analizar otra imagen', on_click=_analizar_otra_imagen, use_container_width=True)
