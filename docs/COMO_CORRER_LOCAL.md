# Cómo correr la demo en local

Requiere que el backend (`detecciondanosiavback`) esté corriendo primero
-- esta demo no funciona por su cuenta, siempre llama a la API real.

```bash
pip install -r requirements.txt
streamlit run ui_streamlit.py
```

Por defecto apunta a `http://localhost:8000`. Para apuntar a otra URL
(otro entorno, otro puerto):

```bash
API_URL=http://<otra-url>:8000 streamlit run ui_streamlit.py
```

También se puede cambiar la URL desde la barra lateral de la app, sin
reiniciar el proceso.
